"""
SpatialMap experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierSpatialMapMonitor which:
  - Verifies the model's directional claims during <think> via side-streams
  - Injects a structured step format after </think> (no meta-prompt needed)
  - Verifies each step as the model fills in the structured template
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import numpy as np
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierSpatialMapMonitor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "microsoft/Phi-4-reasoning"
# =================================================


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


def get_output_dirs(main_model: str, base_dir: str = "../../Outputs_TTS/SpatialMapResults"):
    """Create and return output directory paths based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def get_question_type(idx: int) -> str:
    """Determine question type based on index range.
    
    Dataset structure (1500 examples total):
    - 0-499: Q0 (direction finding)
    - 500-999: Q1 (object finding)
    - 1000-1499: Q2 (counting)
    """
    if idx < 500:
        return "direction"
    elif idx < 1000:
        return "object"
    else:
        return "counting"


def build_simple_prompt(example):
    """Build a simple user prompt from the spatial map example.

    No system / meta prompt is used — the structured step format is
    injected by the monitor after ``</think>``.
    """
    description = str(example.get("prompt", ""))
    # Trim trailing boiler-plate instructions that the dataset appends
    description_trimmed = description[:-143] if len(description) > 143 else description
    return description_trimmed


def extract_solution(text: str) -> str:
    """Extract the boxed answer from the response (after </think>)."""
    if "</think>" in text:
        answer_section = text.split("</think>")[-1]
    else:
        answer_section = text
    
    # Strip injected <format>...</format> template blocks so we don't
    # accidentally match the placeholder \boxed{LETTER} from the template.
    answer_section = re.sub(r'<format>.*?</format>', '', answer_section, flags=re.DOTALL)
    
    matches = re.findall(r'\\boxed\{([^}]*)\}', answer_section)
    if matches:
        return matches[-1].strip()
    
    match = re.search(r'(?:answer|Answer)[:\s]+([A-D])', answer_section)
    if match:
        return match.group(1).strip()
    
    return None


def count_tokens(text: str, tokenizer) -> int:
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def init_llm_server(model_name, max_tokens=20000, port=8000):
    """Initialize LLM server configuration."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def save_output(idx: int, output: str, output_dir: str):
    """Save output to file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"output_{idx}.txt")
    with open(filepath, 'w') as f:
        f.write(output)
    logger.info(f"Saved output to {filepath}")


def save_prompt(idx, prompt_with_answer, reason_dir):
    """Save reasoning trace to file."""
    os.makedirs(reason_dir, exist_ok=True)
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)
    logger.info(f"Saved reasoning trace to {filename}")


def evaluate_spatialmap_answer(answer, options, ground_truth):
    """
    Evaluate a SpatialMap MCQ answer and return (is_correct, extracted_answer, message).
    
    Args:
        answer: Raw model output
        options: Dictionary mapping option letters (A/B/C/D) to their values
        ground_truth: The correct answer value
        
    Returns:
        Tuple of (is_correct, extracted_answer, message)
    """
    sol = extract_solution(answer)
    gt_sol = str(ground_truth).strip()
    
    if not sol:
        return False, None, "No expression found"
    
    sol = sol.strip()
    
    # Case 1: LLM returned option letter (A/B/C/D)
    if sol in options:
        if options[sol] == gt_sol:
            return True, sol, f"Correct: option {sol} -> {options[sol]}"
        else:
            return False, sol, f"Incorrect: expected '{gt_sol}', got '{options[sol]}' (option {sol})"
    
    # Case 2: LLM returned the actual answer text
    # First check if sol matches ground truth directly
    if sol.lower() == gt_sol.lower():
        return True, sol, f"Correct: answer text matches ground truth: {sol}"
    
    # Check if sol matches any option value
    for opt_letter, opt_value in options.items():
        if sol.lower() == opt_value.lower():
            if opt_value == gt_sol:
                return True, sol, f"Correct: answer text {sol} (option {opt_letter})"
            else:
                return False, sol, f"Incorrect: expected '{gt_sol}', got '{opt_value}' (option {opt_letter})"
    
    return False, sol, f"Solution '{sol}' not found in options or ground truth"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SpatialMap experiments with StepVerifierSpatialMapMonitor")
    parser.add_argument("--model", type=str, default=MAIN_MODEL,
                        help="Model name for generation")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to run (e.g., '0,100,200')")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=1500, help="End index")
    parser.add_argument("--num_examples", "-n", type=int, default=None,
                        help="Number of examples to run (overrides start/end)")
    parser.add_argument("--max_corrections", type=int, default=5,
                        help="Maximum number of correction attempts per example")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--newline_threshold", type=int, default=20,
                        help="Number of \\n\\n in thinking before triggering side verification")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of \\n\\n to skip before starting side-chain verification (warmup period)")
    args = parser.parse_args()

    logger.info(f"Thinking-phase verification: always on")
    logger.info(f"  Newline threshold: {args.newline_threshold}")
    logger.info(f"  Warmup: {args.warmup}")
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load dataset (spatial_map_text_only has 1500 examples)
    dataset = load_dataset("microsoft/VISION_LANGUAGE", 'spatial_map_text_only', split='val')
    
    # Setup LLM server
    llm_server = init_llm_server(args.model, port=args.port)
    
    # Load tokenizer for accurate token counting
    logger.info(f"Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")
    
    # Setup output directory
    output_dirs = get_output_dirs(args.model)
    reason_dir = output_dirs["reasoning"]
    
    # Determine indices
    max_idx = len(dataset) - 1
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    elif args.num_examples:
        # Sample evenly across all 1500 examples (0-1499)
        indices = np.linspace(0, min(max_idx, 1499), args.num_examples, dtype=int)
    else:
        indices = range(args.start, min(args.end, max_idx + 1))
    
    # Stats tracking
    results = []
    total_correct = 0
    total_examples = 0
    total_reasoning_tokens = 0
    num_attempted = 0  # examples where a \boxed{} answer was produced (not "no solution")
    reasoning_token_counts = []
    per_example_results = []  # list of dicts for CSV
    
    # Per-type stats
    stats_by_type = {
        "direction": {"total": 0, "correct": 0},
        "object": {"total": 0, "correct": 0},
        "counting": {"total": 0, "correct": 0},
    }
    
    for idx in indices:
        example = dataset[idx]
        user_prompt = build_simple_prompt(example)
        if str(example.get("ground_truth", "")).strip() == "Q4":
            target_options = ["A", "B"]
        else:
            target_options = ["A", "B", "C", "D"] 
        keys = "|".join(map(re.escape, target_options))
        pattern = r'\b([A-D])\.\s*(.*?)(?=\s*[A-D]\.|$)'
        raw = re.findall(pattern, user_prompt, flags=re.DOTALL)

        options = {k: v.strip().rstrip(".") for k, v in raw}
        
        # Determine question type
        question_type = get_question_type(idx)
        
        # Build prompt with Phi-4-reasoning system prompt
        phi_system_prompt = (
            "You are Phi, a language model trained by Microsoft to help users. "
            "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process "
            "before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle "
            "of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop "
            "well-considered thinking process. Please structure your response into two main sections: Thought and Solution "
            "using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, "
            "detail your reasoning process in steps. Each step should include detailed considerations such as analysing "
            "questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, "
            "refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, "
            "explorations, and reflections from the Thought section, systematically present the final solution that you "
            "deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed "
            "to reach the conclusion. Now, try to solve the following question through the above guidelines."
        )
        full_prompt = f"<|im_start|>system<|im_sep|>\n{phi_system_prompt}<|im_end|>\n<|im_start|>user<|im_sep|>\n{user_prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n<think>\n"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Example {idx} ({question_type})")
        logger.info(f"{'='*60}")
        
        # Always use ThinkingPhaseStepVerifierSpatialMapMonitor:
        # Phase 1 — verifies during <think> via side-streams
        # Phase 2a — injects structured step format after </think>
        # Phase 2b — verifies structured output as model fills it in
        monitor = ThinkingPhaseStepVerifierSpatialMapMonitor(
            name="spatialmap_thinking_verifier",
            problem_text=user_prompt,
            llm_server=llm_server,
            prompt=full_prompt,
            newline_threshold=args.newline_threshold,
            max_corrections=args.max_corrections,
            answer_start_token="</think>",
            warmup_newlines=args.warmup,
        )
        
        logger.info(f"Z3 solver initialized with {len(monitor.z3_solver.parsed_relations)} relations")
        
        # Run with stream_completion
        try:
            answer = asyncio.run(stream_completion(
                full_prompt,
                llm_server=llm_server,
                monitors=(monitor,),
                add_delay=False,
                termination_requires_validation=False,
                async_execution=True
            ))
        except Exception as e:
            logger.error(f"Error running example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Save reasoning trace
        save_prompt(int(idx), answer, reason_dir)
        logger.info(f"Raw final output:\n{answer}")

        # Count generated tokens
        reasoning_tokens = count_tokens(answer, tokenizer)
        total_reasoning_tokens += reasoning_tokens
        reasoning_token_counts.append(reasoning_tokens)
        logger.info(f"Generated tokens in this example: {reasoning_tokens}")
        
        # Evaluate the answer
        gt_sol = str(example.get("ground_truth", "")).strip()
        is_correct, extracted_answer, message = evaluate_spatialmap_answer(answer, options, gt_sol)
        
        # "attempted" = model produced a real \boxed{} answer (not "no solution")
        attempted = (extracted_answer is not None and extracted_answer.strip().lower() != "no solution")
        if attempted:
            num_attempted += 1
        
        if extracted_answer:
            logger.info(f"Extracted answer: {extracted_answer}")
        logger.info(message)
        
        if is_correct:
            total_correct += 1
            stats_by_type[question_type]["correct"] += 1
            
        total_examples += 1
        stats_by_type[question_type]["total"] += 1
        
        # Log result
        result = {
            'idx': int(idx),
            'question_type': question_type,
            'correct': is_correct,
            'attempted': attempted,
            'sol': extracted_answer,
            'gt': gt_sol,
            'reasoning_tokens': reasoning_tokens,
            'num_relations': len(monitor.z3_solver.parsed_relations),
            'verified_claims': len(monitor.verified_claims),
        }
        results.append(result)
        
        per_example_results.append({
            "index": int(idx),
            "question_type": question_type,
            "correct": is_correct,
            "attempted": attempted,
            "sol": extracted_answer if extracted_answer else "",
            "gt": gt_sol,
            "tokens": reasoning_tokens,
            "num_relations": len(monitor.z3_solver.parsed_relations),
            "verified_claims": len(monitor.verified_claims),
            "message": message,
        })
        
        logger.info(f"Result: sol={extracted_answer}, gt={gt_sol}, correct={is_correct}, attempted={attempted}")
        logger.info(f"Verified claims: {len(monitor.verified_claims)}")
        logger.info(f"Reasoning tokens: {reasoning_tokens}")
    
    # Compute final metrics
    accuracy = total_correct / total_examples if total_examples > 0 else 0
    soundness = total_correct / num_attempted if num_attempted > 0 else 0  # correct / attempted
    avg_reasoning_tokens = total_reasoning_tokens / total_examples if total_examples > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Correct: {total_correct}")
    logger.info(f"Attempted (produced \\boxed answer): {num_attempted}/{total_examples}")
    logger.info(f"Accuracy: {accuracy:.4f} ({total_correct}/{total_examples})")
    logger.info(f"Soundness: {soundness:.4f} ({total_correct}/{num_attempted})")
    logger.info(f"Total reasoning tokens: {total_reasoning_tokens}")
    logger.info(f"Avg reasoning tokens: {avg_reasoning_tokens:.1f}")
    
    # Per-type breakdown
    logger.info(f"\nPer-type breakdown:")
    for qtype, stats in stats_by_type.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            logger.info(f"  {qtype}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    print(f"\nFinal Accuracy: {total_correct}/{total_examples} ({accuracy:.2%})")
    print(f"Soundness: {total_correct}/{num_attempted} ({soundness:.2%})")
    print(f"Average Reasoning Tokens: {avg_reasoning_tokens:.2f}")
    print(f"Total Reasoning Tokens: {total_reasoning_tokens}")
    
    # Save per-example CSV
    csv_file = os.path.join(output_dirs["csv_saved"], f"results_{total_examples}examples.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["index", "question_type", "correct", "attempted", "sol", "gt", "tokens", "num_relations", "verified_claims", "message"])
        writer.writeheader()
        writer.writerows(per_example_results)
    logger.info(f"Per-example CSV saved to {csv_file}")
    
    # Save summary
    summary = {
        'model': args.model,
        'total_examples': total_examples,
        'correct': total_correct,
        'attempted': num_attempted,
        'accuracy': accuracy,
        'soundness': soundness,
        'total_reasoning_tokens': total_reasoning_tokens,
        'avg_reasoning_tokens': avg_reasoning_tokens,
        'max_corrections': args.max_corrections,
        'stats_by_type': stats_by_type,
        'results': results,
    }
    
    summary_path = os.path.join(output_dirs["base"], "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_path}")
    
    # Save results summary to a text file
    results_file = os.path.join(output_dirs["base"], f"EAT_{total_examples}examples_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"SpatialMap Step Verification Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Number of Examples: {total_examples}\n")
        f.write(f"Max Corrections: {args.max_corrections}\n")
        f.write(f"Newline Threshold: {args.newline_threshold}\n")
        f.write(f"Warmup: {args.warmup}\n")
        f.write(f"\n")
        f.write(f"Results:\n")
        f.write(f"---------\n")
        f.write(f"Correct: {total_correct}/{total_examples}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Attempted (produced \\boxed answer): {num_attempted}/{total_examples}\n")
        f.write(f"Soundness (correct/attempted): {soundness:.2%}\n\n")
        f.write(f"Per-type Breakdown:\n")
        f.write(f"---------------------------\n")
        for qtype, stats in stats_by_type.items():
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                f.write(f"  {qtype}: {acc:.2%} ({stats['correct']}/{stats['total']})\n")
        f.write(f"\nToken Statistics:\n")
        f.write(f"---------------------------\n")
        f.write(f"Total Tokens: {total_reasoning_tokens}\n")
        f.write(f"Average Tokens: {avg_reasoning_tokens:.2f}\n")
        if reasoning_token_counts:
            f.write(f"Median Tokens: {float(np.median(reasoning_token_counts)):.0f}\n")
            f.write(f"Min Tokens: {min(reasoning_token_counts)}\n")
            f.write(f"Max Tokens: {max(reasoning_token_counts)}\n")
            f.write(f"Std Dev: {np.std(reasoning_token_counts):.2f}\n")
    
    logger.info(f"Results saved to {results_file}")
    print(f"Results saved to {results_file}")
