"""
Maze experiment with thinking-phase step verification.

Uses ThinkingPhaseStepVerifierMazeMonitor which:
  - Verifies the model's traced path during <think> via side-streams
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

from datasets import load_dataset
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import ThinkingPhaseStepVerifierMazeMonitor
from interwhen.utils.maze_verifier import parse_maze_from_prompt

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "Qwen/QwQ-32B"
# =================================================

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Walk up to find the repo root (contains pyproject.toml), output to its parent
_dir = _SCRIPT_DIR
while _dir != os.path.dirname(_dir) and not os.path.isfile(os.path.join(_dir, "pyproject.toml")):
    _dir = os.path.dirname(_dir)
_OUTPUT_ROOT = os.path.dirname(_dir)


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


def get_output_dirs(main_model: str, base_dir: str = None):
    """Create and return output directory paths based on model name."""
    if base_dir is None:
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "MazeResults")
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

def remove_last_paragraph(s: str) -> str:
    return s[:-143]

def build_prompt_from_example(example): #(original prompt config)

    pre_prompt = "You are an expert problem solver. Carefully read the following multiple-choice question and think through the solution step-by-step before providing your final answer. Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
    description = example.get("prompt")
    description = str(description)
    description = remove_last_paragraph(description)
    return pre_prompt, description


def extract_solution_mcq(text):
    """Extract MCQ solution from model output."""
    patterns = [
        r"\\boxed\{([^}]*)\}",
        r"boxed\{([^}]*)\}",
        r"\*\*([A-D])\*\*",
        r"answer[:\s]*([A-D])",
        r"(?:^|\n)([A-D])(?:\s|$|\.)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            expr = matches[-1].strip()
            choice_match = re.search(r"\b([ABCD])\b", expr, flags=re.IGNORECASE)
            if choice_match:
                return choice_match.group(1).upper()
    return None


def count_tokens(text: str, tokenizer) -> int:
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def init_llm_server(modelname, max_tokens=32768, port=8000):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "do_sample": True,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def save_prompt(idx, prompt_with_answer, reason_dir):
    """Save reasoning trace to file."""
    os.makedirs(reason_dir, exist_ok=True)
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)
    logger.info(f"Saved reasoning trace to {filename}")


def get_log_filename(main_model: str, num_examples: int, base_dir: str = None) -> str:
    """Generate log filename based on model name."""
    if base_dir is None:
        base_dir = os.path.join(_OUTPUT_ROOT, "Outputs_TTS", "MazeResults")
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")


def evaluate_mcq_answer(answer, options, ground_truth):
    sol = extract_solution_mcq(answer)
    gt_sol = str(ground_truth).strip()
    if not sol:
        return False, None, "No expression found"
    sol = sol.strip()
    if sol in options:
        if options[sol] == gt_sol:
            return True, sol, f"Correct: option {sol} -> {options[sol]}"
        return False, sol, f"Incorrect: expected '{gt_sol}', got '{options[sol]}' (option {sol})"
    if sol.lower() == gt_sol.lower():
        return True, sol, f"Correct: answer text matches ground truth: {sol}"
    for opt_letter, opt_value in options.items():
        if sol.lower() == opt_value.lower():
            if opt_value == gt_sol:
                return True, sol, f"Correct: answer text {sol} (option {opt_letter})"
            return False, sol, f"Incorrect: expected '{gt_sol}', got '{opt_value}' (option {opt_letter})"
    return False, sol, f"Solution '{sol}' not found in options or ground truth"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run maze experiments with step verification")
    parser.add_argument("--model", type=str, default=MAIN_MODEL,
                        help="Model name for generation")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to run (e.g., '3000,3500,4000')")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=10, help="End index")
    parser.add_argument("--num_examples", "-n", type=int, default=None,
                        help="Number of examples to run (overrides start/end)")
    parser.add_argument("--max_corrections", type=int, default=5,
                        help="Maximum number of correction attempts per example")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--newline_threshold", type=int, default=20,
                        help="Number of \\n in thinking before triggering side verification")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of \\n to skip before starting side-chain verification (warmup period)")
    args = parser.parse_args()

    logger.info(f"Thinking-phase verification: always on")
    logger.info(f"  Newline threshold: {args.newline_threshold}")
    logger.info(f"  Warmup: {args.warmup}")
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load dataset
    dataset = load_dataset("microsoft/VISION_LANGUAGE", 'maze_text_only', split='val')
    
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
    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    elif args.num_examples:
        # Use 1499 as endpoint (1500 is out of bounds since dataset size is 1500)
        indices = np.linspace(0, 1499, args.num_examples, dtype=int)
    else:
        indices = range(args.start, args.end)
    
    # Stats tracking
    results = []
    total_correct = 0
    total_examples = 0
    total_reasoning_tokens = 0
    num_attempted = 0  # examples where a \boxed{} answer was produced
    reasoning_token_counts = []
    per_example_results = []  # list of dicts for CSV
    
    for idx in indices:
        example = dataset[idx]
        pre_prompt, user_prompt = build_prompt_from_example(example)
        if str(example.get("ground_truth", "")).strip() == "Q4":
            target_options = ["A", "B"]
        else:
            target_options = ["A", "B", "C", "D"] 
        keys = "|".join(map(re.escape, target_options))
        pattern = rf'\b({keys})\.\s*([A-Za-z0-9]+)\b'
        options = dict(re.findall(pattern, user_prompt))
        
        full_prompt = f"<|im_start|>system\n{pre_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Parse maze from prompt
        grid, start_pos, exit_pos = parse_maze_from_prompt(user_prompt)
        
        if not grid or not start_pos or not exit_pos:
            logger.error(f"Could not parse maze for example {idx}")
            continue
        
        # Detect question type from prompt (auto-detection)
        question_type = ThinkingPhaseStepVerifierMazeMonitor.detect_question_type(user_prompt)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Example {idx} ({question_type})")
        logger.info(f"Maze: S={start_pos}, E={exit_pos}, grid={len(grid)}x{len(grid[0]) if grid else 0}")
        logger.info(f"{'='*60}")
        
        # Always use ThinkingPhaseStepVerifierMazeMonitor:
        # Phase 1 — verifies during <think> via side-streams
        # Phase 2a — injects structured step format after </think>
        # Phase 2b — verifies structured output as model fills it in
        monitor = ThinkingPhaseStepVerifierMazeMonitor(
            name="maze_thinking_verifier",
            grid=grid,
            start_pos=start_pos,
            exit_pos=exit_pos,
            llm_server=llm_server,
            prompt=full_prompt,
            question_type=question_type,
            newline_threshold=args.newline_threshold,
            max_corrections=args.max_corrections,
            answer_start_token="</think>",
            warmup_newlines=args.warmup,
        )
        
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
        
        gt_sol = str(example.get("ground_truth", "")).strip()
        is_correct, extracted_answer, message = evaluate_mcq_answer(answer, options, gt_sol)
        
        # "attempted" = model produced a real \boxed{} answer (not "no solution")
        attempted = (extracted_answer is not None and extracted_answer.strip().lower() != "no solution")
        if attempted:
            num_attempted += 1
        
        if extracted_answer:
            logger.info(f"Extracted answer: {extracted_answer}")
        logger.info(message)
        
        if is_correct:
            total_correct += 1
        
        total_examples += 1
        # Log result
        result = {
            'idx': int(idx),  # Convert numpy int64 to Python int
            'question_type': question_type,
            'correct': is_correct,
            'attempted': attempted,
            'sol': extracted_answer,
            'gt': gt_sol,
            'reasoning_tokens': reasoning_tokens,
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
            "message": message,
        })
        
        logger.info(f"Result: sol={extracted_answer}, gt={gt_sol}, correct={is_correct}, attempted={attempted}")
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
    
    print(f"\nFinal Accuracy: {total_correct}/{total_examples} ({accuracy:.2%})")
    print(f"Soundness: {total_correct}/{num_attempted} ({soundness:.2%})")
    print(f"Average Reasoning Tokens: {avg_reasoning_tokens:.2f}")
    print(f"Total Reasoning Tokens: {total_reasoning_tokens}")
    
    # Save per-example CSV
    csv_file = os.path.join(output_dirs["csv_saved"], f"results_{total_examples}examples.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["index", "question_type", "correct", "attempted", "sol", "gt", "tokens", "message"])
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
        'results': results,
    }
    
    summary_path = os.path.join(output_dirs["base"], "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_path}")
    
    # Save results summary to a text file
    results_file = os.path.join(output_dirs["base"], f"EAT_{total_examples}examples_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Maze Step Verification Results\n")
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
        f.write(f"Token Statistics:\n")
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