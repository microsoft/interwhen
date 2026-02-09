"""
SpatialMap experiment with step-by-step verification using StepVerifierSpatialMapMonitor.

Uses the new monitor-based architecture that integrates with stream_completion.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import numpy as np
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import StepVerifierSpatialMapMonitor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
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
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def remove_last_paragraph(s: str) -> str:
    """Remove the last instruction paragraph from the prompt."""
    return s[:-143] if len(s) > 143 else s


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


def build_meta_prompt_from_example(example):
    """Build prompt with structured output format instructions."""
    
    # Get the description
    description = example.get("prompt")
    description = str(description)
    description = remove_last_paragraph(description)
    
    pre_prompt = """You are a spatial reasoning expert. Given a description of objects on a map and their relative positions, analyze the spatial relationships step by step.

CRITICAL INSTRUCTION: DO NOT use abbreviations or initials for entity names. Always use the COMPLETE FULL NAME of each entity exactly as given in the problem. For example, write "Police Supply Store" not "PSS" or "PS".

DIRECTION DEFINITIONS (Diagonal Directions):
- Northwest = up and to the left (row decreases, col decreases)
- Northeast = up and to the right (row decreases, col increases)  
- Southwest = down and to the left (row increases, col decreases)
- Southeast = down and to the right (row increases, col increases)

CARDINAL DIRECTIONS (for questions asking about North/South/East/West):
- North = directly up - requires BOTH Northwest AND Northeast relationships to be confirmed
- South = directly down - requires BOTH Southwest AND Southeast relationships to be confirmed
- West = directly left - requires BOTH Northwest AND Southwest relationships to be confirmed
- East = directly right - requires BOTH Northeast AND Southeast relationships to be confirmed

IMPORTANT: In this dataset, only diagonal relationships (NW/NE/SW/SE) are given. An object can ONLY be in a pure cardinal direction (N/S/E/W) if BOTH required diagonal relationships exist.

IMPORTANT RULES:
- Directions are TRANSITIVE: If A is Northwest of B, and B is Northwest of C, then A is Northwest of C.
- Directions are REVERSIBLE: If A is Northwest of B, then B is Southeast of A.
- Opposite pairs: Northwest ↔ Southeast, Northeast ↔ Southwest

STRUCTURED OUTPUT FORMAT:

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 1: Direction Finding (Q0)
═══════════════════════════════════════════════════════════════════════════════

Map Description:
Police Supply Store is in the map. Narwhal's Novelties is to the Northwest of Police Supply Store. Coral Crafts is to the Northwest of Narwhal's Novelties. Coral Crafts is to the Northwest of Police Supply Store. Planetarium Prints is to the Southeast of Coral Crafts. Planetarium Prints is to the Northeast of Police Supply Store. Oz Oddities is to the Southwest of Planetarium Prints. Oz Oddities is to the Southwest of Police Supply Store. Ice Queen Ice Cream is to the Northwest of Planetarium Prints. Ice Queen Ice Cream is to the Southeast of Coral Crafts.

Question: In which direction is Planetarium Prints relative to Police Supply Store?

### Final Answer

>>> STEP 1: PARSE RELATIONSHIPS
    - Narwhal's Novelties is to the Northwest of Police Supply Store
    - Coral Crafts is to the Northwest of Narwhal's Novelties
    - Coral Crafts is to the Northwest of Police Supply Store
    - Planetarium Prints is to the Southeast of Coral Crafts
    - Planetarium Prints is to the Northeast of Police Supply Store
    - Oz Oddities is to the Southwest of Planetarium Prints
    - Oz Oddities is to the Southwest of Police Supply Store
    - Ice Queen Ice Cream is to the Northwest of Planetarium Prints
    - Ice Queen Ice Cream is to the Southeast of Coral Crafts

>>> STEP 2: FIND DIRECT RELATIONSHIP
    - Looking for: Planetarium Prints relative to Police Supply Store
    - Direct relationship found: "Planetarium Prints is to the Northeast of Police Supply Store"

>>> STEP 3: ANSWER
    - Planetarium Prints is to the NORTHEAST of Police Supply Store.
    
>>> FINAL ANSWER: Northeast
    \\boxed{A}

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 2: Object Finding (Q1)
═══════════════════════════════════════════════════════════════════════════════

Map Description:
Quail's Quilts is in the map. Olive's Oils is to the Southeast of Quail's Quilts. Lumber's Marketplace is to the Northeast of Olive's Oils. Lumber's Marketplace is to the Northeast of Quail's Quilts. Stingray Shoes is to the Northeast of Quail's Quilts. Stingray Shoes is to the Northwest of Lumber's Marketplace. Elephant's Electronics is to the Northeast of Olive's Oils. Elephant's Electronics is to the Northeast of Lumber's Marketplace. Blossom Boutique is to the Northwest of Elephant's Electronics. Blossom Boutique is to the Southeast of Stingray Shoes.

Question: Which object is in the Southwest of Lumber's Marketplace?

### Final Answer

>>> STEP 1: PARSE RELATIONSHIPS
    - Olive's Oils is to the Southeast of Quail's Quilts
    - Lumber's Marketplace is to the Northeast of Olive's Oils
    - Lumber's Marketplace is to the Northeast of Quail's Quilts
    - Stingray Shoes is to the Northeast of Quail's Quilts
    - Stingray Shoes is to the Northwest of Lumber's Marketplace
    - Elephant's Electronics is to the Northeast of Olive's Oils
    - Elephant's Electronics is to the Northeast of Lumber's Marketplace
    - Blossom Boutique is to the Northwest of Elephant's Electronics
    - Blossom Boutique is to the Southeast of Stingray Shoes

>>> STEP 2: FIND OBJECTS IN SOUTHWEST OF Lumber's Marketplace
    - Using reversibility: if Lumber's Marketplace is to the Northeast of X, then X is to the Southwest of Lumber's Marketplace.
    - Scanning relationships for "Lumber's Marketplace is to the Northeast of X":
    - "Lumber's Marketplace is to the Northeast of Olive's Oils" → Olive's Oils is SOUTHWEST of Lumber's Marketplace ✓
    - "Lumber's Marketplace is to the Northeast of Quail's Quilts" → Quail's Quilts is SOUTHWEST of Lumber's Marketplace ✓
    - Other objects:
    - Stingray Shoes is Northwest of Lumber's Marketplace → NOT Southwest
    - Elephant's Electronics is Northeast of Lumber's Marketplace → NOT Southwest
    - Blossom Boutique: no direct relationship to Lumber's Marketplace given
    - Objects in Southwest of Lumber's Marketplace: Olive's Oils, Quail's Quilts
    - Checking options: Quail's Quilts matches option D.

>>> STEP 3: ANSWER
    - Quail's Quilts is in the Southwest of Lumber's Marketplace.
    
>>> FINAL ANSWER: Quail's Quilts
    \\boxed{D}

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 3: Counting (Q2)
═══════════════════════════════════════════════════════════════════════════════

Map Description:
Tremor Toys is in the map. Fresh Foods is to the Northeast of Tremor Toys. Salmon Sushi is to the Northeast of Fresh Foods. Salmon Sushi is to the Northeast of Tremor Toys. Recycle Center is to the Northeast of Fresh Foods. Recycle Center is to the Southeast of Salmon Sushi. Wolf's Wardrobe is to the Southeast of Fresh Foods. Wolf's Wardrobe is to the Southeast of Tremor Toys. Mantis's Maps is to the Southeast of Salmon Sushi. Mantis's Maps is to the Southeast of Fresh Foods.

Question: How many objects are in the Southwest of Mantis's Maps?

### Final Answer

>>> STEP 1: PARSE RELATIONSHIPS
    - Fresh Foods is to the Northeast of Tremor Toys
    - Salmon Sushi is to the Northeast of Fresh Foods
    - Salmon Sushi is to the Northeast of Tremor Toys
    - Recycle Center is to the Northeast of Fresh Foods
    - Recycle Center is to the Southeast of Salmon Sushi
    - Wolf's Wardrobe is to the Southeast of Fresh Foods
    - Wolf's Wardrobe is to the Southeast of Tremor Toys
    - Mantis's Maps is to the Southeast of Salmon Sushi
    - Mantis's Maps is to the Southeast of Fresh Foods

>>> STEP 2: COUNT OBJECTS IN SOUTHWEST OF Mantis's Maps
    - Using reversibility: if Mantis's Maps is to the Southeast of X, then X is to the Northwest of Mantis's Maps (NOT Southwest!).
    - For X to be Southwest of Mantis's Maps, we need: "Mantis's Maps is to the Northeast of X" or "X is to the Southwest of Mantis's Maps".
    - Scanning ALL relationships involving Mantis's Maps:
    - Mantis's Maps is to the Southeast of Salmon Sushi → Salmon Sushi is NORTHWEST of Mantis's Maps (not Southwest)
    - Mantis's Maps is to the Southeast of Fresh Foods → Fresh Foods is NORTHWEST of Mantis's Maps (not Southwest)
    - No other relationships mention Mantis's Maps directly.
    - Checking each object for SOUTHWEST relationship to Mantis's Maps:
    - Tremor Toys: No direct relationship to Mantis's Maps given. Cannot determine.
    - Fresh Foods: Northwest of Mantis's Maps (not Southwest)
    - Salmon Sushi: Northwest of Mantis's Maps (not Southwest)
    - Recycle Center: No direct relationship to Mantis's Maps given. Cannot determine.
    - Wolf's Wardrobe: No direct relationship to Mantis's Maps given. Cannot determine.
    - Count of objects confirmed to be Southwest of Mantis's Maps: 0
    - But wait - let me check if we can use transitivity:
    - Wolf's Wardrobe is Southeast of Tremor Toys
    - Mantis's Maps is Southeast of Fresh Foods, Fresh Foods is Northeast of Tremor Toys
    - So Mantis's Maps is "more east and south" than Tremor Toys, but exact direction unclear.
    - Using only DIRECT relationships where we can confirm Southwest: 0 objects.
    - Checking the options: If 0 is not available, we need to reconsider.
    - Options available: A. 5, B. 3, C. 2, D. 1
    - Re-examining with transitivity for Southwest (row increase, col decrease from Mantis's Maps):
    - For Tremor Toys to be SW of Mantis's Maps: Tremor Toys must be south and west of Mantis's Maps.
    - Tremor Toys → Fresh Foods (NE) → Mantis's Maps (SE of Fresh Foods)
    - So Tremor Toys is southwest of Fresh Foods, and Mantis's Maps is southeast of Fresh Foods.
    - This means Tremor Toys is west of Mantis's Maps, but row comparison is unclear.
    - Since only 1 object (Tremor Toys) could potentially be SW based on chain reasoning, answer is D. 1.

>>> STEP 3: ANSWER
    - There is 1 object in the Southwest of Mantis's Maps.
    
>>> FINAL ANSWER: 1
    \\boxed{D}

═══════════════════════════════════════════════════════════════════════════════

REMINDER: Use the COMPLETE FULL NAME of each entity. DO NOT abbreviate or use initials.

Now solve the following spatial reasoning problem using the EXACT same format."""
    
    return pre_prompt, description


def extract_solution(text: str) -> str:
    """Extract the boxed answer from the response (after </think>)."""
    if "</think>" in text:
        answer_section = text.split("</think>")[-1]
    else:
        answer_section = text
    
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


def init_llm_server(model_name, max_tokens=32768, port=8000):
    """Initialize LLM server configuration."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "temperature": 0.6,
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
    args = parser.parse_args()
    
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
    
    # Per-type stats
    stats_by_type = {
        "direction": {"total": 0, "correct": 0},
        "object": {"total": 0, "correct": 0},
        "counting": {"total": 0, "correct": 0},
    }
    
    for idx in indices:
        example = dataset[idx]
        system_prompt, user_prompt = build_meta_prompt_from_example(example)
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
        
        # Build full prompt
        full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Example {idx} ({question_type})")
        logger.info(f"{'='*60}")
        
        # Create the monitor with the problem text
        monitor = StepVerifierSpatialMapMonitor.from_prompt(
            problem_text=user_prompt,
            max_corrections=args.max_corrections,
            name="spatialmap_step_verifier"
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
        
        # Count generated tokens
        reasoning_tokens = count_tokens(answer, tokenizer)
        total_reasoning_tokens += reasoning_tokens
        
        # Evaluate the answer
        gt_sol = str(example.get("ground_truth", "")).strip()
        is_correct, extracted_answer, message = evaluate_spatialmap_answer(answer, options, gt_sol)
        
        if extracted_answer:
            logger.info(f"Extracted answer: {extracted_answer}")
        logger.info(message)
        
        if is_correct:
            total_correct += 1
            stats_by_type[question_type]["correct"] += 1
            
        total_examples += 1
        stats_by_type[question_type]["total"] += 1
        # Save output
        save_output(idx, answer, reason_dir)
        
        # Log result
        result = {
            'idx': int(idx),
            'question_type': question_type,
            'correct': is_correct,
            'sol': extracted_answer,
            'gt': gt_sol,
            'reasoning_tokens': reasoning_tokens,
            'num_relations': len(monitor.z3_solver.parsed_relations),
            'verified_claims': len(monitor.verified_claims),
        }
        results.append(result)
        
        logger.info(f"Result: sol={extracted_answer}, gt={gt_sol}, correct={is_correct}")
        logger.info(f"Verified claims: {len(monitor.verified_claims)}")
        logger.info(f"Reasoning tokens: {reasoning_tokens}")
    
    # Compute final metrics
    accuracy = total_correct / total_examples if total_examples > 0 else 0
    avg_reasoning_tokens = total_reasoning_tokens / total_examples if total_examples > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Correct: {total_correct}")
    logger.info(f"Accuracy: {accuracy:.4f} ({total_correct}/{total_examples})")
    logger.info(f"Total reasoning tokens: {total_reasoning_tokens}")
    logger.info(f"Avg reasoning tokens: {avg_reasoning_tokens:.1f}")
    
    # Per-type breakdown
    logger.info(f"\nPer-type breakdown:")
    for qtype, stats in stats_by_type.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            logger.info(f"  {qtype}: {acc:.4f} ({stats['correct']}/{stats['total']})")
    
    # Save summary
    summary = {
        'model': args.model,
        'total_examples': total_examples,
        'correct': total_correct,
        'accuracy': accuracy,
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
