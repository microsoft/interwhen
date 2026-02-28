"""
Maze experiment with step-by-step verification using StepVerifierMazeMonitor.

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
from interwhen.monitors import StepVerifierMazeMonitor
from interwhen.utils.maze_verifier import parse_maze_from_prompt

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


def get_output_dirs(main_model: str, base_dir: str = "../../Outputs_TTS/MazeResults/metaPrompt"):
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


def save_prompt(idx, prompt_with_answer, reason_dir):
    """Save reasoning trace to a text file."""
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)


def build_meta_prompt_from_example(example):
    """Build prompt for maze example."""
    system_prompt = """You are a maze-solving AI. Given a maze in ASCII format, analyze it step by step.

COORDINATE SYSTEM:
- Rows are numbered from top (row 0) to bottom
- Columns are numbered from left (col 0) to right
- Movement: UP (row decreases), DOWN (row increases), LEFT (col decreases), RIGHT (col increases)

TURN DEFINITIONS:
- RIGHT_TURN = 90° clockwise change (e.g., DOWN→LEFT, LEFT→UP, UP→RIGHT, RIGHT→DOWN)
- LEFT_TURN = 90° counterclockwise change (e.g., DOWN→RIGHT, RIGHT→UP, UP→LEFT, LEFT→DOWN)

RELATIVE POSITION DEFINITIONS:
- "directly to the left" = same row, E has smaller column than S
- "directly to the right" = same row, E has larger column than S
- "directly above" = same column, E has smaller row than S
- "directly below" = same column, E has larger row than S
- "top left" = E has smaller row AND smaller column than S
- "top right" = E has smaller row AND larger column than S
- "bottom left" = E has larger row AND smaller column than S
- "bottom right" = E has larger row AND larger column than S

IMPORTANT: Follow the EXACT output format below. Do NOT use <think> tags.

EXAMPLE 1: Counting Right Turns
Question: How many right turns are there in the path from S to E?

>>> LOCATE START AND EXIT:
    S position: (3,5)
    E position: (1,1)

>>> STEP 1: Move DOWN from (3,5) to (4,5)
    Current position: (4,5)
    Previous direction: —
    Current direction: DOWN
    Turn type: STRAIGHT
    Running count: Right=0, Left=0

>>> STEP 2: Move DOWN from (4,5) to (5,5)
    Current position: (5,5)
    Previous direction: DOWN
    Current direction: DOWN
    Turn type: STRAIGHT
    Running count: Right=0, Left=0

>>> STEP 3: Move LEFT from (5,5) to (5,4)
    Current position: (5,4)
    Previous direction: DOWN
    Current direction: LEFT
    Turn type: RIGHT_TURN
    Running count: Right=1, Left=0

>>> FINAL ANSWER: Right turns = 2
    \\boxed{C}

EXAMPLE 2: Counting Total Turns
Question: How many total turns are there in the path from S to E?

>>> LOCATE START AND EXIT:
    S position: (3,5)
    E position: (1,1)

>>> STEP 1: Move DOWN from (3,5) to (4,5)
    Current position: (4,5)
    Previous direction: —
    Current direction: DOWN
    Turn type: STRAIGHT
    Running count: Right=0, Left=0, Total=0

[... continue for all steps ...]

>>> FINAL ANSWER: Total turns = 2
    \\boxed{C}

EXAMPLE 3: Relative Position
Question: Is the exit (E) to the top left of the starting point (S)?

>>> LOCATE START AND EXIT:
    S position: (3,5)
    E position: (1,1)

>>> COMPARE POSITIONS:
    Row comparison: E row (1) < S row (3) → E is ABOVE S ✓
    Col comparison: E col (1) < S col (5) → E is LEFT of S ✓

>>> ANALYSIS:
    E is above S (smaller row): YES
    E is left of S (smaller col): YES
    Therefore E is at TOP LEFT of S.

>>> ANSWER: YES, E is to the top left of S.
    \\boxed{A}

════════════════════════════════════════════════════════════════════════════════
Now solve the following maze using the EXACT same format. First locate S and E, then trace the path step by step."""

    # Get the maze description (trimmed to remove trailing instructions)
    description = str(example.get("prompt", ""))
    description_trimmed = description[:-143] if len(description) > 143 else description
    
    return system_prompt, description_trimmed


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


def get_question_type_from_index(idx: int) -> str:
    """Determine question type based on index range.
    
    Dataset structure:
    - 3000-3499: right turns
    - 3500-3999: total turns
    - 4000-4500: relative position
    """
    if idx < 3500:
        return "right_turns"
    elif idx < 4000:
        return "total_turns"
    else:
        return "relative_position"


def init_llm_server(model_name, max_tokens=22000, port=8000):
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

def evaluate_maze_answer(answer, options, ground_truth):
    """
    Evaluate a Maze MCQ answer and return (is_correct, extracted_answer, message).
    
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
    parser = argparse.ArgumentParser(description="Run maze experiments with StepVerifierMazeMonitor")
    parser.add_argument("--model", type=str, default=MAIN_MODEL,
                        help="Model name for generation")
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated indices to run (e.g., '3000,3500,4000')")
    parser.add_argument("--start", type=int, default=3000, help="Start index")
    parser.add_argument("--end", type=int, default=3010, help="End index")
    parser.add_argument("--num_examples", "-n", type=int, default=None,
                        help="Number of examples to run (overrides start/end)")
    parser.add_argument("--max_corrections", type=int, default=5,
                        help="Maximum number of correction attempts per example")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load dataset
    dataset = load_dataset("microsoft/VISION_LANGUAGE", 'maze', split='val')
    
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
        # Use 4499 as endpoint (4500 is out of bounds since dataset size is 4500)
        indices = np.linspace(3000, 4499, args.num_examples, dtype=int)
    else:
        indices = range(args.start, args.end)
    
    # Stats tracking
    results = []
    total_correct = 0
    total_examples = 0
    total_reasoning_tokens = 0
    
    for idx in indices:
        example = dataset[idx]
        system_prompt, user_prompt = build_meta_prompt_from_example(example)
        if str(example.get("ground_truth", "")).strip() == "Q4":
            target_options = ["A", "B"]
        else:
            target_options = ["A", "B", "C", "D"] 
        keys = "|".join(map(re.escape, target_options))
        pattern = rf'\b({keys})\.\s*([A-Za-z0-9]+)\b'
        options = dict(re.findall(pattern, user_prompt))
        
        # Build full prompt with Phi-4-reasoning ChatML format
        full_prompt = f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n<|im_start|>user<|im_sep|>\n{user_prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n<think>\n"
        
        # Parse maze from prompt
        grid, start_pos, exit_pos = parse_maze_from_prompt(user_prompt)
        
        if not grid or not start_pos or not exit_pos:
            logger.error(f"Could not parse maze for example {idx}")
            continue
        
        # Detect question type from prompt (auto-detection)
        # Falls back to index-based if no turn keywords found
        question_type = StepVerifierMazeMonitor.detect_question_type(user_prompt)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Example {idx} ({question_type})")
        logger.info(f"Maze: S={start_pos}, E={exit_pos}, grid={len(grid)}x{len(grid[0]) if grid else 0}")
        logger.info(f"{'='*60}")
        
        # Create the monitor
        monitor = StepVerifierMazeMonitor(
            name="maze_step_verifier",
            answer_start_token="</think>",
            grid=grid,
            start_pos=start_pos,
            exit_pos=exit_pos,
            max_corrections=args.max_corrections,
            question_type=question_type,
        )
        
        # Run with stream_completion
        try:
            answer = asyncio.run(stream_completion(
                full_prompt,
                llm_server=llm_server,
                monitors=(),
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
        save_prompt(idx, answer, reason_dir)

        # Count generated tokens
        reasoning_tokens = count_tokens(answer, tokenizer)
        total_reasoning_tokens += reasoning_tokens
        
        gt_sol = str(example.get("ground_truth", "")).strip()
        is_correct, extracted_answer, message = evaluate_maze_answer(answer, options, gt_sol)
        
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
            'sol': extracted_answer,
            'gt': gt_sol,
            'reasoning_tokens': reasoning_tokens,
        }
        results.append(result)
        
        logger.info(f"Result: sol={extracted_answer}, gt={gt_sol}, correct={is_correct}")
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
    
    # Save summary
    summary = {
        'model': args.model,
        'total_examples': total_examples,
        'correct': total_correct,
        'accuracy': accuracy,
        'total_reasoning_tokens': total_reasoning_tokens,
        'avg_reasoning_tokens': avg_reasoning_tokens,
        'max_corrections': args.max_corrections,
        'results': results,
    }
    
    summary_path = os.path.join(output_dirs["base"], "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved summary to {summary_path}")