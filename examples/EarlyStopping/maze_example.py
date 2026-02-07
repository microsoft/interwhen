import argparse
import asyncio
import json
import logging
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import csv

from datasets import load_dataset
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import SimpleTextReplaceMonitor, KstableAnswerMCQMonitor, EATMonitor, DEERMonitor
import re

# ============== MODEL CONFIGURATION ==============
# Change these model names to scale experiments easily
MAIN_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# =================================================

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    # Get the last part after '/' and replace any problematic characters
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name

def get_output_dirs(main_model: str, base_dir: str = "../MazeResults"):
    """Create and return output directory paths based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
        "plots": os.path.join(output_base, "plots"),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_log_filename(main_model: str, num_examples: int, base_dir: str = "../MazeResults") -> str:
    """Generate log filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")

def get_token_filename(main_model: str, num_examples: int, base_dir: str = "../MazeResults") -> str:
    """Generate token CSV filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.csv")

def remove_last_paragraph(s: str) -> str:
    return s[:-143]

logger = logging.getLogger(__name__)

def load_maze_dataset(split="val"):
    ds = load_dataset("microsoft/VISION_LANGUAGE", "maze", split=split)
    return ds

def init_llm_server(modelname, max_tokens=200, port=8000): #
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "do_sample" : True,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed" : 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def build_prompt_from_example(example): #(original prompt config)

    pre_prompt = """You are an expert problem solver. Carefully read the following multiple-choice question and think through the solution step-by-step before providing your final answer. Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"""

    description = example.get("prompt")
    description = str(description)

    # remove the unecessary parts of the prompt and then add the prompt that we need.
    description = remove_last_paragraph(description)
    return pre_prompt , description


def extract_solution(text):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if not matches:
        return None

    expr = matches[-1].strip()   # take last boxed content

    # find one of A/B/C/D inside the boxed content
    choice_match = re.search(r"\b([ABCD])\b", expr, flags=re.IGNORECASE)
    if not choice_match:
        return None

    return choice_match.group(1).upper()

def save_prompt(idx, prompt_with_answer, reason_dir):
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)


def count_tokens(text, tokenizer):
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


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

    parser = argparse.ArgumentParser(description="Maze problem solver with LLM and monitors")
    parser.add_argument("--thinking", "-t", action="store_true", help="Enable chain-of-thought output")
    parser.add_argument("--monitor", "-m", default = True, action="store_true", help="Enable step-by-step monitor")
    parser.add_argument("--num_examples", "-n", type=int, default=1500, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--earlystop_model", type=str, default=EARLYSTOP_MODEL, help="Model to use for early stopping")
    args = parser.parse_args()

    main_model = args.main_model
    earlystop_model = args.earlystop_model
    
    output_dirs = get_output_dirs(main_model)
    logfile = get_log_filename(main_model, args.num_examples)
    token_filename = get_token_filename(main_model, args.num_examples)
    reason_dir = output_dirs["reasoning"]

    log_level = logging.DEBUG if args.debug else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w"),  
            logging.StreamHandler()
        ],
        force=True,
    )

    logger.info(f"Main model: {main_model}")
    logger.info(f"Early stop model: {earlystop_model}")
    logger.info(f"Output directory: {output_dirs['base']}")

    dataset = load_maze_dataset()

    llm_server = init_llm_server(main_model, max_tokens=15000)

    # Load tokenizer for accurate token counting
    logger.info(f"Loading tokenizer for {main_model}...")
    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")

    num_correct = 0
    N = args.num_examples
    total_generated_tokens = 0
    generated_token_counts = []
    total = len(dataset)
    indices = np.linspace(3000, total-1, N, dtype=int).tolist()

    for idx in indices:
        example = dataset[idx]
        prompt1, prompt2 = build_prompt_from_example(example)
        if str(example.get("ground_truth", "")).strip() == "Q4":
            target_options = ["A", "B"]
        else:
            target_options = ["A", "B", "C", "D"] 
        keys = "|".join(map(re.escape, target_options))
        pattern = rf'\b({keys})\.\s*([A-Za-z0-9]+)\b'
        options = dict(re.findall(pattern, prompt2))

        if args.monitor:
            # Use K-stable answer monitor to detect when equation stabilizes k times
            # monitors = (SimpleTextReplaceMonitor("IsCheck", "</think>", async_execution=False),)
            monitors=(KstableAnswerMCQMonitor(
                name="maze_kstable",
                k=3,
                options=options,  # Validate equations use exactly these numbers
                answer_start_token="</think>"
            ),)
        else:
            monitors = ()

        logger.info(f"---- length of monitors {len(monitors)} ----")
        logger.info(f"---- Example {idx} ----")

        # Run LLM with streaming + monitor

        answer = asyncio.run(stream_completion(
            f"<|im_start|>system\n{prompt1}<|im_end|>\n<|im_start|>user\n{prompt2}<|im_end|>\n<|im_start|>assistant\n",
            llm_server=llm_server,
            monitors=monitors,
            add_delay=False,
            termination_requires_validation=False,
            async_execution=True
        ))

        save_prompt(idx, answer, reason_dir)
        logger.info(f"Raw final output:\n{answer}")

        generated_tokens = count_tokens(answer, tokenizer)
        generated_token_counts.append(generated_tokens)
        total_generated_tokens += generated_tokens
        logger.info(f"Generated tokens in this example: {generated_tokens}")

        # Evaluate the answer
        gt_sol = str(example.get("ground_truth", "")).strip()
        is_correct, extracted_answer, message = evaluate_maze_answer(answer, options, gt_sol)
        
        if extracted_answer:
            logger.info(f"Extracted answer: {extracted_answer}")
        logger.info(message)
        
        if is_correct:
            num_correct += 1

    # Calculate final statistics
    avg_generated_tokens = total_generated_tokens / N if N > 0 else 0
    accuracy = num_correct / N if N > 0 else 0
    
    print(f"\nFinal Accuracy: {num_correct}/{N} ({accuracy:.2%})")
    print(f"Average Generated Tokens: {avg_generated_tokens:.2f}")
    print(f"Total Generated Tokens: {total_generated_tokens}")
    
    # Save results to a text file
    results_file = logfile.replace('.log', '_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Maze Evaluation Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {main_model}\n")
        f.write(f"Number of Examples: {N}\n")
        f.write(f"Monitor Enabled: {args.monitor}\n\n")
        f.write(f"Results:\n")
        f.write(f"---------\n")
        f.write(f"Correct: {num_correct}/{N}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n\n")
        f.write(f"Generated Token Statistics:\n")
        f.write(f"---------------------------\n")
        f.write(f"Total Generated Tokens: {total_generated_tokens}\n")
        f.write(f"Average Generated Tokens: {avg_generated_tokens:.2f}\n")
        if generated_token_counts:
            f.write(f"Min Generated Tokens: {min(generated_token_counts)}\n")
            f.write(f"Max Generated Tokens: {max(generated_token_counts)}\n")
            f.write(f"Std Dev: {np.std(generated_token_counts):.2f}\n")
    
    logger.info(f"Results saved to {results_file}")
    print(f"Results saved to {results_file}")