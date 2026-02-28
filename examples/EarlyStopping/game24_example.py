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
from interwhen.monitors import SimpleTextReplaceMonitor, KstableAnswerGame24Monitor, EATMonitor, DEERMonitor

# ============== MODEL CONFIGURATION ==============
# Change these model names to scale experiments easily
MAIN_MODEL = "microsoft/Phi-4-reasoning"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# =================================================

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name

def get_output_dirs(main_model: str, base_dir: str = "../../Outputs_Kstable2/Gameof24_results"):
    """Create and return output directory paths based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_log_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_Kstable2/Gameof24_results") -> str:
    """Generate log filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")

def get_token_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_Kstable2/Gameof24_results") -> str:
    """Generate token CSV filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.csv")

def save_prompt(idx, prompt_with_answer, reason_dir):
    filename = os.path.join(reason_dir, f"reason_{idx}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)

logger = logging.getLogger(__name__)


def load_game24_dataset():
    ds = load_dataset("nlile/24-game", split="train")
    return ds

def init_llm_server(modelname, max_tokens=200, port=8000):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 50,
        "top_p": 0.95,
        "temperature": 0.8,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed" : 42
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def build_prompt(nums):
    a, b, c, d = nums
    boxed = r"\boxed{}"
    base_prompt = f"""
    You are solving the Game of 24.
    
    You are given four numbers: {a}, {b}, {c}, {d}
    
    Your job is to produce a valid arithmetic expression using:
    - ALL four numbers exactly once
    - ONLY +, -, *, /
    - The expression must evaluate to exactly 24.
    
    Please reason step by step, and put your final answer containing only the expression within {boxed}.""".strip()

    return base_prompt


def count_tokens(text, tokenizer):
    """Count the total number of tokens in the generated text using the tokenizer."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def extract_solution(text):
    
    # Use a more robust extraction that handles nested braces in \boxed{}
    # Find \boxed{ and then match braces properly
    boxed_pattern = r"\\boxed\{"
    matches = list(re.finditer(boxed_pattern, text))
    if not matches:
        return None
    
    # Get the last \boxed{} content by matching braces
    last_match = matches[-1]
    start = last_match.end()  # Position right after \boxed{
    brace_count = 1
    end = start
    while end < len(text) and brace_count > 0:
        if text[end] == '{':
            brace_count += 1
        elif text[end] == '}':
            brace_count -= 1
        end += 1
    
    expr = text[start:end-1].strip()  # -1 to exclude the closing brace

    # 1. Convert \frac{a}{b} to (a/b)
    frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    while re.search(frac_pattern, expr):
        expr = re.sub(frac_pattern, r"(\1/\2)", expr)

    # 2. Replace LaTeX operators
    replacements = {
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/",
    }
    for latex, op in replacements.items():
        expr = expr.replace(latex, op)

    # 3. Cleanup (remove LaTeX formatting artifacts)
    expr = expr.replace(r"\,", "").replace(r"\ ", "")
    expr = expr.replace(r"\left", "").replace(r"\right", "")

    # 4. Handle implicit multiplication (e.g., "(11+1)(1+1)" -> "(11+1)*(1+1)")
    # Insert * between: )( , )number, number(, )(
    expr = re.sub(r'\)\s*\(', ')*(', expr)  # )( -> )*(
    expr = re.sub(r'\)\s*(\d)', r')*\1', expr)  # )number -> )*number
    expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)  # number( -> number*(

    return expr

def extract_numbers_from_expr(expr):
    """Extract all numbers (including decimals) from an expression."""
    # Match integers and decimals
    numbers = re.findall(r'\d+\.?\d*', expr)
    return [int(float(n)) if float(n).is_integer() else float(n) for n in numbers]

def validate_numbers_used(expr, expected_nums):
    """Check if the expression uses exactly the given numbers (each exactly once)."""
    used_nums = extract_numbers_from_expr(expr)
    # Sort both lists to compare
    return sorted(used_nums) == sorted(expected_nums)

def evaluate_expression(expr, expected_nums=None):
    try:
        # First check if expression uses exactly the given numbers
        if expected_nums is not None:
            if not validate_numbers_used(expr, expected_nums):
                return False
        
        value = eval(expr, {"__builtins__": None}, {})
        return abs(value - 24) < 1e-6
    except Exception:
        return False


def evaluate_game24_answer(answer, nums):
    """
    Evaluate a Game24 answer and return (is_correct, expr, error_message).
    
    Args:
        answer: Raw model output
        nums: Expected numbers to use
        
    Returns:
        Tuple of (is_correct, extracted_expression, error_message)
    """
    expr = extract_solution(answer)
    
    if not expr:
        return False, None, "No expression found"
    
    if evaluate_expression(expr, expected_nums=nums):
        return True, expr, "Correct solution (evaluates to 24 using exactly the given numbers)"
    else:
        used_nums = extract_numbers_from_expr(expr)
        if sorted(used_nums) != sorted(nums):
            return False, expr, f"Incorrect: Expression uses {used_nums}, expected {nums}"
        else:
            return False, expr, "Expression does not evaluate to 24"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Game of 24 step-by-step solver with monitors")
    parser.add_argument("--thinking", "-t", action="store_true", help="Enable chain-of-thought output")
    parser.add_argument("--monitor", "-m", default = True, action="store_true", help="Enable step-by-step monitor")
    parser.add_argument("--num_examples", "-n", type=int, default=1362, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--earlystop_model", type=str, default=EARLYSTOP_MODEL, help="Model to use for early stopping")
    args = parser.parse_args()

    # Use models from args (allows command-line override)
    main_model = args.main_model
    earlystop_model = args.earlystop_model

    # Setup output directories based on model name
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

    dataset = load_game24_dataset()

    llm_server = init_llm_server(main_model, max_tokens=20000)

    # Load tokenizer for accurate token counting
    logger.info(f"Loading tokenizer for {main_model}...")
    tokenizer = AutoTokenizer.from_pretrained(main_model, trust_remote_code=True)
    logger.info("Tokenizer loaded successfully.")

    num_correct = 0
    N = args.num_examples
    total_generated_tokens = 0
    generated_token_counts = []

    # total = len(dataset)
    indices = np.linspace(0, len(dataset)-1, N, dtype=int)

    for idx in indices: #for idx in indices:
        example = dataset[idx]
        nums = example["numbers"]

        prompt = build_prompt(nums)

        if args.monitor:
            # Use K-stable answer monitor to detect when equation stabilizes k times
            # monitors = (SimpleTextReplaceMonitor("IsCheck", "</think>", async_execution=False),)
            monitors=(KstableAnswerGame24Monitor(
                name="game24_kstable",
                k=2,
                expected_nums=nums,  # Validate equations use exactly these numbers
                answer_start_token="</think>"
            ),)
            # monitors = (
            #     EATMonitor(
            #         name="EAT_monitor",
            #         model_name=earlystop_model,
            #         alpha=0.2,
            #         delta=0.02,
            #         min_steps=4,
            #         answer_start_token="</think>",
            #         async_execution=True
            #     ),
            # )
        else:
            monitors = ()

        logger.info(f"---- length of monitors {len(monitors)} ----")
        logger.info(f"---- Example {idx+1} ----")
        logger.info(f"Numbers: {nums}")

        system_prompt = (
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

        answer = asyncio.run(stream_completion(
            f"<|im_start|>system<|im_sep|>\n{system_prompt}<|im_end|>\n<|im_start|>user<|im_sep|>\n{prompt}<|im_end|>\n<|im_start|>assistant<|im_sep|>\n",
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
        is_correct, expr, message = evaluate_game24_answer(answer, nums)
        
        if expr:
            logger.info(f"Extracted expression: {expr}")
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
        f.write(f"Game of 24 Evaluation Results\n")
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
