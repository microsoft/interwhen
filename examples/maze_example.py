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

from interwhen import stream_completion
from interwhen.monitors import SimpleTextReplaceMonitor, KstableAnswerMCQMonitor
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

    num_correct = 0
    N = args.num_examples
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
        sol = extract_solution(answer)
        gt_sol = str(example.get("ground_truth", "")).strip()         
        if not sol:
            logger.info("No expression found.")
            continue          
        sol = sol.strip()      
        # Case 1: LLM returned option letter (A/B/C/D)
        if sol in options:
            logger.info(f"Extracted option letter: {sol} -> {options[sol]}")
            if options[sol] == gt_sol:
                logger.info("Correct solution answer matches")
                num_correct += 1
            else:
                logger.info(f"Incorrect solution: expected '{gt_sol}', got '{options[sol]}'")
        else:
            # Case 2: LLM returned the actual answer text
            # Check if sol matches any option value or ground truth directly
            matched = False
            
            # First check if sol matches ground truth directly
            if sol.lower() == gt_sol.lower():
                logger.info(f"Extracted answer text matches ground truth: {sol}")
                num_correct += 1
                matched = True
            else:
                # Check if sol matches any option value
                for opt_letter, opt_value in options.items():
                    if sol.lower() == opt_value.lower():
                        logger.info(f"Extracted answer text: {sol} (option {opt_letter})")
                        if opt_value == gt_sol:
                            logger.info("Correct solution answer matches")
                            num_correct += 1
                        else:
                            logger.info(f"Incorrect solution: expected '{gt_sol}', got '{opt_value}'")
                        matched = True
                        break
            
            if not matched:
                logger.info(f"Solution '{sol}' not found in options or ground truth")

    print(f"\nFinal Accuracy: {num_correct}/{N}")