import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import shutil
import numpy as np
import csv
import asyncio
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from interwhen.interject import stream_completion
from interwhen.monitors import EATMonitor, StepVerifierVerinaMonitor
from interwhen.utils.verina_code_example_utils import *

logger = logging.getLogger(__name__)

# Model config
MAIN_MODEL = "Qwen/QwQ-32B"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    # Get the last part after '/' and replace any problematic characters
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


def get_output_dirs(main_model: str, base_dir: str = "../../../../Outputs_TTS/verina_code_results"):
    """Create and return output directory paths based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output_verina"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
        "plots": os.path.join(output_base, "plots"),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def get_log_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_TTS/verina_code_results") -> str:
    """Generate log filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.log")


def get_token_filename(main_model: str, num_examples: int, base_dir: str = "../../Outputs_TTS/verina_code_results") -> str:
    """Generate token CSV filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"EAT_{num_examples}examples.csv")

# Paths
_SCRIPT_DIR = Path(__file__).parent.resolve()
VERINA_ROOT = (_SCRIPT_DIR / "../../../../verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

# LLM Server Setup
def init_llm_server(modelname: str, max_tokens: int = 20480, port: int = 8000) -> dict:
    """Initialize LLM server configuration"""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "seed": 42,
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


# Saving and plotting Utils
def save_reasoning_trace(idx: int, data_id: str, prompt_with_answer: str, reason_dir: str):
    """Save the full reasoning trace to a file"""
    filename = os.path.join(reason_dir, f"reason_{idx}_{data_id}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)


def save_results_csv(results: list, output_path: str):
    """Save results to CSV file"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "data_id", "compiles", "all_tests_pass", "num_tests", "num_tests_passed", "reasoning_tokens", "generated_code","finally_wrong"])
        for r in results:
            # Escape newlines in generated_code for CSV compatibility
            code_escaped = r["generated_code"].replace("\n", "\\n") if r["generated_code"] else ""
            writer.writerow([
                r["idx"], 
                r["data_id"], 
                r["compiles"], 
                r["all_tests_pass"], 
                r["num_tests"],
                r["num_tests_passed"],
                r["reasoning_tokens"], 
                code_escaped,
                r['finally_wrong']
            ])


def compute_average_tokens(token_file: str) -> float:
    """Compute average reasoning tokens from the token file"""
    if not os.path.exists(token_file):
        return 0.0
    
    tokens = []
    with open(token_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row:
                tokens.append(int(row[0]))
    
    return np.mean(tokens) if tokens else 0.0


def plot_entropy_ewma(monitors, save_path):
    """Plot entropy and EWMA metrics."""
    entropy = monitors[0].entropy
    ema_mean = monitors[0].ema_means
    ema_var = monitors[0].ema_vars

    chunks_no = list(range(1, len(entropy) + 1))

    if monitors[0].exit_point is None:
        exit_point = len(entropy) - 1
    else:
        exit_point = monitors[0].exit_point - 1
    plt.figure(figsize=(12, 7))
    plt.plot(chunks_no, entropy, label="Entropy", linewidth=1.8)
    plt.plot(chunks_no, ema_mean, label="EWMA Mean", linewidth=1.8)
    plt.plot(chunks_no, ema_var, label="EWMA Variance", linewidth=1.8)

    plt.axvline(exit_point, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

    # Star markers on each curve
    plt.plot(exit_point, entropy[exit_point], "r*", markersize=14)
    plt.plot(exit_point, ema_mean[exit_point], "r*", markersize=14)
    plt.plot(exit_point, ema_var[exit_point], "r*", markersize=14)

    # Label the exit point
    plt.text(exit_point + 0.3, entropy[exit_point],
             f" Exit @ {exit_point}", color="red", fontsize=10)

    plt.xlabel("Chunk Index")
    plt.ylabel("Value")
    plt.title("EAT per Chunk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_entropy_ewma2(monitors, save_path):
    """Plot EWMA variance."""
    chunks_no = list(range(1, len(monitors[0].entropy) + 1))
    plt.figure(figsize=(12, 7))
    plt.plot(chunks_no, monitors[0].ema_vars, label="EWMA Variance", linewidth=1.8)
    if monitors[0].exit_point is None:
        exit_point = len(monitors[0].ema_vars) - 1
    else:
        exit_point = monitors[0].exit_point - 1
    plt.axvline(exit_point, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    plt.plot(exit_point, monitors[0].ema_vars[exit_point], "r*", markersize=14)

    plt.xlabel("Chunk Index")
    plt.ylabel("Value")
    plt.title("EWMA Variance per Chunk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_entropy_ewma3(monitors, save_path):
    """Plot DEER confidence."""
    chunks_no = list(range(1, len(monitors[0].confidence) + 1))
    plt.figure(figsize=(12, 7))
    plt.plot(chunks_no, monitors[0].confidence, label="DEER Confidence", linewidth=1.8)

    plt.xlabel("Chunk Index")
    plt.ylabel("Value")
    plt.title("DEER Confidence per Chunk")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verina benchmark solver with LLM and monitors")
    parser.add_argument("--monitor", "-m", action="store_true", default=True, help="Enable monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=189, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
    parser.add_argument("--port", "-p", type=int, default=8001, help="LLM server port")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--earlystop_model", type=str, default=EARLYSTOP_MODEL, help="Model to use for early stopping")
    parser.add_argument("--k_steps", "-k", type=int, default=75, help="Newlines threshold for forcing code output")
    parser.add_argument("--tasks", "-t", type=str, default=None, help="Comma-separated list of task IDs to run (e.g., verina_advanced_10,verina_basic_2)")
    parser.add_argument("--max_corrections", type=int, default=5,help="Maximum number of correction attempts per example")
    args = parser.parse_args()
    
    # Use models from args
    main_model = args.main_model
    earlystop_model = args.earlystop_model
    
    # Setup output directories based on model name
    output_dirs = get_output_dirs(main_model)
    logfile = get_log_filename(main_model, args.num_examples)
    token_file = get_token_filename(main_model, args.num_examples)
    reason_dir = output_dirs["reasoning"]
    csv_dir = output_dirs["csv_saved"]
    
    # Setup logging
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
    
    # Load dataset
    logger.info("Loading verina dataset...")
    dataset = load_verina_dataset()
    logger.info(f"Loaded {len(dataset)} tasks")

    print("=============testing lean compile=================")
    test_lean_compile()    
    # Setup LLM
    llm_server = init_llm_server(main_model, max_tokens=20480, port=args.port)
    
    # Filter tasks if --tasks is specified
    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
        filtered_dataset = [d for d in dataset if d.data_id in task_ids]
        logger.info(f"Filtered to {len(filtered_dataset)} tasks: {task_ids}")
        dataset = filtered_dataset
        N = len(dataset)
        indices = list(range(N))
    else:
        # Select examples
        N = args.num_examples if args.num_examples > 0 else len(dataset)
        total = len(dataset)
        indices = np.linspace(0, total - 1, N, dtype=int)
    
    # Results tracking
    results = []
    num_correct = 0
    
    logger.info(f"Running on {N} examples...")
    
    for i, idx in enumerate(indices):

        print("SAMPLE ",i+1)
        data = dataset[idx]
        logger.info(f"\n{'='*50}")
        logger.info(f"[{i+1}/{N}] Task: {data.data_id}")
        logger.info(f"{'='*50}")
        
        # Build prompt
        prompt = build_full_prompt(data)
        
        # Convert BenchmarkData to dict for the monitor
        task_data = {
            "data_id": data.data_id,
            "description": data.description,
            "signature": data.signature,
            "lean_data": data.lean_data,
            "spec_desc": data.spec_desc,
            "tests": data.tests,
            "metadata": data.metadata,
        }
        
        # Setup monitors
        if args.monitor:
            monitors = [
                StepVerifierVerinaMonitor(
                    name="VerinaStepVerifier",
                    task_data=task_data,
                    llm_server=llm_server,
                    prompt=prompt,
                    k_steps=40,  # Force code after every K newlines
                    compile_timeout=120,
                    max_corrections=args.max_corrections,
                ),
            ]
        else:
            monitors = []
        
        # Run LLM with streaming + monitor
        try:
            answer = asyncio.run(
                stream_completion(
                    prompt,
                    prev_text="",
                    llm_server=llm_server,
                    monitors=monitors,
                    add_delay=False,
                    num_calls_index=0,
                    async_execution=True,
                )
            )
            prompt_with_answer = prompt + answer
            print("ANSWER: ",answer)
        except Exception as e:
            print(f"Error during LLM generation: {e}")
            logger.error(f"Error during LLM generation: {e}")
            results.append({
                "idx": idx,
                "data_id": data.data_id,
                "compiles": False,
                "all_tests_pass": False,
                "num_tests": len(data.tests) if data.tests else 0,
                "num_tests_passed": 0,
                "reasoning_tokens": 0,
                "generated_code": "",
                "any_code_generated": False
            })
            continue
        
        # Save reasoning trace
        save_reasoning_trace(idx, data.data_id, prompt_with_answer, reason_dir)
        
        generated_code = extract_code_from_response(answer)
        logger.info(f"Extracted code: {generated_code}")
        old_code = generated_code
        
        # Final code verification loop - retry if compilation fails
        if args.monitor and monitors and generated_code:
            final_code, final_compiles, final_output, num_final_retries = asyncio.run(
                monitors[0].verify_final_code(
                    code=generated_code,
                    prompt_with_answer=prompt_with_answer,
                    max_retries=1
                )
            )
            if final_code != generated_code:
                logger.info(f"[Final verification] Code fixed after {num_final_retries} retries")
                generated_code = final_code
        
        # check for soundness
        if args.monitor and monitors and generated_code:
            compiled, _ = monitors[0].sync_verify_compilation(generated_code)
        else:
            compiled=True
        
        # Evaluate - now includes unit tests
        compiles, all_tests_pass, compile_output, test_results = evaluate_generated_code(data, generated_code, idx)
        
        num_tests = len(data.tests) if data.tests else 0
        num_tests_passed = sum(1 for v in test_results.values() if v == "pass")
        
        if compiles and all_tests_pass:
            logger.info(f"✓ PASS - Code compiles and all {num_tests} tests pass")
            num_correct += 1
            logger.info(f"Running Accuracy so far: {(num_correct/(i+1))*100:.2f}%")
        elif compiles:
            logger.info(f"✗ PARTIAL - Code compiles but {num_tests - num_tests_passed}/{num_tests} tests failed")
            logger.debug(f"Test results: {test_results}")
            logger.info(f"Running Accuracy so far: {(num_correct/(i+1))*100:.2f}%")
        else:
            logger.info(f"✗ FAIL - Compilation error")
            logger.debug(f"Compile output: {compile_output[:500]}")
            logger.info(f"Running Accuracy so far: {(num_correct/(i+1))*100:.2f}%")
        logger.info(f"Code Generated: {True if old_code.strip() else False}")
        
        results.append({
            "idx": idx,
            "data_id": data.data_id,
            "compiles": compiles,
            "all_tests_pass": all_tests_pass,
            "num_tests": num_tests,
            "num_tests_passed": num_tests_passed,
            "reasoning_tokens": 0,
            "generated_code": generated_code,
            "num_times_code_forced": monitors[0].get_force_count() if monitors else 0,
            "finally_wrong": not compiled,
        })
    
    # Save final results
    results_csv = os.path.join(output_dirs["base"], "verina_results.csv")
    save_results_csv(results, results_csv)
    
    # Compute average reasoning tokens
    avg_tokens = compute_average_tokens(token_file)
    
    # Compute statistics
    num_compile = sum(1 for r in results if r["compiles"])
    num_all_tests_pass = sum(1 for r in results if r["all_tests_pass"])
    
    # Print summary
    compile_rate = num_compile / N if N > 0 else 0
    accuracy = num_all_tests_pass / N if N > 0 else 0
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"Model: {main_model}")
    print(f"Total examples: {N}")
    print(f"Successful compilations: {num_compile} ({compile_rate:.2%})")
    print(f"All tests pass: {num_all_tests_pass} ({accuracy:.2%})")
    print(f"Average reasoning tokens: {avg_tokens:.2f}")
    print(f"Results saved to: {results_csv}")
    print(f"Reasoning traces saved to: {reason_dir}")
    
    # Save summary
    summary_file = os.path.join(output_dirs["base"], "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "model": main_model,
            "earlystop_model": earlystop_model,
            "total_examples": N,
            "num_compile": num_compile,
            "compile_rate": compile_rate,
            "num_all_tests_pass": num_all_tests_pass,
            "accuracy": accuracy,
            "avg_reasoning_tokens": avg_tokens,
        }, f, indent=2)
    
    logger.info(f"Experiment completed. Summary saved to {summary_file}")
