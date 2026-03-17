"""
ZebraLogic experiment with step verification using the ZebraLogicMonitor.

Uses the new monitor-based architecture that integrates with stream_completion.
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from multiprocessing import Pool

from tqdm import tqdm
from transformers import AutoTokenizer

from interwhen import stream_completion
from interwhen.monitors import ZebraLogicMonitor
from interwhen.utils.zebralogic_helper import (
    get_zebralogic_dataset,
    zebra_correctness,
    extract_last_json,
    SYSTEM_PROMPT_VANILLA,
    USER_PROMPT_TEMPLATE,
)

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level tokenizer (initialized in __main__ for multiprocessing)
tokenizer = None

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


def build_prompt(problem, tok):
    """Build the full generation prompt for a ZebraLogic problem.

    Uses the statefeedback system prompt which instructs the model to:
    - Reason about the problem in text
    - Accept and use feedback if anything is wrong
    - Continue until all variables are assigned
    """
    problem_text = problem['puzzle_clean']
    system_prompt = SYSTEM_PROMPT_VANILLA
    user_prompt = USER_PROMPT_TEMPLATE.format(problem_text=problem_text)

    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True
    )
    return prompt


def run(args, problem):
    """Run a single ZebraLogic problem."""
    global tokenizer
    problem_id = problem['id']
    output_dir = args.output_dir
    output_file = f"{output_dir}/outputs_solver.jsonl"

    llm_server = init_llm_server(args.solver_lm, max_tokens=16 * 1024, port=args.port)
    prompt = build_prompt(problem, tokenizer)

    if args.monitor:
        monitors = [ZebraLogicMonitor(
            name="ZebraLogicMonitor",
            instance=problem,
            llm=args.solver_lm,
            tokenizer=tokenizer,
            step_token=args.step_token,
            step_interval=args.step_interval,
            port=args.port,
            max_corrections=args.monitor_max_corrections,
        )]
    else:
        monitors = []

    output_text = asyncio.run(stream_completion(
        prompt,
        llm_server=llm_server,
        monitors=tuple(monitors) if monitors else [],
        add_delay=False,
        async_execution=not args.debug,
    ))

    # Evaluate correctness
    candidate = extract_last_json(output_text)
    if candidate:
        c, s, m, t = zebra_correctness(problem, candidate)
        accuracy = c / t if t > 0 else 0
    else:
        c, s, m, t, accuracy = 0, 0, 0, 0, 0

    output = {
        'problem_id': problem_id,
        'problem': problem,
        'output_text': output_text,
        'correct': c,
        'skip': s,
        'missing': m,
        'total': t,
        'accuracy': accuracy,
    }
    with open(output_file, "a") as f:
        f.write(json.dumps(output, default=str) + "\n")

    return output


def _run_wrapper(args_problem):
    return run(*args_problem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZebraLogic LLM Solver with Forking Monitor")
    parser.add_argument('--solver_lm', type=str,
                        help='Solver LLM model name')
    parser.add_argument('--port', type=int, default=8000,
                        help='vLLM server port')
    parser.add_argument('--monitor', '-m', action='store_true',
                        help='Enable Forking monitor mode')
    parser.add_argument('--monitor_max_corrections', type=int, default=50,
                        help='Maximum monitor feedback corrections per problem')
    parser.add_argument('--step_token', type=str, default='\n\n',
                        help='Token used to identify steps in the output for monitoring')
    parser.add_argument('--step_interval', type=int, default=25,
                        help='Number of occurrences of the step token before calling the monitor')
    parser.add_argument('--n_processes', '-p', type=int, default=16,
                        help='Number of parallel processes')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug logging and single-process mode')
    parser.add_argument('--continue_from', '-c', type=str, default=None,
                        help='Continue from a specific output directory')
    parser.add_argument('--extra', type=str, default='',
                        help='Extra text description for output directory')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )

    # Initialize tokenizer (module-level for multiprocessing)
    tokenizer = AutoTokenizer.from_pretrained(args.solver_lm)

    # Load dataset
    logger.info("Loading ZebraLogic dataset...")
    ds = get_zebralogic_dataset()

    # Setup output directory
    if args.continue_from:
        output_dir = f'Outputs_TTS/zebralogic/{args.continue_from}'
    else:
        output_dir = f'Outputs_TTS/zebralogic/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        if args.extra:
            output_dir += f'-{args.extra}'
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/args.json', 'w') as f:
            json.dump(vars(args), f, indent=4)

    args.output_dir = output_dir

    output_file = f"{output_dir}/outputs_solver.jsonl"

    if args.continue_from:
        with open(output_file, "r") as f:
            completed_ids = {json.loads(line)['problem']['id'] for line in f}
        ds_run = [p for p in ds if p['id'] not in completed_ids]
        logger.warning(f"Continuing from {args.continue_from}, "
                        f"skipping {len(completed_ids)} completed problems.")
    else:
        with open(output_file, "w") as f:
            f.write("")
        ds_run = ds

    if not args.debug:
        with Pool(processes=args.n_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(_run_wrapper, [(args, p) for p in ds_run]),
                total=len(ds_run),
            ))
    else:
        # Single-process mode for debugging
        _run_wrapper((args, ds_run[0]))

    # Print summary
    output_file = f"{output_dir}/outputs_solver.jsonl"
    if os.path.exists(output_file):
        total, correct = 0, 0
        with open(output_file, "r") as f:
            for line in f:
                result = json.loads(line)
                total += 1
                if result.get('accuracy', 0) == 1.0:
                    correct += 1
        if total > 0:
            print(f"\nResults: {correct}/{total} = {correct/total:.4f} accuracy")
