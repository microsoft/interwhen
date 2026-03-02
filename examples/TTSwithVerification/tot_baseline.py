#!/usr/bin/env python3
"""Command-line Tree-of-Thought baseline runner for interwhen datasets."""

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import numpy as np
from datasets import load_dataset

from interwhen.tree_of_thought import (
    SearchMethod,
    ToTSearchConfig,
    TreeOfThoughtSearch,
    build_tot_problem,
)

LOGGER = logging.getLogger("tot_baseline")


# ============== Helper Functions ==============

def remove_last_paragraph(s: str) -> str:
    return s[:-143] if len(s) > 143 else s


def build_maze_prompt(example):
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following multiple-choice question "
        "and think through the solution step-by-step before providing your final answer. "
        "Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
    )
    description = remove_last_paragraph(str(example.get("prompt")))
    return pre_prompt, description


def build_spatialmap_prompt(example):
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following multiple-choice question "
        "and think through the solution step-by-step before providing your final answer." 
        "Provide your final answer option by enclosing it within \\boxed{A/B/C/D}.:"
    )
    description = remove_last_paragraph(str(example.get("prompt")))
    return pre_prompt, description


def extract_solution_game24(text):
    boxed_pattern = r"\\boxed\{"
    matches = list(re.finditer(boxed_pattern, text))
    if not matches:
        return None
    last_match = matches[-1]
    start = last_match.end()
    brace_count = 1
    end = start
    while end < len(text) and brace_count > 0:
        if text[end] == "{":
            brace_count += 1
        elif text[end] == "}":
            brace_count -= 1
        end += 1
    expr = text[start:end - 1].strip()

    frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    while re.search(frac_pattern, expr):
        expr = re.sub(frac_pattern, r"(\1/\2)", expr)

    replacements = {
        r"\times": "*",
        r"\cdot": "*",
        r"\div": "/",
    }
    for latex, op in replacements.items():
        expr = expr.replace(latex, op)

    expr = expr.replace(r"\\,", "").replace(r"\\ ", "")
    expr = re.sub(r"\)\s*\(", ")*(", expr)
    expr = re.sub(r"\)\s*(\d)", r")*\1", expr)
    expr = re.sub(r"(\d)\s*\(", r"\1*(", expr)

    return expr


def extract_numbers_from_expr(expr):
    numbers = re.findall(r"\d+\.?\d*", expr)
    return [int(float(n)) if float(n).is_integer() else float(n) for n in numbers]


def validate_numbers_used(expr, expected_nums):
    used_nums = extract_numbers_from_expr(expr)
    return sorted(used_nums) == sorted(expected_nums)


def evaluate_expression(expr, expected_nums=None):
    try:
        if expected_nums is not None and not validate_numbers_used(expr, expected_nums):
            return False
        value = eval(expr, {"__builtins__": None}, {})
        return abs(value - 24) < 1e-6
    except Exception:
        return False


def evaluate_game24_answer(answer, nums):
    expr = extract_solution_game24(answer)
    if not expr:
        return False, None, "No expression found"
    if evaluate_expression(expr, expected_nums=nums):
        return True, expr, "Correct solution (evaluates to 24 using exactly the given numbers)"
    used_nums = extract_numbers_from_expr(expr)
    if sorted(used_nums) != sorted(nums):
        return False, expr, f"Incorrect: Expression uses {used_nums}, expected {nums}"
    return False, expr, "Expression does not evaluate to 24"


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
    
    standalone = re.findall(r"\b([ABCD])\b", text)
    if standalone:
        return standalone[-1].upper()
    
    return None


def extract_options_from_prompt(prompt_text, target_options):
    pattern = r"\b([A-D])\.\s*(.*?)(?=\s*[A-D]\.\s*|$)"
    raw = re.findall(pattern, prompt_text, flags=re.DOTALL)
    options = {k: v.strip().rstrip(".") for k, v in raw}
    if target_options:
        options = {k: v for k, v in options.items() if k in target_options}
    return options


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


def extract_solution_zebralogic(text):
    """Extract JSON solution from ZebraLogic model output."""
    if not text:
        return None

    def _try_parse(candidate: str):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                # Unwrap if it's a wrapper with "answer" key
                if "answer" in parsed and isinstance(parsed["answer"], dict):
                    inner = parsed["answer"]
                    if any(re.match(r"^house\s*\d+$", str(k).strip(), flags=re.IGNORECASE) for k in inner.keys()):
                        return inner
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    # Try to extract JSON from code blocks
    patterns = [
        r"```json\s*({.*?})\s*```",  # Markdown code block
        r"```\s*({.*?})\s*```",  # Generic code block
        r"({\s*['\"]House\s*\d+['\"].*?})",  # Direct JSON starting with House
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            json_str = matches[-1].strip()
            solution = _try_parse(json_str)
            if solution is not None:
                return solution
    
    # Try parsing entire last large JSON-like structure
    try:
        # Find potential JSON starting with {
        json_match = re.search(r"({\s*(?:['\"]House|['{\"\[])+[\s\S]*})", text)
        if json_match:
            json_str = json_match.group(1)
            solution = _try_parse(json_str)
            if solution is not None:
                return solution
    except (json.JSONDecodeError, AttributeError):
        pass

    # Last-chance extraction: parse top-level JSON object spans and keep the
    # last one that parses and looks like a house assignment dictionary.
    stack = []
    spans = []
    for idx, ch in enumerate(text):
        if ch == "{":
            stack.append(idx)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                spans.append((start, idx + 1))
    for start, end in reversed(spans):
        candidate = text[start:end]
        solution = _try_parse(candidate)
        if solution is not None:
            # Handle wrapped solution with "answer" key
            if isinstance(solution, dict) and "answer" in solution:
                answer = solution["answer"]
                if isinstance(answer, dict) and any(
                    re.match(r"^house\s*\d+$", str(key).strip(), flags=re.IGNORECASE)
                    for key in answer.keys()
                ):
                    return answer
            # Direct house keys
            if any(
                re.match(r"^house\s*\d+$", str(key).strip(), flags=re.IGNORECASE)
                for key in solution.keys()
            ):
                return solution
    
    return None


async def _request_zebralogic_json(prompt: str, llm_server: Dict[str, Any]) -> str:
    """Submit a strict-JSON request for ZebraLogic and return raw model content."""
    payload = dict(llm_server["payload"])
    payload["temperature"] = 0.0
    payload["messages"] = [
        {
            "role": "system",
            "content": (
                "You solve Zebra Logic puzzles and MUST return strictly valid JSON only. "
                "No markdown fences. No explanation. No extra text."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    payload["response_format"] = {"type": "json_object"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            llm_server["url"],
            headers=llm_server["headers"],
            json=payload,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"].strip()


async def finalize_zebralogic_json(problem: str, trajectory: str, llm_server: Dict[str, Any]) -> str:
    """Ask the model to convert an existing trajectory into strict final JSON only."""
    prompt = (
        "Convert the reasoning into the final Zebra Logic answer JSON.\n"
        "Output ONLY valid JSON (no markdown, no explanation).\n"
        "Use exact feature/value names from the puzzle.\n\n"
        "PUZZLE:\n"
        f"{problem}\n\n"
        "REASONING:\n"
        f"{trajectory}\n"
    )
    return await _request_zebralogic_json(prompt, llm_server)


async def solve_zebralogic_json_direct(problem: str, llm_server: Dict[str, Any]) -> str:
    """Directly solve ZebraLogic and return strict final JSON."""
    prompt = (
        "Solve the Zebra Logic puzzle and provide the final house assignments.\n"
        "Output ONLY valid JSON with keys like 'House 1', 'House 2', etc.\n"
        "Use exact feature/value names from the puzzle text.\n\n"
        "PUZZLE:\n"
        f"{problem}\n"
    )
    return await _request_zebralogic_json(prompt, llm_server)


def evaluate_zebralogic_answer(answer, ground_truth):
    """Evaluate ZebraLogic solution against ground truth."""
    extracted = extract_solution_zebralogic(answer)
    
    if extracted is None:
        return False, None, "Could not extract valid JSON solution"
    
    # Normalize keys (handle 'House X', 'house x', etc.)
    def normalize_solution(sol):
        normalized = {}
        for key, value in sol.items():
            # Normalize house key
            house_match = re.search(r"House\s*(\d+)", key, re.IGNORECASE)
            if house_match:
                house_num = house_match.group(1)
                normalized[f"House {house_num}"] = value if isinstance(value, dict) else value
            else:
                normalized[key] = value
        return normalized
    
    extracted_norm = normalize_solution(extracted)
    ground_truth_norm = normalize_solution(ground_truth) if isinstance(ground_truth, dict) else ground_truth
    
    # Simple exact match on normalized solution
    if extracted_norm == ground_truth_norm:
        return True, extracted_norm, "Correct: Solution matches ground truth exactly"
    
    # Check if as string they're close (for JSON format differences)
    extracted_str = json.dumps(extracted_norm, sort_keys=True)
    gt_str = json.dumps(ground_truth_norm, sort_keys=True) if isinstance(ground_truth_norm, dict) else str(ground_truth_norm)
    
    if extracted_str == gt_str:
        return True, extracted_norm, "Correct: Solution matches ground truth (format normalized)"
    
    # Partial credit: check if majority of houses match
    if isinstance(ground_truth_norm, dict) and isinstance(extracted_norm, dict):
        matches = sum(1 for k in extracted_norm if k in ground_truth_norm and extracted_norm[k] == ground_truth_norm[k])
        total = max(len(extracted_norm), len(ground_truth_norm))
        if matches > 0:
            accuracy = matches / total
            return False, extracted_norm, f"Partial match: {matches}/{total} houses correct ({accuracy:.1%})"
    
    return False, extracted_norm, "Incorrect: Solution does not match ground truth"


def load_dataset_for_task(task):
    if task == "game24":
        return load_dataset("nlile/24-game", split="train")
    if task == "maze":
        return load_dataset("microsoft/VISION_LANGUAGE", "maze_text_only", split="val")
    if task == "spatialmap":
        return load_dataset("microsoft/VISION_LANGUAGE", "spatial_map_text_only", split="val")
    if task == "zebralogic":
        return load_dataset("WildEval/ZebraLogic", name="grid_mode", split="test")
    raise ValueError(f"Unsupported task: {task}")


def resolve_indices(task, dataset_len, args):
    if args.indices:
        return [int(x.strip()) for x in args.indices.split(",")]
    if args.xrange:
        parts = args.xrange.split("-")
        if len(parts) == 2:
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                return list(range(start, end))
            except ValueError:
                raise ValueError(f"Invalid xrange format: {args.xrange}. Use 'start-end'")
    if args.num_examples:
        return list(np.linspace(0, dataset_len - 1, args.num_examples, dtype=int))
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else dataset_len
    return list(range(start, end))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tree-of-Thought search on a subset of the supported tasks",
    )
    parser.add_argument("--task", choices=["game24", "maze", "spatialmap", "zebralogic"], required=True)
    parser.add_argument("--k", type=int, default=1, help="Unused placeholder to mirror other baselines")
    parser.add_argument("--num_examples", "-n", type=int, default=None)
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--xrange", type=str, default=None)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--model", default="Qwen/QwQ-32B")
    parser.add_argument("--llm_url", default="http://localhost:{port}/v1/chat/completions")
    parser.add_argument(
        "--ports",
        default="8000",
        help="Comma-separated list of vLLM ports to round-robin across",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search_method", choices=["bfs", "dfs", "beam"], default="beam")
    parser.add_argument("--branching_factor", type=int, default=4)
    parser.add_argument("--max_depth", type=int, default=1)
    parser.add_argument("--beam_width", type=int, default=2)
    parser.add_argument("--sure_threshold", type=float, default=0.7)
    parser.add_argument("--likely_threshold", type=float, default=0.5)
    parser.add_argument("--impossible_threshold", type=float, default=0.2)
    parser.add_argument("--max_candidates_per_level", type=int, default=3)
    parser.add_argument("--early_termination", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of ToT examples to run concurrently",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/tot_baseline",
        help="Directory to store per-example JSON logs and summary",
    )
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def parse_port_list(port_str: str) -> List[int]:
    return [int(p.strip()) for p in port_str.split(",") if p.strip()]


def build_llm_server(args: argparse.Namespace, port: int) -> Dict[str, Any]:
    payload = {
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "stream": False,
        "seed": args.seed,
    }
    return {
        "url": args.llm_url.format(port=port),
        "headers": {"content-type": "application/json"},
        "payload": payload,
    }


def build_tot_config(args: argparse.Namespace) -> ToTSearchConfig:
    method = SearchMethod[args.search_method.upper()]
    return ToTSearchConfig(
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        search_method=method,
        beam_width=args.beam_width,
        sure_threshold=args.sure_threshold,
        likely_threshold=args.likely_threshold,
        impossible_threshold=args.impossible_threshold,
        early_termination=args.early_termination,
        cache_evaluations=not args.no_cache,
        max_candidates_per_level=args.max_candidates_per_level,
    )


def ensure_output_dir(base_dir: str, task: str) -> Path:
    path = Path(base_dir).expanduser().resolve() / task
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_eval(task: str, example: Dict[str, Any]) -> Tuple:
    if task == "game24":
        nums = list(example.get("numbers", []))
        return (lambda output: evaluate_game24_answer(output, nums), {"numbers": nums})
    if task == "zebralogic":
        # ZebraLogic ground truth is the solution JSON
        ground_truth = example.get("solution", {})
        meta = {"ground_truth_sample": str(ground_truth)[:100]}
        return (lambda output: evaluate_zebralogic_answer(output, ground_truth), meta)
    gt = str(example.get("ground_truth", "")).strip()
    target_options = ["A", "B"] if gt == "Q4" else ["A", "B", "C", "D"]
    if task == "maze":
        _, user_prompt = build_maze_prompt(example)
    else:
        _, user_prompt = build_spatialmap_prompt(example)
    options = extract_options_from_prompt(user_prompt, target_options)
    meta = {"options": options, "ground_truth": gt}
    return (lambda output: evaluate_mcq_answer(output, options, gt), meta)


async def run_single_example(
    idx: int,
    task: str,
    example: Dict[str, Any],
    tot_config: ToTSearchConfig,
    llm_server: Dict[str, Any],
) -> Dict[str, Any]:
    eval_fn, eval_meta = prepare_eval(task, example)
    problem = build_tot_problem(task, example, nums=example.get("numbers"))
    tot = TreeOfThoughtSearch(tot_config)
    search_result = await tot.search(task, problem, llm_server)
    best_traj = search_result.get("best_trajectory", "")
    best_value = search_result.get("best_value", 0.0)
    is_correct, extracted, message = eval_fn(best_traj)

    # ZebraLogic often ends with partial reasoning trajectories; add strict-JSON
    # recovery passes before scoring.
    finalized_answer = None
    direct_answer = None
    if task == "zebralogic" and (not is_correct):
        try:
            finalized_answer = await finalize_zebralogic_json(problem, best_traj, llm_server)
            final_is_correct, final_extracted, final_message = eval_fn(finalized_answer)
            if final_extracted is not None or final_is_correct:
                is_correct = final_is_correct
                extracted = final_extracted
                message = final_message
                best_traj = finalized_answer
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("ZebraLogic finalization failed for index %s: %s", idx, exc)

    if task == "zebralogic" and (not is_correct):
        try:
            direct_answer = await solve_zebralogic_json_direct(problem, llm_server)
            direct_is_correct, direct_extracted, direct_message = eval_fn(direct_answer)
            if direct_extracted is not None or direct_is_correct:
                is_correct = direct_is_correct
                extracted = direct_extracted
                message = direct_message
                best_traj = direct_answer
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("ZebraLogic direct solve failed for index %s: %s", idx, exc)

    return {
        "index": int(idx),
        "best_value": best_value,
        "best_trajectory": best_traj,
        "raw_best_trajectory": search_result.get("best_trajectory", ""),
        "finalized_answer": finalized_answer,
        "direct_answer": direct_answer,
        "search_stats": search_result.get("search_stats", {}),
        "decision_tree": search_result.get("decision_tree", []),
        "correct": bool(is_correct),
        "extracted": extracted,
        "message": message,
        "evaluation_meta": eval_meta,
    }


async def run_tot_baseline(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    dataset = load_dataset_for_task(args.task)
    indices = resolve_indices(args.task, len(dataset), args)
    output_dir = ensure_output_dir(args.output_dir, args.task)
    tot_config = build_tot_config(args)
    ports = parse_port_list(args.ports)
    if not ports:
        raise ValueError("At least one port must be specified via --ports")
    concurrency = max(1, args.concurrency)
    port_lock = asyncio.Lock()
    port_index = {"value": 0}

    async def next_port() -> int:
        async with port_lock:
            port = ports[port_index["value"] % len(ports)]
            port_index["value"] += 1
            return port

    semaphore = asyncio.Semaphore(concurrency)

    async def process_index(idx: int) -> Dict[str, Any]:
        async with semaphore:
            example = dataset[int(idx)]
            port = await next_port()
            llm_server = build_llm_server(args, port)
            LOGGER.info("Running ToT on example %s via port %s", idx, port)
            try:
                record = await run_single_example(idx, args.task, example, tot_config, llm_server)
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Failed example %s", idx)
                record = {
                    "index": int(idx),
                    "error": str(exc),
                    "best_trajectory": "",
                    "correct": False,
                }
            example_path = output_dir / f"example_{idx}.json"
            with example_path.open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            return record

    processed = await asyncio.gather(*[process_index(idx) for idx in indices])

    total = len(processed)
    correct = sum(1 for r in processed if r.get("correct"))
    summary = {
        "task": args.task,
        "model": args.model,
        "total_examples": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "search_method": args.search_method,
        "config": {
            "branching_factor": args.branching_factor,
            "max_depth": args.max_depth,
            "beam_width": args.beam_width,
            "sure_threshold": args.sure_threshold,
            "likely_threshold": args.likely_threshold,
            "impossible_threshold": args.impossible_threshold,
            "max_candidates_per_level": args.max_candidates_per_level,
            "early_termination": args.early_termination,
            "cache_evaluations": not args.no_cache,
            "ports": ports,
            "concurrency": concurrency,
        },
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Accuracy %.2f (%d/%d)", summary["accuracy"], correct, total)


if __name__ == "__main__":
    asyncio.run(run_tot_baseline(parse_args()))