"""
Nash Equilibrium Step Verifier Example for interwhen.

This example demonstrates multi-agent Nash Equilibrium verification of
reasoning steps from a vision-language model using the interwhen framework.

Three verifier agents (Visual, Logical, Causal) independently score each
candidate reasoning step, and a Nash Equilibrium is computed to select the
most collectively-agreed-upon best step.

Requirements:
- A running vLLM server for the main model (Qwen2.5-VL-7B-Instruct)
- A running vLLM server for the agent model (Qwen3-VL-8B-Instruct)
- vlmeval, deepeval, qwen_vl_utils installed

Usage:
    python nash_example.py --dataset 3DSRBench --num_candidates 3
"""

import argparse
import asyncio
import os
import json
import base64
import random
import logging
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image

from interwhen import stream_completion

# ─────────────────────────── logging ────────────────────────────
logger = logging.getLogger(__name__)


# ─────────────────── LLM server helpers ─────────────────────────

def init_llm_server(model_name: str, max_tokens: int = 1000, port: int = 8000) -> dict:
    """Build the llm_server dict expected by stream_completion."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": 0.8,
        "top_p": 0.6,
        "do_sample": True,
        "stream": True,
        "stop": ["\n", "<|im_end|>"],
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


def init_agent_server(model_name: str, max_tokens: int = 100, port: int = 8001) -> dict:
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


# ─────────────────────── utility functions ───────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def pil_to_base64(image_path: str, fmt: str = "JPEG") -> str:
    pil_image = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    pil_image.save(buffer, format=fmt)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def write_jsonl(path: str, records: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_python(obj):
    """Convert numpy scalars to native Python types for JSON serialisation."""
    if isinstance(obj, dict):
        return {str(k): to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def after_think(text: str) -> str:
    """Strip everything up to and including the closing </think> tag."""
    if "</think>" not in text:
        return text
    return text.split("</think>", 1)[1].lstrip()


# ─────────────────────── verifier prompts ───────────────────────

VA_SYSTEM = """\
You are a Visual Verification Specialist. Your task is to verify if a
reasoning step accurately describes the image provided.

Scoring guidelines:
- 1.0 (Confirmed): Object and spatial relation are clearly visible.
- 0.8 (Highly Likely): Relative terms match a human observer's perspective.
- 0.5 (Ambiguous): Object exists but spatial description is vague.
- 0.0 (False): Object missing or description contradicts visual evidence.

Output ONLY a number between 0.0 and 1.0."""

VA_PROMPT = """\
TASK: Audit the "Current Reasoning Step" for visual accuracy using ONLY the image.

QUESTION: {question}
PREVIOUS REASONING STEPS (ASSUMED CORRECT): {previous_steps}
CURRENT REASONING STEP TO VERIFY: {current_step}

OUTPUT: A single number between 0.0 and 1.0."""

LA_SYSTEM = """\
You are a Formal Logic Auditor. Determine the logical validity of the
"Current Step" based strictly on provided premises.

Rules:
1. INTERNAL CONSISTENCY: evaluate from Question and Previous Steps only.
2. NO EXTERNAL KNOWLEDGE: outside facts lower the score.
3. FORMAL VALIDITY: penalise logical fallacies.

Output MUST be a single float between 0.0 and 1.0."""

LA_PROMPT = """\
QUESTION: {question}
PREVIOUS REASONING STEPS (ASSUMED CORRECT): {previous_steps}
CURRENT REASONING STEP TO VERIFY: {current_step}

Rubric:
- 1.0: Strict entailment
- 0.7-0.9: Strong inference
- 0.4-0.6: Weak inference
- 0.1-0.3: Logical leap
- 0.0: Contradiction

Output ONLY the numerical score."""

CA_SYSTEM = """\
You are a Causal Logic Auditor. Evaluate whether a reasoning step is a
valid link in a causal chain.

OUTPUT RULE: Exactly one real number between 0.0 and 1.0. No text."""

CA_PROMPT = """\
QUESTION: {question}
PREVIOUS REASONING STEPS (ASSUMED CORRECT): {previous_steps}
CURRENT REASONING STEP TO VERIFY: {current_step}

Rubric:
- 1.0: Essential causal link
- 0.7-0.9: Plausible step
- 0.5: Neutral/informational
- 0.2-0.4: Weak/irrelevant
- 0.0: Causal break

Output only the numerical score."""


async def _score_one(agent_server: dict, system: str, user: str) -> float:
    import copy
    server = copy.deepcopy(agent_server)
    server["payload"]["messages"] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    # remove 'prompt' key if present — chat endpoint uses 'messages'
    server["payload"].pop("prompt", None)

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            server["url"],
            headers=server["headers"],
            json=server["payload"],
        ) as response:
            text = ""
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                    text += delta
                except (json.JSONDecodeError, KeyError):
                    continue

    try:
        return float(text.strip())
    except ValueError:
        logger.warning("Score parsing failed for output: %r", text)
        return 0.5


async def get_scores(
    agent_server: dict,
    image_b64: str,
    question: str,
    prev_steps: str,
    curr_step: str,
) -> dict:
    """
    Compute VA, LA, CA scores for a single candidate step.
    The image_b64 is embedded in the user prompt for VL models.
    """
    image_tag = f"<img src='{image_b64}'/>\n"

    va_user = image_tag + VA_PROMPT.format(
        question=question, previous_steps=prev_steps, current_step=curr_step
    )
    la_user = image_tag + LA_PROMPT.format(
        question=question, previous_steps=prev_steps, current_step=curr_step
    )
    ca_user = image_tag + CA_PROMPT.format(
        question=question, previous_steps=prev_steps, current_step=curr_step
    )

    va, la, ca = await asyncio.gather(
        _score_one(agent_server, VA_SYSTEM, va_user),
        _score_one(agent_server, LA_SYSTEM, la_user),
        _score_one(agent_server, CA_SYSTEM, ca_user),
    )
    return {"V": va, "L": la, "C": ca}


# ─────────────────── Nash equilibrium core ───────────────────────

def compute_nash_equilibrium(raw_scores: dict, lambdas: dict) -> dict:
    """
    Solve the simultaneous linear system:
        (1 + λ_i) s*_i  −  (1/(n−1)) Σ_{j≠i} s*_j  =  λ_i ŝ_i

    Args:
        raw_scores: {agent: raw_score}
        lambdas:    {agent: λ}  honesty weight per agent

    Returns:
        {agent: equilibrium_score}  clipped to [0, 1]
    """
    agents = list(raw_scores.keys())
    n = len(agents)

    A = np.zeros((n, n))
    for i, a_i in enumerate(agents):
        for j in range(n):
            A[i, j] = (1 + lambdas[a_i]) if i == j else -1.0 / (n - 1)

    B = np.array([lambdas[agents[i]] * raw_scores[agents[i]] for i in range(n)])

    try:
        s_star = np.linalg.solve(A, B)
    except np.linalg.LinAlgError:
        s_star = np.array(list(raw_scores.values()))

    return dict(zip(agents, np.clip(s_star, 0.0, 1.0)))


def select_best_step(
    all_scores: dict,
    lambdas: dict | None = None,
    epsilon: float = 0.1,
    tau: float = 0.6,
) -> tuple[int | None, dict]:
    """
    Select the best candidate step using Nash Equilibrium stability.

    Args:
        all_scores: {step_id: {agent: raw_score}}
        lambdas:    per-agent honesty weights (default: V=1.0, L=1.5, C=0.8)
        epsilon:    max dispersion for a step to be "accepted"
        tau:        min mean score for a step to be "accepted"

    Returns:
        (best_step_id, full_results_dict)
    """
    if lambdas is None:
        lambdas = {"V": 1.0, "L": 1.5, "C": 0.8}

    results = {}
    for step_id, agent_scores in all_scores.items():
        s_star = compute_nash_equilibrium(agent_scores, lambdas)
        vals = np.array(list(s_star.values()))
        mean_conf = float(vals.mean())
        dispersion = float(np.abs(vals - mean_conf).mean())
        accepted = (dispersion < epsilon) and (mean_conf > tau)
        results[step_id] = {
            "s_star": s_star,
            "mean": round(mean_conf, 4),
            "dispersion": round(dispersion, 4),
            "accepted": accepted,
        }

    valid = {k: v for k, v in results.items() if v["accepted"]}
    if not valid:
        best_id = max(results, key=lambda k: results[k]["mean"] - results[k]["dispersion"])
    else:
        best_id = max(valid, key=lambda k: valid[k]["mean"])

    return best_id, results


# ─────────────────── per-sample reasoning loop ───────────────────

async def run_sample(
    idx: int,
    question: str,
    image_b64: str,
    gt_answer: str,
    llm_server: dict,
    agent_server: dict,
    num_candidates: int = 3,
    max_steps: int = 20,
) -> dict:
    """
    Run Nash-guided step-by-step reasoning for a single VQA sample.

    Generates candidate next-steps, scores them with three verifier agents,
    computes Nash Equilibrium, picks the best step, and repeats until EOS.
    """
    image_tag = f"<img src='{image_b64}'/>\n"
    base_prompt = f"{image_tag}{question}. Reason step by step."

    # ── first token stream: kick off reasoning ────────────────────
    total_response = await stream_completion(
        base_prompt, llm_server=llm_server, monitors=[]
    )

    allstep_scores: list = []
    allstep_diagnostics: list = []
    allstep_candidates: list = []

    for step_idx in range(max_steps):
        # Check for EOS in last generated output
        if "<|im_end|>" in total_response or total_response.endswith("</s>"):
            logger.info("EOS detected at step %d", step_idx)
            break

        # ── generate num_candidates next steps ───────────────────
        scores: dict = {}
        candidates: dict = {}

        candidate_tasks = [
            stream_completion(
                base_prompt + total_response,
                llm_server=llm_server,
                monitors=[],
            )
            for _ in range(num_candidates)
        ]
        candidate_outputs = await asyncio.gather(*candidate_tasks, return_exceptions=True)

        for i, output in enumerate(candidate_outputs):
            if isinstance(output, Exception):
                logger.warning("Candidate %d failed: %s", i, output)
                continue
            candidate_response = output

            try:
                agent_scores = await get_scores(
                    agent_server,
                    image_b64,
                    question,
                    total_response,
                    candidate_response,
                )
                scores[i] = agent_scores
                candidates[i] = candidate_response
            except Exception as e:
                logger.warning("Score computation failed for candidate %d: %s", i, e)

        if not scores:
            logger.warning("No valid candidates at step %d; breaking.", step_idx)
            break

        # ── Nash selection ────────────────────────────────────────
        best_step, diagnostics = select_best_step(scores)
        if best_step is None:
            best_step = 0  # fallback: pick first candidate

        selected = candidates[best_step]
        total_response += selected

        allstep_scores.append(scores)
        allstep_diagnostics.append(diagnostics)
        allstep_candidates.append(candidates)

        logger.info(
            "[%d] step=%d | best_candidate=%d | mean=%.3f | disp=%.3f",
            idx, step_idx, best_step,
            diagnostics[best_step]["mean"],
            diagnostics[best_step]["dispersion"],
        )

    extracted = after_think(total_response)

    return {
        "idx": idx,
        "question": question,
        "ground_truth": gt_answer,
        "total_response": total_response,
        "extracted_answer": extracted,
        "all_step_scores": allstep_scores,
        "all_step_diagnostics": to_python(allstep_diagnostics),
        "all_step_candidates": allstep_candidates,
        "num_steps": len(allstep_scores),
    }


async def main(args):
    from vlmeval.dataset import build_dataset  # imported here to keep top-level clean

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    llm_server   = init_llm_server(args.main_model,  port=args.main_port)
    agent_server = init_agent_server(args.agent_model, port=args.agent_port)

    # ── load dataset ─────────────────────────────────────────────
    ds = build_dataset(args.dataset)
    if ds is None:
        raise ValueError(f"Unknown dataset: {args.dataset!r}")
    logger.info("Loaded %s | size=%d", args.dataset, len(ds))

    output_dir = os.path.join("exp_out", "nash_equilibrium")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{args.dataset}.jsonl")

    records = []
    N = min(args.num_examples, len(ds))

    for idx in range(N):
        prmpt      = ds.build_prompt(idx)
        question   = next((m["value"] for m in prmpt if m["type"] == "text"), None)
        image_path = next((m["value"] for m in prmpt if m["type"] == "image"), None)
        gt_answer  = ds[idx]["answer"]

        if image_path is None or question is None:
            logger.warning("Skipping idx=%d: missing image or question", idx)
            continue

        image_b64 = pil_to_base64(image_path)

        logger.info("── [%d/%d] idx=%d ──", idx + 1, N, idx)
        try:
            row = await run_sample(
                idx, question, image_b64, gt_answer,
                llm_server, agent_server,
                num_candidates=args.num_candidates,
                max_steps=args.max_steps,
            )
        except Exception as e:
            logger.error("Failed idx=%d: %s", idx, e)
            continue

        records.append(row)
        logger.info("[%d] extracted_answer=%r", idx, row["extracted_answer"][:80])

        if (idx + 1) % args.checkpoint_interval == 0:
            write_jsonl(output_file + ".ckpt", records)
            logger.info("Checkpoint saved (%d records)", len(records))

    write_jsonl(output_file, records)
    logger.info("Saved %d records to %s", len(records), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nash Equilibrium step verification with interwhen"
    )
    parser.add_argument("--dataset",             type=str,  default="3DSRBench")
    parser.add_argument("--num_examples",  "-n", type=int,  default=100)
    parser.add_argument("--num_candidates",       type=int,  default=3)
    parser.add_argument("--max_steps",            type=int,  default=20)
    parser.add_argument("--checkpoint_interval",  type=int,  default=50)
    parser.add_argument("--main_model",   type=str,  default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--agent_model",  type=str,  default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--main_port",    type=int,  default=8000)
    parser.add_argument("--agent_port",   type=int,  default=8001)
    parser.add_argument("--debug", "-d",  action="store_true")
    args = parser.parse_args()

    asyncio.run(main(args))