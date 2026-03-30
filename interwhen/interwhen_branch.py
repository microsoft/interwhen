"""
interwhen_branch.py
───────────────────
A drop-in companion to interwhen's stream_completion that adds *branching*:
after the initial prompt, N candidate continuations are streamed in parallel,
each through its own set of monitors.  A user-supplied (or default) selector
function picks the winning branch and returns it.

Public API
──────────
    branch_completion(prompt, ...)

Mirrors every keyword argument of stream_completion so it can be substituted
without further changes to calling code.

Example
───────
    from interwhen_branch import branch_completion
    from interwhen.monitors import SimpleTextReplaceMonitor

    llm_server = init_llm_server("Qwen/QwQ-32B", port=8000)

    answer = asyncio.run(
        branch_completion(
            prompt,
            llm_server=llm_server,
            num_branches=3,
            monitors=[SimpleTextReplaceMonitor("check", "</think>")],
        )
    )
"""

import asyncio
import httpx
import json
import logging
from typing import Callable, Sequence

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# Internal helpers (taken from interwhen.interject, kept local so
# this file can be dropped into the repo without circular imports)
# ─────────────────────────────────────────────────────────────────

async def _cancel_tasks(tasks: list) -> None:
    """Cancel a list of asyncio Tasks and swallow CancelledError."""
    if not tasks:
        return
    for t in tasks:
        if not t.done():
            t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _stream_one(
    prompt: str,
    prev_text: str = "",
    llm_server: dict | None = None,
    monitors: Sequence = (),
    add_delay: bool = False,
    num_calls_index: int = 0,
    async_execution: bool = True,
) -> str:
    """
    Stream a single completion — identical logic to interwhen.stream_completion.

    Returns the full generated text (prev_text + new tokens).
    Recursive calls handle monitor-triggered corrections.
    """
    stop_event = asyncio.Event()
    stop_info  = {"generated_text": None, "feedback": None, "token_index": None}
    monitor_tasks: list = []

    logger.debug("=" * 50 + f" call #{num_calls_index} " + "=" * 50)

    generated_text = prev_text
    llm_server["payload"]["prompt"] = prompt + prev_text

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            llm_server["url"],
            headers=llm_server["headers"],
            json=llm_server["payload"],
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)["choices"][0]["text"]
                except (json.JSONDecodeError, KeyError, IndexError) as exc:
                    logger.debug("Skipping malformed SSE data: %r (%s)", data, exc)
                    continue

                if stop_event.is_set():
                    break

                generated_text += chunk

                if monitors and not stop_event.is_set():
                    step_flag, step = monitors[0].step_extractor(chunk, generated_text)
                    if step_flag and not stop_event.is_set():
                        task = asyncio.create_task(
                            monitors[0].verify(
                                step, len(generated_text) - len(chunk),
                                stop_event, stop_info,
                            )
                        )
                        monitor_tasks.append(task)
                        if not async_execution:
                            await task

                if add_delay:
                    await asyncio.sleep(0.1)

    # Finalise monitor tasks
    if monitors and async_execution:
        if stop_event.is_set():
            await _cancel_tasks(monitor_tasks)
        else:
            await asyncio.gather(*monitor_tasks, return_exceptions=True)

    if stop_event.is_set():
        if num_calls_index >= 50:
            logger.info("Maximum correction attempts reached.")
            return generated_text

        corrected = await monitors[0].fix(generated_text, stop_info)

        if stop_info.get("feedback") == "\nthe answer is \\boxed{no solution}":
            return corrected
        if stop_info.get("phase") == "final_answer_correct":
            return corrected

        return await _stream_one(
            prompt,
            prev_text=corrected,
            llm_server=llm_server,
            monitors=monitors,
            add_delay=add_delay,
            num_calls_index=num_calls_index + 1,
            async_execution=async_execution,
        )

    return generated_text


# ─────────────────────────────────────────────────────────────────
# Default branch selector
# ─────────────────────────────────────────────────────────────────

def _default_selector(branches: list[str]) -> str:
    """
    Default branch selector: pick the longest branch.

    Replace this with a Nash-equilibrium selector, a reward model call,
    k-stability check, etc.

    Args:
        branches: list of completed branch texts (one per branch)

    Returns:
        The selected branch text
    """
    return max(branches, key=len)


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

async def branch_completion(
    prompt: str,
    prev_text: str = "",
    llm_server: dict | None = None,
    monitors: Sequence = (),
    add_delay: bool = False,
    num_calls_index: int = 0,
    termination_requires_validation: bool = False,
    async_execution: bool = True,
    # ── branching-specific arguments ──────────────────────────────
    num_branches: int = 3,
    branch_monitors: list[Sequence] | None = None,
    selector: Callable[[list[str]], str] | None = None,
) -> str:
    """
    Stream N branches in parallel after the prompt and return the winner.

    Signature is a superset of stream_completion so callers can swap one
    for the other by simply adding the branching kwargs.

    Args:
        prompt:      The full prompt string handed to the LLM.
        prev_text:   Text already generated (e.g. from a previous correction
                     round). Appended to the prompt before generation.
        llm_server:  Dict with keys ``url``, ``headers``, ``payload`` — same
                     format as returned by ``init_llm_server``.
        monitors:    Sequence of VerifyMonitor instances shared across all
                     branches (used when ``branch_monitors`` is None).
        add_delay:   Insert a 0.1 s sleep between chunks (useful for demos).
        num_calls_index: Correction-round counter forwarded from callers.
        termination_requires_validation: Passed through (unused here, kept
                     for API parity with stream_completion).
        async_execution: Whether to run monitor tasks asynchronously.
        num_branches: Number of independent continuations to generate.
        branch_monitors: Optional list of per-branch monitor sequences.
                     If provided, len must equal ``num_branches``.
                     If None, every branch uses the same ``monitors``.
        selector:    Callable ``(List[str]) -> str`` that chooses the winning
                     branch.  Defaults to ``_default_selector`` (longest).

    Returns:
        The text of the selected winning branch.
    """
    if llm_server is None:
        raise ValueError("llm_server must be provided.")

    if selector is None:
        selector = _default_selector

    # Each branch gets its own monitor list; default: share the same monitors.
    if branch_monitors is None:
        branch_monitors = [monitors] * num_branches
    elif len(branch_monitors) != num_branches:
        raise ValueError(
            f"branch_monitors length ({len(branch_monitors)}) "
            f"must equal num_branches ({num_branches})."
        )

    logger.info(
        "branch_completion: spawning %d branches (call #%d)",
        num_branches, num_calls_index,
    )

    # ── deep-copy the mutable payload for each branch so concurrent
    #    tasks don't clobber each other's prompt field ─────────────
    import copy

    branch_tasks = []
    for branch_idx in range(num_branches):
        branch_server = copy.deepcopy(llm_server)
        branch_task = asyncio.create_task(
            _stream_one(
                prompt=prompt,
                prev_text=prev_text,
                llm_server=branch_server,
                monitors=branch_monitors[branch_idx],
                add_delay=add_delay,
                num_calls_index=num_calls_index,
                async_execution=async_execution,
            ),
            name=f"branch_{branch_idx}",
        )
        branch_tasks.append(branch_task)

    results = await asyncio.gather(*branch_tasks, return_exceptions=True)

    # Filter out exceptions, log them
    valid_branches: list[str] = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning("Branch %d raised an exception: %s", i, result)
        else:
            valid_branches.append(result)

    if not valid_branches:
        raise RuntimeError("All branches failed — no valid completion returned.")

    selected = selector(valid_branches)
    logger.info(
        "branch_completion: selected branch (len=%d) from %d valid branches",
        len(selected), len(valid_branches),
    )
    return selected