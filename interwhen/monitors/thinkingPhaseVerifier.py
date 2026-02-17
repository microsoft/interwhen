"""
Thinking Phase Step Verifier for Game of 24.

This monitor verifies Game-of-24 solutions by injecting a structured-output
prompt after ``</think>`` — whether that ``</think>`` was forced by us (during
the thinking phase) or produced naturally by the model.

Workflow
--------
A) **DURING the thinking phase** (inside ``<think>...</think>``):
   After every *N* newlines in the thinking trace:
   1. Inject ``\\n</think>\\n`` + a *structured output prompt* that asks the
      model to list its current solution as verified steps.
   2. Stream from the vLLM server to collect those steps.
   3. Verify each step with the existing ``verify_step`` utilities.
   4. If **wrong** -> remove the injected ``</think>`` + prompt + generated
      steps, append ``Wait, that approach is wrong. ...`` inside the thinking
      trace, and let the model keep thinking.
   5. If **correct** -> keep ``</think>`` + the structured prompt, let
      ``stream_completion`` recurse so the model finishes (Phase B verifies
      each step as it streams).

B) **AFTER a natural ``</think>``**:
   Inject the same structured output prompt so the model outputs its steps
   in the verifiable format. Then behave identically to
   ``StepVerifierGame24Monitor`` — verify each step as it appears and give
   ``[VERIFIER FEEDBACK ...]`` on the first error so the model retries.
"""

import re
import json
import logging
from typing import List, Optional, Tuple, Dict, Any
from copy import deepcopy

import httpx

from .base import VerifyMonitor
from ..utils.game24_verifier import (
    parse_step, verify_step, format_feedback, can_reach_24, format_number
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
#  The structured-output prompt injected right after </think>.
#  It tells the model to present its solution in the step format that
#  our verifier can parse, AND to put the final answer in \boxed{}.
# ──────────────────────────────────────────────────────────────────────
STRUCTURED_OUTPUT_PROMPT = (
    "Now present your solution step-by-step in the following format. "
    "Use ALL four numbers exactly once with +, -, *, / to reach 24.\n"
    "\n"
    ">Step1\n"
    "available numbers: [a, b, c, d]\n"
    "suggested operation: a * b = result1\n"
    "remaining numbers: [result1, c, d]\n"
    "\n"
    ">Step2\n"
    "available numbers: [result1, c, d]\n"
    "suggested operation: result1 + c = result2\n"
    "remaining numbers: [result2, d]\n"
    "\n"
    ">Step3\n"
    "available numbers: [result2, d]\n"
    "suggested operation: result2 - d = result3\n"
    "remaining numbers: [result3]\n"
    "\n"
    "> Final expression: \\boxed{expression using original numbers}"
)


class ThinkingPhaseStepVerifierGame24Monitor(VerifyMonitor):
    """
    Monitor that adds thinking-phase verification on top of the standard
    StepVerifierGame24Monitor behaviour.

    During thinking: every N newlines -> force ``</think>`` + structured
        prompt, stream from vLLM, verify.  Roll back on error, commit on
        success.

    After natural ``</think>``: inject the same structured prompt, then
        verify each step (identical to StepVerifierGame24Monitor).
    """

    def __init__(
        self,
        name: str,
        original_numbers: List[int],
        llm_server: dict,
        prompt: str,
        newline_threshold: int = 15,
        max_corrections: int = 5,
        answer_start_token: str = "</think>",
        async_execution: bool = True,
    ):
        super().__init__(name)
        self.original_numbers = [float(x) for x in original_numbers]
        self.llm_server = llm_server
        self.prompt = prompt
        self.newline_threshold = newline_threshold
        self.max_corrections = max_corrections
        self.answer_start_token = answer_start_token
        self.async_execution = async_execution

        # ---- thinking-phase state ----
        self._think_phase_corrections = 0

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------
    def _count_feedback_blocks(self, text: str) -> int:
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))

    def _is_in_thinking_phase(self, generated_text: str) -> bool:
        return self.answer_start_token not in generated_text

    # ------------------------------------------------------------------
    #  _get_current_available / _extract_last_step_info
    #  (identical to StepVerifierGame24Monitor)
    # ------------------------------------------------------------------
    def _get_current_available(self, generated_text: str) -> List[float]:
        if self.answer_start_token not in generated_text:
            return self.original_numbers.copy()

        text_after_think = generated_text.split(self.answer_start_token)[-1]

        step_pattern = re.compile(
            r'>\s*Step\s*(\d+)\s*\n'
            r'available\s+numbers?\s*:\s*\[([^\]]+)\]\s*\n'
            r'suggested\s+operation\s*:\s*([^\n]+?)\s*\n'
            r'remaining\s+numbers?\s*:\s*\[([^\]]+)\]',
            re.IGNORECASE,
        )

        sections = re.split(
            r'\[VERIFIER FEEDBACK[^\]]*\]\s*', text_after_think, flags=re.DOTALL
        )
        last_section = sections[-1]
        steps_in_last_section = list(step_pattern.finditer(last_section))

        if not steps_in_last_section:
            return self.original_numbers.copy()

        last_step = steps_in_last_section[-1]
        step_num_to_verify = int(last_step.group(1))

        if step_num_to_verify == 1:
            return self.original_numbers.copy()

        target_step = step_num_to_verify - 1

        for step_match in steps_in_last_section[:-1]:
            if int(step_match.group(1)) == target_step:
                try:
                    return [
                        float(x.strip())
                        for x in step_match.group(4).strip().split(',')
                        if x.strip()
                    ]
                except Exception:
                    pass

        for section in reversed(sections[:-1]):
            for step_match in reversed(list(step_pattern.finditer(section))):
                if int(step_match.group(1)) == target_step:
                    try:
                        return [
                            float(x.strip())
                            for x in step_match.group(4).strip().split(',')
                            if x.strip()
                        ]
                    except Exception:
                        pass

        return self.original_numbers.copy()

    def _extract_last_step_info(self, generated_text: str):
        if self.answer_start_token not in generated_text:
            return None, None

        text_after_think = generated_text.split(self.answer_start_token)[-1]
        sections = re.split(
            r'\[VERIFIER FEEDBACK[^\]]*\]\s*', text_after_think, flags=re.DOTALL
        )
        text = sections[-1]

        step_pattern = re.compile(
            r'(>\s*Step\s*(\d+)\s*\n'
            r'available\s+numbers?\s*:\s*\[([^\]]+)\]\s*\n'
            r'suggested\s+operation\s*:\s*([^\n]+?)\s*\n'
            r'remaining\s+numbers?\s*:\s*\[([^\]]+)\])',
            re.IGNORECASE,
        )
        all_steps = list(step_pattern.finditer(text))
        if not all_steps:
            return None, None

        last_step = all_steps[-1]
        step_num = int(last_step.group(2))
        step_text = (
            f">Step{step_num}\n"
            f"available numbers: [{last_step.group(3).strip()}]\n"
            f"suggested operation: {last_step.group(4).strip()}\n"
            f"remaining numbers: [{last_step.group(5).strip()}]"
        )
        return step_num, parse_step(step_text)

    def _count_complete_steps(self, text: str) -> int:
        """Return how many complete step blocks are in the text."""
        step_pattern = re.compile(
            r'>\s*Step\s*\d+\s*\n'
            r'available\s+numbers?\s*:\s*\[([^\]]+)\]\s*\n'
            r'suggested\s+operation\s*:\s*([^\n]+?)\s*\n'
            r'remaining\s+numbers?\s*:\s*\[([^\]]+)\]',
            re.IGNORECASE,
        )
        return len(step_pattern.findall(text))

    # ------------------------------------------------------------------
    #  _stream_and_verify_steps
    # ------------------------------------------------------------------
    async def _stream_and_verify_steps(self, text_so_far: str):
        """
        Stream from the vLLM server with ``prompt + text_so_far`` (which
        already ends with the structured output prompt).

        As each complete step block appears, verify it immediately.
        - If a step is WRONG -> stop streaming, return the error info.
        - If all steps pass and the model finishes -> return full text.

        Returns:
            (full_text, is_all_valid, error_info_or_None)
        """
        payload = deepcopy(self.llm_server["payload"])
        payload["prompt"] = self.prompt + text_so_far
        payload["max_tokens"] = min(payload.get("max_tokens", 2048), 2048)

        generated = ""
        last_verified_step_count = 0

        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self.llm_server["url"],
                headers=self.llm_server["headers"],
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[len("data: "):].strip()
                        if data == "[DONE]":
                            break
                        chunk = json.loads(data)["choices"][0]["text"]
                        generated += chunk
                        logger.debug(f"[vLLM side-stream] chunk: {chunk!r}")

                        # Check if a new complete step appeared
                        current_step_count = self._count_complete_steps(generated)
                        if current_step_count > last_verified_step_count:
                            full_text = text_so_far + generated
                            step_num, parsed = self._extract_last_step_info(full_text)

                            if (step_num is not None
                                    and parsed is not None
                                    and parsed.get('available_numbers') is not None):
                                current_available = self._get_current_available(full_text)
                                is_valid, errors, new_available = verify_step(
                                    parsed, current_available,
                                    self.original_numbers, step_num,
                                )

                                if not is_valid:
                                    logger.info(
                                        f"[ThinkingPhaseVerifier] Side-stream: "
                                        f"Step {step_num} FAILED: {errors}"
                                    )
                                    return (
                                        full_text,
                                        False,
                                        {"step_num": step_num,
                                         "errors": errors,
                                         "available": current_available},
                                    )
                                else:
                                    logger.info(
                                        f"[ThinkingPhaseVerifier] Side-stream: "
                                        f"Step {step_num} verified OK"
                                    )

                            last_verified_step_count = current_step_count

        full_text = text_so_far + generated
        logger.info(
            f"[ThinkingPhaseVerifier] Side-stream finished. "
            f"Generated {len(generated)} chars, "
            f"{last_verified_step_count} steps verified."
        )
        return full_text, True, None

    # ------------------------------------------------------------------
    #  step_extractor
    # ------------------------------------------------------------------
    def step_extractor(self, chunk: str, generated_text: str):
        """
        Phase 1 (thinking): trigger when total newlines cross the next
            multiple of ``newline_threshold``.
        Phase 2 (after </think>): trigger when a natural ``</think>``
            is detected (to inject the structured prompt), or when a
            complete step block appears (for verification).
        """
        # ===== PHASE 1: still inside <think> =====
        if self._is_in_thinking_phase(generated_text):
            if self._think_phase_corrections >= self.max_corrections:
                return False, None

            total_newlines = generated_text.count('\n')

            if chunk.endswith('\n') and total_newlines > 0 and total_newlines % self.newline_threshold == 0:
                logger.info(
                    f"[ThinkingPhaseVerifier] Total newlines={total_newlines}, "
                    f"hit multiple of N={self.newline_threshold}. "
                    f"Forcing step generation."
                )
                return True, generated_text

            return False, None

        # ===== PHASE 2: after </think> =====

        # Sub-case 2a: </think> is present but structured prompt is not
        # -> trigger so verify() can inject it.
        if STRUCTURED_OUTPUT_PROMPT not in generated_text:
            logger.info(
                "[ThinkingPhaseVerifier] </think> present but structured "
                "output prompt missing - will inject it."
            )
            return True, generated_text

        # Sub-case 2b: structured prompt already injected -> detect
        # complete steps for verification (same as StepVerifierGame24Monitor).
        think_end_pos = generated_text.find(self.answer_start_token) + len(self.answer_start_token)
        text_after_think = generated_text[think_end_pos:]

        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()

        text = text_after_think[last_feedback_end:]
        text_start_in_generated = think_end_pos + last_feedback_end

        step_pattern = re.compile(
            r'(>\s*Step\s*(\d+)\s*\n'
            r'available\s+numbers?\s*:\s*\[([^\]]+)\]\s*\n'
            r'suggested\s+operation\s*:\s*([^\n]+?)\s*\n'
            r'remaining\s+numbers?\s*:\s*\[([^\]]+)\])',
            re.IGNORECASE,
        )
        all_steps = list(step_pattern.finditer(text))
        if not all_steps:
            return False, None

        last_complete_step = all_steps[-1]

        # Already moved past this step?
        text_after_last_step = text[last_complete_step.end():]
        if re.search(r'>\s*Step\s*\d+', text_after_last_step, re.IGNORECASE):
            return False, None

        end_pos = text_start_in_generated + last_complete_step.end()
        return True, generated_text[:end_pos]

    # ------------------------------------------------------------------
    #  verify
    # ------------------------------------------------------------------
    async def verify(self, step: str, token_index: int, event, event_info):
        """
        Case 1 - still in thinking (no </think> in step):
            Inject </think> + structured prompt, stream from vLLM to get
            steps, verify them, then either rollback (wrong) or commit
            (correct).

        Case 2a - natural </think> just appeared, structured prompt not
            yet injected:
            Signal fix() to append the structured output prompt.

        Case 2b - after </think> + structured prompt already injected:
            Identical to StepVerifierGame24Monitor - verify each step.
        """

        # ==================================================================
        # CASE 1: Thinking phase
        # ==================================================================
        if self.answer_start_token not in step:
            logger.info(
                "[ThinkingPhaseVerifier] Injecting </think> + structured "
                "prompt and streaming steps from vLLM inside verify()"
            )

            # Build text with injected </think> + structured prompt
            text_with_think_end = (
                step + "\n" + self.answer_start_token + "\n"
                + STRUCTURED_OUTPUT_PROMPT + "\n"
            )

            # Stream from vLLM, verifying each step as it appears
            full_text, is_all_valid, error_info = await self._stream_and_verify_steps(
                text_with_think_end
            )

            if is_all_valid:
                # All steps correct -> inject </think> + structured prompt
                # and let stream_completion recurse so the model generates
                # verified steps that Phase 2b checks.
                logger.info(
                    "[ThinkingPhaseVerifier] All side-streamed steps verified OK "
                    "- injecting </think> + structured prompt"
                )
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = self.answer_start_token
                    event_info["correction_index"] = token_index
                    event_info["phase"] = "inject_think_end"
                    event.set()
                return step, self.answer_start_token

            else:
                # Step is WRONG -> rollback into thinking
                errors = error_info["errors"]
                step_num = error_info["step_num"]
                logger.info(
                    f"[ThinkingPhaseVerifier] Step {step_num} FAILED: {errors}"
                )
                error_summary = "; ".join(errors)
                thinking_feedback = (
                    f"\n\nWait, that approach is wrong. {error_summary}. "
                    f"Let me reconsider and try a different approach.\n"
                )
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = thinking_feedback
                    event_info["correction_index"] = token_index
                    event_info["errors"] = errors
                    event_info["failed_step"] = step_num
                    event_info["phase"] = "rollback_to_thinking"
                    event.set()
                return step, thinking_feedback

        # ==================================================================
        # CASE 2a: </think> present but structured prompt missing
        # ==================================================================
        if STRUCTURED_OUTPUT_PROMPT not in step:
            logger.info(
                "[ThinkingPhaseVerifier] </think> present but structured "
                "prompt missing -> injecting it"
            )
            structured_prompt_text = "\n" + STRUCTURED_OUTPUT_PROMPT + "\n"
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = structured_prompt_text
                event_info["correction_index"] = token_index
                event_info["phase"] = "inject_structured_prompt"
                event.set()
            return step, structured_prompt_text

        # ==================================================================
        # CASE 2b: After </think> + structured prompt - standard verify
        # ==================================================================

        # ---- max-corrections guard ----
        num_corrections = (
            self._count_feedback_blocks(step)
            + self._think_phase_corrections
        )
        if num_corrections >= self.max_corrections:
            fb = "\nthe answer is \\boxed{no solution}"
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = fb
                event_info["correction_index"] = token_index
                event_info["errors"] = ["Max corrections reached"]
                event_info["failed_step"] = None
                event.set()
            return step, fb

        # ---- extract & verify step ----
        step_num, parsed = self._extract_last_step_info(step)
        if step_num is None or parsed is None or parsed.get('available_numbers') is None:
            return step, None

        current_available = self._get_current_available(step)
        is_valid, errors, new_available = verify_step(
            parsed, current_available, self.original_numbers, step_num
        )

        if is_valid:
            return step, None

        # ---- step has errors -> standard feedback ----
        logger.info(f"[ThinkingPhaseVerifier] Step {step_num} FAILED: {errors}")
        feedback = format_feedback(errors, step_num, current_available)
        if not event.is_set():
            event_info["generated_text"] = step
            event_info["feedback"] = feedback
            event_info["correction_index"] = token_index
            event_info["errors"] = errors
            event_info["failed_step"] = step_num
            event_info["phase"] = "standard_verify"
            event.set()
        return step, feedback

    # ------------------------------------------------------------------
    #  fix
    # ------------------------------------------------------------------
    async def fix(self, generated_text: str, event_info: dict, fix_method=None):
        """
        Applies the appropriate fix depending on the phase:

        inject_think_end
            Append ``</think>`` + structured output prompt so the model
            regenerates the steps naturally.

        rollback_to_thinking
            Strip everything from the inject point, append ``Wait ...``
            feedback inside the thinking trace.

        inject_structured_prompt
            Append the structured output prompt after a natural
            ``</think>`` (no rollback needed).

        standard_verify
            Append ``[VERIFIER FEEDBACK ...]`` (same as
            StepVerifierGame24Monitor).
        """
        phase = event_info.get("phase", "standard_verify")

        if phase == "inject_think_end":
            logger.info(
                "[ThinkingPhaseVerifier] fix(): injecting </think> + "
                "structured prompt"
            )
            return (
                event_info["generated_text"]
                + "\n" + self.answer_start_token + "\n"
                + STRUCTURED_OUTPUT_PROMPT + "\n"
            )

        if phase == "rollback_to_thinking":
            logger.info("[ThinkingPhaseVerifier] fix(): rolling back into thinking")

            base_text = event_info["generated_text"]
            result = base_text.rstrip() + event_info["feedback"]

            # Reset thinking-phase state for the next cycle
            self._think_phase_corrections += 1

            logger.info(
                f"[ThinkingPhaseVerifier] Rolled back. "
                f"Think-phase corrections: {self._think_phase_corrections}/{self.max_corrections}"
            )
            return result

        if phase == "inject_structured_prompt":
            logger.info(
                "[ThinkingPhaseVerifier] fix(): appending structured "
                "output prompt after natural </think>"
            )
            return event_info["generated_text"] + event_info["feedback"]

        # standard_verify
        logger.info("[ThinkingPhaseVerifier] fix(): standard step feedback")
        return event_info["generated_text"] + event_info["feedback"]
