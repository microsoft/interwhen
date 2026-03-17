"""
Thinking-phase verifier for Game of 24.

Verifies expressions by forking a side-stream during the thinking phase
to ask the model about its current progress.

Workflow
--------
A) **DURING the thinking phase** (inside ``<think>...</think>``):
   After a warmup period, every *N* newlines in the thinking trace:
   1. Inject ``</think> The expression that I found till now is {`` and
      stream ~20 tokens to extract the expression the model outputs.
   2. Verify the expression against Game-of-24 rules.
   3. If **wrong** -> inject error feedback into thinking trace.
   4. If **correct AND complete** -> inject early-stop message + ``</think>``.
   5. If **correct AND partial** -> no feedback, let model keep thinking.

B) **AFTER a natural ``</think>``**:
   Inject the expression extraction prompt so the model outputs its
   answer expression, then verify in the same way.
"""

import re
import json
import logging
from typing import List, Tuple, Optional
from copy import deepcopy

import httpx

from .base import VerifyMonitor
from ._common import find_complete_boxed
from ..utils.game24_verifier import (
    can_reach_24, is_close, format_number, safe_eval,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Prompts injected to elicit an expression from the model.
# ---------------------------------------------------------------------------

# Injected during the thinking phase (after </think>)
THINKING_PHASE_EXPRESSION_PROMPT = (
    "</think>\nThe expression that I found till now is {"
)

# Injected after a natural </think> to force the model to emit \boxed{expr}
FINAL_EXPRESSION_PROMPT = (
    "\nThe final expression is \\boxed"
)


# ---------------------------------------------------------------------------
#  Expression verification helpers
# ---------------------------------------------------------------------------

def _extract_numbers_from_expr(expr: str) -> List[float]:
    """Extract all numbers (integers and decimals) from an expression string."""
    numbers = re.findall(r'\d+\.?\d*', expr)
    return [int(float(n)) if float(n) == int(float(n)) else float(n) for n in numbers]


def _normalize_number(n) -> float:
    """Normalize a number for comparison."""
    return float(n)


def verify_expression(expr_str: str, original_numbers: List[float]) -> Tuple[str, bool, List[str], Optional[List[float]]]:
    """
    Verify an expression against the Game of 24 rules.

    Returns:
        (status, is_valid, errors, unused_numbers_or_None)
        - status: "complete" | "partial" | "error"
        - is_valid: True if the expression is valid (no errors)
        - errors: List of error messages
        - unused_numbers: Numbers from original not used in expr (None if errors)
    """
    errors = []
    fmt = format_number

    used_numbers = _extract_numbers_from_expr(expr_str)
    if not used_numbers:
        errors.append(f"No numbers found in expression: {expr_str}")
        return "error", False, errors, None

    original_copy = [_normalize_number(n) for n in original_numbers]
    matched_indices = []
    for used_n in used_numbers:
        used_norm = _normalize_number(used_n)
        found = False
        for i, orig_n in enumerate(original_copy):
            if i not in matched_indices and is_close(used_norm, orig_n):
                matched_indices.append(i)
                found = True
                break
        if not found:
            errors.append(
                f"Number {fmt(used_norm)} in expression is not available in "
                f"original numbers {[fmt(n) for n in original_numbers]} "
                f"(or was already used)"
            )

    if errors:
        return "error", False, errors, None

    unused = [original_copy[i] for i in range(len(original_copy)) if i not in matched_indices]

    try:
        value = eval(expr_str, {"__builtins__": None}, {})
        value = float(value)
    except Exception as e:
        errors.append(f"Cannot evaluate expression '{expr_str}': {e}")
        return "error", False, errors, None

    all_used = len(unused) == 0

    if all_used:
        if not is_close(value, 24):
            errors.append(
                f"Expression '{expr_str}' evaluates to {fmt(value)}, not 24."
            )
            return "error", False, errors, None
        return "complete", True, [], []
    else:
        remaining = [value] + unused
        can_reach, example = can_reach_24(remaining)
        if not can_reach:
            remaining_str = [fmt(n) for n in remaining]
            errors.append(
                f"Expression '{expr_str}' evaluates to {fmt(value)}. "
                f"Remaining numbers (including result) are {remaining_str}. "
                f"Cannot reach 24 from these numbers. This is a dead end."
            )
            return "error", False, errors, None
        return "partial", True, [], unused


# ---------------------------------------------------------------------------
#  Monitor
# ---------------------------------------------------------------------------

class ThinkingPhaseStepVerifierGame24Monitor(VerifyMonitor):
    """
    Monitor that verifies Game-of-24 expressions during and after thinking.

    During thinking: every N newlines (after warmup) -> fork a
        side-stream asking for the current expression, verify it, and
        give appropriate feedback.

    After natural ``</think>``: inject expression prompt, verify the
        final answer.
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
        warmup_newlines: int = 0,
    ):
        super().__init__(name)
        self.original_numbers = [float(x) for x in original_numbers]
        self.llm_server = llm_server
        self.prompt = prompt
        self.newline_threshold = newline_threshold
        self.max_corrections = max_corrections
        self.answer_start_token = answer_start_token
        self.async_execution = async_execution
        self.warmup_newlines = warmup_newlines

        # ---- state ----
        self._think_phase_corrections = 0
        self._verified_expression = None  # set by Phase 1 early-stop

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(n: float) -> str:
        if abs(n - round(n)) < 1e-9:
            return str(int(round(n)))
        return f"{n:.4f}".rstrip('0').rstrip('.')

    def _count_feedback_blocks(self, text: str) -> int:
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))

    def _is_in_thinking_phase(self, generated_text: str) -> bool:
        return self.answer_start_token not in generated_text

    @staticmethod
    def _extract_braced_expression(text: str) -> Optional[str]:
        """Extract the first expression wrapped in { } from *text*.

        Handles nested braces so that e.g. ``{(3+5)*7}`` is extracted correctly.
        """
        start = text.find('{')
        if start == -1:
            return None
        brace_count = 0
        end = start
        while end < len(text):
            if text[end] == '{':
                brace_count += 1
            elif text[end] == '}':
                brace_count -= 1
                if brace_count == 0:
                    break
            end += 1
        if brace_count != 0:
            return None
        expr = text[start + 1:end].strip()
        if not expr:
            return None
        # Basic cleanup: remove LaTeX
        expr = expr.replace(r'\times', '*').replace(r'\cdot', '*').replace(r'\div', '/')
        expr = expr.replace(r'\,', '').replace(r'\ ', '')
        expr = expr.replace(r'\left', '').replace(r'\right', '')
        # Replace Unicode math operators (QwQ frequently uses these)
        expr = expr.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u2212', '-')
        expr = expr.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash
        frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
        while re.search(frac_pattern, expr):
            expr = re.sub(frac_pattern, r"(\1/\2)", expr)
        # Handle implicit multiplication
        expr = re.sub(r'\)\s*\(', ')*(', expr)
        expr = re.sub(r'\)\s*(\d)', r')*\1', expr)
        expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
        return expr

    @staticmethod
    def _extract_boxed_expression(text: str) -> Optional[str]:
        """Extract expression from \\boxed{...} in text."""
        boxed_pattern = r"\\boxed\{"
        matches = list(re.finditer(boxed_pattern, text))
        if not matches:
            return None
        last_match = matches[-1]
        start = last_match.end()
        brace_count = 1
        end = start
        while end < len(text) and brace_count > 0:
            if text[end] == '{':
                brace_count += 1
            elif text[end] == '}':
                brace_count -= 1
            end += 1
        expr = text[start:end - 1].strip()
        expr = expr.replace(r'\times', '*').replace(r'\cdot', '*').replace(r'\div', '/')
        expr = expr.replace(r'\,', '').replace(r'\ ', '')
        expr = expr.replace(r'\left', '').replace(r'\right', '')
        expr = expr.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u2212', '-')
        expr = expr.replace('\u2013', '-').replace('\u2014', '-')
        frac_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
        while re.search(frac_pattern, expr):
            expr = re.sub(frac_pattern, r"(\1/\2)", expr)
        expr = re.sub(r'\)\s*\(', ')*(', expr)
        expr = re.sub(r'\)\s*(\d)', r')*\1', expr)
        expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
        return expr

    # ------------------------------------------------------------------
    #  _side_stream_expression  (streams ~20 tokens to get {expr})
    # ------------------------------------------------------------------
    async def _side_stream_expression(self, text_so_far: str, max_new_tokens: int = 20) -> Optional[str]:
        """
        Send ``prompt + text_so_far`` to vLLM, stream at most
        *max_new_tokens* tokens, and try to extract an expression from
        the output that appears inside ``{ }``.
        """
        fmt = self._fmt
        nums_str = ", ".join(fmt(n) for n in self.original_numbers)
        logger.info(
            f"[Side-stream] Starting expression extraction\n"
            f"  Original numbers : [{nums_str}]\n"
            f"  Max new tokens   : {max_new_tokens}"
        )

        payload = deepcopy(self.llm_server["payload"])
        payload["prompt"] = self.prompt + text_so_far
        payload["max_tokens"] = max_new_tokens
        payload.pop("logprobs", None)

        generated = ""

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
                        logger.debug(f"[Side-stream] chunk: {chunk!r}")

                        if '}' in generated:
                            break

        full_text = "{" + generated
        expr = self._extract_braced_expression(full_text)
        if expr:
            logger.info(f"[Side-stream] Extracted expression: {expr}")
        else:
            logger.info(
                f"[Side-stream] No expression found in side-stream "
                f"(generated {len(generated)} chars: {generated!r})"
            )
        return expr

    # ------------------------------------------------------------------
    #  step_extractor
    # ------------------------------------------------------------------
    def step_extractor(self, chunk: str, generated_text: str):
        # ===== PHASE 1: still inside <think> =====
        if self._is_in_thinking_phase(generated_text):
            if self._think_phase_corrections >= self.max_corrections:
                return False, None

            total_newlines = generated_text.count('\n')

            if total_newlines < self.warmup_newlines:
                return False, None

            past_warmup = total_newlines - self.warmup_newlines
            if (generated_text.endswith('\n')
                    and past_warmup >= 0
                    and past_warmup % self.newline_threshold == 0):
                logger.info(
                    f"[step_extractor] Phase 1 trigger: \\n count={total_newlines} "
                    f"(warmup={self.warmup_newlines}, past_warmup={past_warmup}, "
                    f"threshold={self.newline_threshold})"
                )
                return True, generated_text

            return False, None

        # ===== PHASE 2: after </think> =====

        # 2a: </think> present but we haven't injected the expression prompt yet
        if FINAL_EXPRESSION_PROMPT.strip() not in generated_text:
            logger.info(
                "[step_extractor] Phase 2a: </think> detected, "
                "expression prompt not yet injected."
            )
            return True, generated_text

        # 2b: trigger once we see a complete \boxed{...}
        think_end_pos = generated_text.find(self.answer_start_token) + len(self.answer_start_token)
        text_after_think = generated_text[think_end_pos:]

        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        text = text_after_think[last_feedback_end:]

        has_boxed = find_complete_boxed(text)
        if has_boxed:
            return True, generated_text

        return False, None

    # ------------------------------------------------------------------
    #  verify
    # ------------------------------------------------------------------
    async def verify(self, step: str, token_index: int, event, event_info):
        # ==================================================================
        # CASE 1: Thinking phase -- side-stream expression verification
        # ==================================================================
        if self.answer_start_token not in step:
            total_dn = step.count('\n')
            logger.info(
                f"[Phase 1] Thinking-phase verification triggered\n"
                f"  \\n count    : {total_dn}\n"
                f"  Thinking len : {len(step)} chars"
            )

            text_with_prompt = step + "\n" + THINKING_PHASE_EXPRESSION_PROMPT

            expr_str = await self._side_stream_expression(text_with_prompt, max_new_tokens=20)

            if expr_str is None:
                logger.info(
                    "[Phase 1] No expression extracted from side-stream. "
                    "Letting model continue thinking."
                )
                return step, None

            status, is_valid, errors, unused = verify_expression(
                expr_str, self.original_numbers
            )

            if not is_valid:
                error_summary = "; ".join(errors)
                self._think_phase_corrections += 1
                logger.info(
                    f"[Phase 1] INVALID expression '{expr_str}'\n"
                    f"  Error(s) : {error_summary}\n"
                    f"  Action   : Inject feedback into thinking trace\n"
                    f"  Corrections: {self._think_phase_corrections}/{self.max_corrections}"
                )
                thinking_feedback = (
                    f"\n\nWait, the expression {expr_str} does not work. "
                    f"{error_summary} "
                    f"I must NOT reuse {expr_str} or any expression I have already tried. "
                    f"Let me try a completely different combination of "
                    f"operations and grouping of numbers.\n"
                )
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = thinking_feedback
                    event_info["correction_index"] = token_index
                    event_info["errors"] = errors
                    event_info["phase"] = "rollback_to_thinking"
                    event.set()
                return step, thinking_feedback

            elif status == "complete":
                self._verified_expression = expr_str
                logger.info(
                    f"[Phase 1] VALID COMPLETE expression '{expr_str}' == 24\n"
                    f"  Action: Inject early-stop message and transition to answer."
                )
                early_stop_msg = (
                    f"\n\nWait, the expression {expr_str} has been verified "
                    f"to equal 24 using all the given numbers. This will be "
                    f"my final answer.\n{self.answer_start_token}\n"
                )
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = early_stop_msg
                    event_info["correction_index"] = token_index
                    event_info["phase"] = "early_stop_answer"
                    event_info["verified_expression"] = expr_str
                    event.set()
                return step, early_stop_msg

            else:
                unused_str = (
                    "[" + ", ".join(self._fmt(n) for n in unused) + "]"
                    if unused else "[]"
                )
                logger.info(
                    f"[Phase 1] VALID PARTIAL expression '{expr_str}'\n"
                    f"  Unused numbers: {unused_str}\n"
                    f"  Action: No error, let model keep thinking."
                )
                return step, None

        # ==================================================================
        # CASE 2a: </think> present but expression prompt not yet injected
        # ==================================================================
        if FINAL_EXPRESSION_PROMPT.strip() not in step:
            logger.info(
                "[Phase 2a] Natural </think> detected. "
                "Injecting expression extraction prompt."
            )
            prompt_text = FINAL_EXPRESSION_PROMPT
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = prompt_text
                event_info["correction_index"] = token_index
                event_info["phase"] = "inject_expression_prompt"
                event.set()
            return step, prompt_text

        # ==================================================================
        # CASE 2b: After </think> + expression prompt -- verify final answer
        # ==================================================================

        num_corrections = self._count_feedback_blocks(step)
        if num_corrections >= self.max_corrections:
            fb = "\nthe answer is \\boxed{no solution}"
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = fb
                event_info["correction_index"] = token_index
                event_info["errors"] = ["Max corrections reached"]
                event_info["phase"] = "standard_verify"
                event.set()
            return step, fb

        think_end_pos = step.find(self.answer_start_token) + len(self.answer_start_token)
        text_after_think = step[think_end_pos:]
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        recent_text = text_after_think[last_feedback_end:]

        expr_str = self._extract_boxed_expression(recent_text)
        if expr_str is not None:
            logger.info(f"[Phase 2b] Extracted expression from \\boxed: '{expr_str}'")

        if expr_str is None:
            return step, None

        status, is_valid, errors, unused = verify_expression(
            expr_str, self.original_numbers
        )

        if is_valid and status == "complete":
            logger.info(f"[Phase 2b] Final expression '{expr_str}' is correct (= 24)")
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = ""
                event_info["correction_index"] = token_index
                event_info["phase"] = "final_answer_correct"
                event_info["verified_expression"] = expr_str
                event.set()
            return step, None

        if is_valid and status == "partial":
            used_numbers = _extract_numbers_from_expr(expr_str)
            errors = [
                f"Expression '{expr_str}' only uses {len(used_numbers)} of "
                f"{len(self.original_numbers)} numbers. After </think>, "
                f"a COMPLETE expression using ALL numbers is required."
            ]

        if not errors:
            errors = [f"Expression '{expr_str}' is not a valid solution."]

        error_summary = "; ".join(errors)
        logger.info(f"[Phase 2b] Final expression FAILED: {error_summary}")

        orig_display = [int(n) if n == int(n) else n for n in self.original_numbers]
        nums_str = ", ".join(str(n) for n in orig_display)
        feedback = (
            f"\n[VERIFIER FEEDBACK:\n"
            f"  The expression {expr_str} is incorrect. {error_summary}\n"
            f"  Do NOT reuse {expr_str} or any previously tried expression.\n"
            f"  Try a completely different approach. Use ALL four numbers "
            f"{nums_str} exactly once, "
            f"evaluating to 24. Wrap in \\boxed{{}}. ]\n"
        )
        if not event.is_set():
            event_info["generated_text"] = step
            event_info["feedback"] = feedback
            event_info["correction_index"] = token_index
            event_info["errors"] = errors
            event_info["phase"] = "standard_verify"
            event.set()
        return step, feedback

    # ------------------------------------------------------------------
    #  fix
    # ------------------------------------------------------------------
    async def fix(self, generated_text: str, event_info: dict, fix_method=None):
        phase = event_info.get("phase", "standard_verify")

        if phase == "rollback_to_thinking":
            base_text = event_info["generated_text"]
            result = base_text.rstrip() + event_info["feedback"]
            logger.info(
                f"[fix] Phase: rollback_to_thinking\n"
                f"  -> Appended error feedback into <think> trace.\n"
                f"  -> Think-phase corrections: {self._think_phase_corrections}/{self.max_corrections}"
            )
            return result

        if phase == "early_stop_answer":
            base_text = event_info["generated_text"]
            result = base_text.rstrip() + event_info["feedback"]
            logger.info(
                f"[fix] Phase: early_stop_answer\n"
                f"  -> Verified expression passed. Injecting early-stop + </think>.\n"
                f"  -> Model will now generate the final answer."
            )
            return result

        if phase == "final_answer_correct":
            expr = event_info.get("verified_expression", "?")
            logger.info(
                f"[fix] Phase: final_answer_correct\n"
                f"  -> Final expression '{expr}' verified correct. Stopping generation."
            )
            return event_info["generated_text"]

        if phase == "inject_expression_prompt":
            logger.info(
                f"[fix] Phase: inject_expression_prompt\n"
                f"  -> Natural </think> detected.\n"
                f"  -> Appending expression extraction prompt."
            )
            return event_info["generated_text"] + event_info["feedback"]

        # standard_verify
        errors = event_info.get("errors", [])
        error_summary = "; ".join(errors) if errors else "unknown"
        logger.info(
            f"[fix] Phase: standard_verify\n"
            f"  -> Expression failed: {error_summary}\n"
            f"  -> Appending [VERIFIER FEEDBACK] so model retries."
        )
        return event_info["generated_text"] + event_info["feedback"]
