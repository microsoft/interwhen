"""
Thinking Phase Verifiers for Game of 24 and Maze.

These monitors verify solutions by forking a side-stream during the thinking
phase to ask the model about its current progress.

Game of 24 Workflow
-------------------
A) **DURING the thinking phase** (inside ``<think>...</think>``):
   After a warmup period, every *N* double-newlines in the thinking trace:
   1. Inject ``</think> The expression that I found till now is {`` and
      stream ~20 tokens to extract the expression the model outputs.
   2. Verify the expression:
      a. Extract numbers used in the expression.
      b. Check each number appears in the original numbers (at most once).
      c. If ALL four numbers are used: evaluate and check == 24.
      d. If partial: evaluate the sub-expression, collect unused original
         numbers, check ``can_reach_24([result] + unused)``.
   3. If **wrong** -> strip the injected text, append
      ``Wait, <error description>.`` inside the thinking trace and let
      the model keep thinking.
   4. If **correct AND complete** (all 4 numbers, equals 24) -> inject
      ``Wait, current expression that I am able to generate seems to be
      passed by the verifier, so let me stop and give the answer.
      </think>`` and then let the model output the final answer.
   5. If **correct AND partial** -> no feedback, let the model keep
      thinking undisturbed.

B) **AFTER a natural ``</think>``**:
   Inject the same expression extraction prompt so the model outputs its
   answer expression, then verify in the same way.  Give feedback on
   errors so the model retries.

Maze Workflow
-------------
A) **DURING the thinking phase** (inside ``<think>...</think>``):
   After a warmup period, every *N* double-newlines in the thinking trace:
   1. Inject a first-person prompt in the LLM's own voice:
      ``Let me output the current steps I have traced so far through
      the maze in the following format:`` + ``<format>...</format>``
      + ``>>> LOCATE START AND EXIT:``.  Stream ~300 tokens to
      extract the model's current traced path steps.
   2. Parse the structured steps and verify each against the maze grid:
      a. Is the move direction correct (delta matches)?
      b. Is from_pos the expected position?
      c. Is to_pos walkable (not a wall)?
      d. Is the turn type correct?
      e. Are running counts correct?
   3. If **errors found** -> strip the injected text, append
      ``Wait, <error description>.`` and let the model keep thinking.
   4. If **path reaches E with all steps correct** -> inject early-stop
      message + ``</think>`` followed by the structured format prompt
      so the model gives the final answer in the specified format.
   5. If **partial but correct so far** -> no feedback, keep thinking.

B) **AFTER ``</think>`` (natural or early-stop)**:
   Phase 2a: Inject the same structured step format template (in the
   LLM's own voice: ``Let me trace the step by step solution...`` +
   ``<format>...</format>`` + ``>>> LOCATE START AND EXIT:``) so the
   model fills it in.

   Phase 2b: Verify each step as the model fills in the template.
   Once ``\\boxed{}`` appears, stop generation.
"""

import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from copy import deepcopy

import httpx

from .base import VerifyMonitor
from ..utils.game24_verifier import (
    can_reach_24, is_close, format_number, safe_eval,
)
from ..utils.maze_verifier import (
    Direction, parse_direction, get_expected_turn_type,
    parse_maze_from_prompt, parse_maze_step, verify_maze_step,
    verify_locate_section, format_maze_feedback, format_locate_feedback,
    DIRECTION_DELTAS, compute_relative_direction,
)
from ..utils.spatialmap_verifier import (
    SpatialMapZ3Solver, extract_step2_claims,
    parse_directional_claims_from_text,
    parse_counting_question, parse_model_count_from_answer,
    parse_direction_question, parse_object_question,
    parse_model_boxed_answer,
    get_possible_directions, get_consistent_object_options,
    get_possible_count_range,
    verify_spatialmap_step, format_spatialmap_feedback,
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

    Args:
        expr_str: The arithmetic expression string (e.g. "1*2", "(3+5)*7/11")
        original_numbers: The four original numbers.

    Returns:
        (status, is_valid, errors, unused_numbers_or_None)
        - status: "complete" | "partial" | "error"
        - is_valid: True if the expression is valid (no errors)
        - errors: List of error messages
        - unused_numbers: Numbers from original not used in expr (None if errors)
    """
    errors = []
    fmt = format_number

    # 1. Extract numbers used in the expression
    used_numbers = _extract_numbers_from_expr(expr_str)
    if not used_numbers:
        errors.append(f"No numbers found in expression: {expr_str}")
        return "error", False, errors, None

    # 2. Check each used number appears in original (at most once)
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

    # 3. Compute unused original numbers
    unused = [original_copy[i] for i in range(len(original_copy)) if i not in matched_indices]

    # 4. Evaluate the expression
    try:
        value = eval(expr_str, {"__builtins__": None}, {})
        value = float(value)
    except Exception as e:
        errors.append(f"Cannot evaluate expression '{expr_str}': {e}")
        return "error", False, errors, None

    # 5. Check based on whether all numbers are used
    all_used = len(unused) == 0

    if all_used:
        # Full expression: must equal 24
        if not is_close(value, 24):
            errors.append(
                f"Expression '{expr_str}' evaluates to {fmt(value)}, not 24."
            )
            return "error", False, errors, None
        # Valid complete solution!
        return "complete", True, [], []
    else:
        # Partial expression: check if remaining numbers + result can reach 24
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
        # Partial but reachable -- valid
        return "partial", True, [], unused


class ThinkingPhaseStepVerifierGame24Monitor(VerifyMonitor):
    """
    Monitor that verifies Game-of-24 expressions during and after thinking.

    During thinking: every N double-newlines (after warmup) -> fork a
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
        # Replace Unicode math operators (QwQ frequently uses these)
        expr = expr.replace('\u00d7', '*').replace('\u00f7', '/').replace('\u2212', '-')
        expr = expr.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash
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

        ``text_so_far`` is expected to end with something like
        ``</think>\\nThe expression that I found till now is {``
        so the model just needs to output the expression body and ``}``.

        Returns the extracted expression string, or None.
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
        # We don't need logprobs for the side-stream
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

                        # As soon as we see '}', we have the expression
                        if '}' in generated:
                            break

        # The model was prompted with "{ " so its output completes the brace.
        # We wrap it back so _extract_braced_expression can parse it.
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
    #  step_extractor -- decides WHEN to trigger verification
    # ------------------------------------------------------------------
    def step_extractor(self, chunk: str, generated_text: str):
        """
        Phase 1 (thinking): trigger when total double-newlines cross the
            next multiple of ``newline_threshold`` (after warmup).
        Phase 2 (after </think>): trigger to inject the expression prompt,
            or when a ``{expression}`` or ``\\boxed{expression}`` appears.
        """
        # ===== PHASE 1: still inside <think> =====
        if self._is_in_thinking_phase(generated_text):
            if self._think_phase_corrections >= self.max_corrections:
                return False, None

            total_double_newlines = generated_text.count('\n\n')

            # Skip until warmup period is reached
            if total_double_newlines < self.warmup_newlines:
                return False, None

            # After warmup, trigger at every newline_threshold multiple
            past_warmup = total_double_newlines - self.warmup_newlines
            if (generated_text.endswith('\n\n')
                    and past_warmup >= 0
                    and past_warmup % self.newline_threshold == 0):
                logger.info(
                    f"[step_extractor] Phase 1 trigger: \\n\\n count={total_double_newlines} "
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

        # 2b: expression prompt was injected (ends with "\boxed").
        #     The model should complete it with "{expression}".
        #     Trigger once we see a complete \boxed{...} (with closing brace).
        think_end_pos = generated_text.find(self.answer_start_token) + len(self.answer_start_token)
        text_after_think = generated_text[think_end_pos:]

        # Look past any previous feedback blocks
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        text = text_after_think[last_feedback_end:]

        has_boxed = re.search(r'\\boxed\{[^}]+\}', text)
        if has_boxed:
            return True, generated_text

        return False, None

    # ------------------------------------------------------------------
    #  verify
    # ------------------------------------------------------------------
    async def verify(self, step: str, token_index: int, event, event_info):
        """
        Case 1 -- still in thinking (no </think> in step):
            Inject ``</think> The expression that I found till now is {``,
            stream ~20 tokens, extract the expression, verify it.
            - Error -> feedback ``Wait, <error>.``
            - Correct & complete -> inject early-stop message
            - Correct & partial -> do nothing, let model keep thinking

        Case 2a -- natural </think> just appeared, expression prompt not
            yet injected:
            Signal fix() to append the expression prompt.

        Case 2b -- after </think> + expression prompt already injected:
            Verify the expression from the model's output.
        """

        # ==================================================================
        # CASE 1: Thinking phase -- side-stream expression verification
        # ==================================================================
        if self.answer_start_token not in step:
            total_dn = step.count('\n\n')
            logger.info(
                f"[Phase 1] Thinking-phase verification triggered\n"
                f"  \\n\\n count  : {total_dn}\n"
                f"  Thinking len : {len(step)} chars"
            )

            # Build text with injected prompt for expression extraction
            text_with_prompt = step + "\n" + THINKING_PHASE_EXPRESSION_PROMPT

            # Side-stream: get expression from the model (~20 tokens)
            expr_str = await self._side_stream_expression(text_with_prompt, max_new_tokens=20)

            if expr_str is None:
                # Model didn't produce a parseable expression -- let it keep thinking
                logger.info(
                    "[Phase 1] No expression extracted from side-stream. "
                    "Letting model continue thinking."
                )
                return step, None

            # Verify the extracted expression
            status, is_valid, errors, unused = verify_expression(
                expr_str, self.original_numbers
            )

            if not is_valid:
                # ---- WRONG: inject error feedback into thinking trace ----
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
                # ---- CORRECT & COMPLETE: early-stop, push to answer ----
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
                # ---- CORRECT & PARTIAL: let model keep thinking ----
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

        # Max-corrections guard
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

        # Extract expression from \boxed{...} — only look at text after
        # the last feedback block to avoid re-extracting old expressions.
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

        # Verify the final expression (must use all 4 numbers and equal 24)
        status, is_valid, errors, unused = verify_expression(
            expr_str, self.original_numbers
        )

        if is_valid and status == "complete":
            logger.info(f"[Phase 2b] Final expression '{expr_str}' is correct (= 24)")
            # Signal STOP so the model doesn't keep generating
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = ""  # nothing to append
                event_info["correction_index"] = token_index
                event_info["phase"] = "final_answer_correct"
                event_info["verified_expression"] = expr_str
                event.set()
            return step, None

        # Build error messages for partial/wrong answers in phase 2
        if is_valid and status == "partial":
            # In phase 2 (after </think>) we need ALL numbers used
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
        """
        Applies the appropriate fix depending on the phase:

        - ``rollback_to_thinking``: Append error feedback into thinking trace.
        - ``early_stop_answer``: Append early-stop message + </think> to
          transition the model to answer generation.
        - ``inject_expression_prompt``: Append expression prompt after </think>.
        - ``standard_verify``: Append [VERIFIER FEEDBACK ...].
        """
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


# =====================================================================
#  Maze Thinking-Phase Prompts
# =====================================================================


def _build_maze_format_block(question_type: str) -> str:
    """
    Build the <format>...</format> block that describes the structured
    output template.  Re-used by both the side-stream (Phase 1) and
    the post-</think> injection (Phase 2a).
    """
    if question_type == "relative_position":
        return (
            "<format>\n"
            ">>> LOCATE START AND EXIT (0-indexed, top-left is (0,0)):\n"
            "    S position: (row, col)\n"
            "    E position: (row, col)\n"
            "\n"
            ">>> COMPARE POSITIONS:\n"
            "    Row comparison: E row (r) vs S row (r) → E is ABOVE/BELOW S\n"
            "    Col comparison: E col (c) vs S col (c) → E is LEFT/RIGHT of S\n"
            "\n"
            ">>> FINAL ANSWER:\n"
            "    \\boxed{LETTER}\n"
            "</format>"
        )
    else:
        count_line = "    Running count: Right=0, Left=0"
        if question_type == "total_turns":
            count_line = "    Running count: Right=0, Left=0, Total=0"

        return (
            "<format>\n"
            ">>> LOCATE START AND EXIT (0-indexed, top-left is (0,0)):\n"
            "    S position: (row, col)\n"
            "    E position: (row, col)\n"
            "\n"
            ">>> STEP 1: Move DOWN from (r1, c1) to (r2, c2)\n"
            "    Current position: (r2, c2)\n"
            "    Previous direction: —\n"
            "    Current direction: DOWN\n"
            "    Turn type: STRAIGHT\n"
            f"{count_line}\n"
            "\n"
            "[... continue for all steps until reaching E ...]\n"
            "\n"
            ">>> FINAL ANSWER:\n"
            "    \\boxed{LETTER}\n"
            "</format>"
        )


def _build_maze_thinking_phase_prompt(question_type: str) -> str:
    """
    Build the side-stream prompt injected during the thinking phase.

    Written in the LLM's own first-person thinking voice so it blends
    naturally with the ``<think>`` trace.  Includes the ``<format>``
    block and the starting marker so the model begins filling in.
    """
    format_block = _build_maze_format_block(question_type)
    return (
        "\n\nLet me output the current steps I have traced so far "
        "through the maze in the following format:\n"
        f"{format_block}\n"
        ">>> LOCATE START AND EXIT (0-indexed, top-left is (0,0)):\n"
    )


def _build_maze_structured_prompt(question_type: str) -> str:
    """
    Build the structured format prompt injected after </think>.

    This is analogous to Game24's step format injection — it gives the
    model a template to fill in so we can parse and verify each step.
    Written in the LLM's own voice so it reads naturally.
    """
    format_block = _build_maze_format_block(question_type)
    return (
        "\nLet me trace the step by step solution through the maze "
        "in the following format:\n"
        f"{format_block}\n"
        ">>> LOCATE START AND EXIT (0-indexed, top-left is (0,0)):\n"
    )


# =====================================================================
#  ThinkingPhaseStepVerifierMazeMonitor
# =====================================================================

class ThinkingPhaseStepVerifierMazeMonitor(VerifyMonitor):
    """
    Monitor that verifies maze path-tracing during and after thinking.

    **No meta-prompt required** — works with a plain user prompt containing
    just the maze and question.  Structure is injected by this monitor
    after ``</think>`` (natural or early-stop), exactly like Game24
    injects its step format.

    Phase 1 – During ``<think>...</think>``:
        Every N double-newlines (after warmup), fork a side-stream that
        injects ``</think>`` + a structured step prompt, stream ~300
        tokens, parse and verify each step against the maze grid.

    Phase 2a – ``</think>`` detected, structured prompt not yet injected:
        Inject the structured step-by-step format template so the model
        fills it in (LOCATE → STEPs → FINAL ANSWER → ``\\boxed{}``).

    Phase 2b – Structured prompt injected, model is generating:
        Verify each completed step as it appears.  Once ``\\boxed{}``
        appears, signal completion.
    """

    def __init__(
        self,
        name: str,
        grid: list,
        start_pos: tuple,
        exit_pos: tuple,
        llm_server: dict,
        prompt: str,
        question_type: str = "right_turns",
        newline_threshold: int = 10,
        max_corrections: int = 5,
        answer_start_token: str = "</think>",
        async_execution: bool = True,
        warmup_newlines: int = 0,
    ):
        super().__init__(name)
        self.grid = grid
        self.start_pos = start_pos
        self.exit_pos = exit_pos
        self.llm_server = llm_server
        self.prompt = prompt
        self.question_type = question_type
        self.newline_threshold = newline_threshold
        self.max_corrections = max_corrections
        self.answer_start_token = answer_start_token
        self.async_execution = async_execution
        self.warmup_newlines = warmup_newlines

        # Build the structured prompt that will be injected after </think>
        self._structured_prompt = _build_maze_structured_prompt(question_type)
        # Build the thinking-phase side-stream prompt (in LLM's own voice)
        self._thinking_phase_prompt = _build_maze_thinking_phase_prompt(question_type)
        # A unique marker to detect whether we already injected it
        self._structured_marker = ">>> LOCATE START AND EXIT (0-indexed, top-left is (0,0)):"

        # ---- state ----
        self._think_phase_corrections = 0
        self._verified_path_complete = False  # True if path reaches E

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------
    def _count_feedback_blocks(self, text: str) -> int:
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))

    def _is_in_thinking_phase(self, generated_text: str) -> bool:
        return self.answer_start_token not in generated_text

    def _structured_prompt_injected(self, generated_text: str) -> bool:
        """Check if structured format was already injected after </think>."""
        if self.answer_start_token not in generated_text:
            return False
        after_think = generated_text.split(self.answer_start_token, 1)[1]
        return self._structured_marker in after_think

    @staticmethod
    def detect_question_type(prompt: str) -> str:
        """Auto-detect question type from prompt text."""
        prompt_lower = prompt.lower()
        if "right turn" in prompt_lower or "right-turn" in prompt_lower:
            return "right_turns"
        if "left turn" in prompt_lower or "left-turn" in prompt_lower:
            return "total_turns"
        if "total" in prompt_lower and "turn" in prompt_lower:
            return "total_turns"
        if "turn" in prompt_lower:
            return "right_turns"
        return "relative_position"

    def _verify_relative_position_answer(self, boxed_answer: str) -> Tuple[bool, Optional[str]]:
        """Verify a relative-position boxed answer (A=Yes / B=No).

        Parses the question from ``self.prompt`` to determine the asked
        direction, computes the true relative direction of E from S,
        and checks whether the model's Yes/No answer is correct.

        Returns ``(is_correct, feedback_or_None)``.
        """
        # Map boxed letter → Yes / No
        answer_map = {"A": "Yes", "B": "No"}
        model_yn = answer_map.get(boxed_answer.strip().upper())
        if model_yn is None:
            # Not A or B – can't verify
            return True, None

        # --- Parse the asked direction from the prompt ---
        # Patterns: "directly to the left of the starting point (S)"
        #           "directly below the starting point (S)"
        #           "to the top right of the starting point (S)"
        m = re.search(
            r'Is the exit \(E\)\s+(.*?)\s+(?:of\s+)?the starting point \(S\)',
            self.prompt, re.IGNORECASE,
        )
        if not m:
            return True, None  # can't parse question, skip verification

        asked_raw = m.group(1).strip().lower()
        # Remove trailing comma and extra clauses like ", with no ..."
        asked_raw = re.sub(r',.*', '', asked_raw).strip()

        # --- Compute actual relative direction ---
        actual = compute_relative_direction(self.start_pos, self.exit_pos)

        # --- Determine expected Yes / No ---
        # "directly to the left … with no vertical displacement"
        #  → same row, E col < S col  → actual in {"west"}
        # "directly below … with no horizontal displacement"
        #  → same col, E row > S row  → actual in {"south"}
        # "to the top right" → E north-east of S → actual == "northeast"
        direction_keywords = {
            "directly to the left":   {"west"},
            "directly to the right":  {"east"},
            "directly above":         {"north"},
            "directly below":         {"south"},
            "to the top left":        {"northwest"},
            "to the top right":       {"northeast"},
            "to the bottom left":     {"southwest"},
            "to the bottom right":    {"southeast"},
        }

        expected_dirs = direction_keywords.get(asked_raw)
        if expected_dirs is None:
            return True, None  # unrecognised pattern, skip

        expected_yn = "Yes" if actual in expected_dirs else "No"

        if model_yn == expected_yn:
            return True, None

        # --- Build feedback ---
        sr, sc = self.start_pos
        er, ec = self.exit_pos
        correct_letter = 'A' if expected_yn == 'Yes' else 'B'
        feedback = (
            f"\n\n[VERIFIER FEEDBACK for relative position:\n"
            f"  ✗ Your answer {boxed_answer} ({model_yn}) is incorrect.\n"
            f"  IMPORTANT: In this task, \"{asked_raw}\" means the GENERAL "
            f"COMPASS DIRECTION, NOT immediate adjacency. It asks whether E "
            f"is in the {actual} direction from S, regardless of distance or "
            f"walls between them.\n"
            f"  S is at row={sr}, col={sc}. E is at row={er}, col={ec}.\n"
            f"  Row difference (E-S): {er - sr} ({'same row' if er == sr else ('E is below S' if er > sr else 'E is above S')}).\n"
            f"  Col difference (E-S): {ec - sc} ({'same col' if ec == sc else ('E is right of S' if ec > sc else 'E is left of S')}).\n"
            f"  Therefore E is {actual} of S → the correct answer to "
            f"\"{asked_raw}\" is {expected_yn}.\n"
            f"  Do NOT consider adjacency or walls. Just compare the row/col "
            f"coordinates of S and E.\n"
            f"  Output \\boxed{{{correct_letter}}} for {expected_yn}. "
            f"This is the verified correct answer — do not argue.]\n\n"
        )
        return False, feedback

    # ------------------------------------------------------------------
    #  _parse_steps_from_text – parse structured steps from side-stream
    # ------------------------------------------------------------------
    def _parse_steps_from_text(self, text: str):
        """
        Parse all structured maze steps from text.

        Returns list of parsed step dicts.
        """
        steps = []

        step_pattern = re.compile(
            r'>>>\s*STEP\s+(\d+):\s*Move\s+\w+\s+from\s+\([^)]+\)\s+to\s+\([^)]+\).*?'
            r'Running count:\s*Right\s*=\s*\d+\s*,\s*Left\s*=\s*\d+[^\n]*',
            re.IGNORECASE | re.DOTALL
        )

        for match in step_pattern.finditer(text):
            parsed = parse_maze_step(match.group(0))
            if parsed:
                steps.append(parsed)

        return steps

    def _verify_all_steps(self, steps):
        """
        Verify a sequence of parsed maze steps against the grid.

        Returns:
            (all_valid, first_error_step_num, errors, final_pos, final_dir,
             right_count, left_count, total_count)
        """
        pos = self.start_pos
        direction = Direction.NONE
        right_count = 0
        left_count = 0
        total_count = 0

        for step in steps:
            is_valid, errors, state = verify_maze_step(
                step=step,
                grid=self.grid,
                expected_from_pos=pos,
                prev_direction=direction,
                expected_right_count=right_count,
                expected_left_count=left_count,
                expected_total_count=total_count,
            )

            if not is_valid:
                return (False, step.get('step_num', 0), errors,
                        pos, direction, right_count, left_count, total_count)

            pos = state['new_pos']
            direction = state['new_direction']
            right_count = state['new_right']
            left_count = state['new_left']
            total_count = state['new_total']

        return (True, None, [], pos, direction,
                right_count, left_count, total_count)

    # ------------------------------------------------------------------
    #  _side_stream_maze_steps – streams tokens to get traced path
    # ------------------------------------------------------------------
    async def _side_stream_maze_steps(self, text_so_far: str, max_new_tokens: int = 300) -> str:
        """
        Send ``prompt + text_so_far`` to vLLM, stream at most
        *max_new_tokens* tokens, and return the generated text.

        ``text_so_far`` is expected to end with the structured maze step
        prompt so the model outputs its traced steps.
        """
        logger.info(
            f"[Maze Side-stream] Starting path extraction\n"
            f"  Maze: S={self.start_pos}, E={self.exit_pos}\n"
            f"  Max new tokens: {max_new_tokens}"
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
                        logger.debug(f"[Maze Side-stream] chunk: {chunk!r}")

                        # Stop if we see FINAL ANSWER or \boxed
                        if '\\boxed' in generated or '>>> FINAL ANSWER' in generated:
                            break

        logger.info(
            f"[Maze Side-stream] Generated {len(generated)} chars"
        )
        return generated

    # ------------------------------------------------------------------
    #  _extract_boxed_answer
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_boxed_answer(text: str) -> Optional[str]:
        """Extract the content of the last \\boxed{...} in text."""
        matches = list(re.finditer(r'\\boxed\{', text))
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
        return text[start:end - 1].strip()

    # ------------------------------------------------------------------
    #  step_extractor
    # ------------------------------------------------------------------
    def step_extractor(self, chunk: str, generated_text: str):
        """
        Phase 1 (thinking): trigger at every newline_threshold multiple
            (after warmup).
        Phase 2 (after </think>): trigger on structured steps or boxed
            answer.
        """
        # ===== PHASE 1: still inside <think> =====
        if self._is_in_thinking_phase(generated_text):
            if self._think_phase_corrections >= self.max_corrections:
                return False, None

            total_double_newlines = generated_text.count('\n\n')

            if total_double_newlines < self.warmup_newlines:
                return False, None

            past_warmup = total_double_newlines - self.warmup_newlines
            if (generated_text.endswith('\n\n')
                    and past_warmup >= 0
                    and past_warmup % self.newline_threshold == 0):
                logger.info(
                    f"[Maze step_extractor] Phase 1 trigger: \\n\\n count={total_double_newlines} "
                    f"(warmup={self.warmup_newlines}, past_warmup={past_warmup}, "
                    f"threshold={self.newline_threshold})"
                )
                return True, generated_text

            return False, None

        # ===== PHASE 2: after </think> =====

        # 2a: structured prompt not yet injected → trigger immediately
        if not self._structured_prompt_injected(generated_text):
            logger.info(
                "[Maze step_extractor] Phase 2a: </think> detected, "
                "structured prompt not yet injected."
            )
            return True, generated_text

        # 2b: structured prompt injected — verify steps / boxed answer
        think_end_pos = generated_text.find(self.answer_start_token) + len(self.answer_start_token)
        text_after_think = generated_text[think_end_pos:]

        # Strip out the injected <format>...</format> template so we only
        # look at actual model output (which starts after the last
        # ">>> LOCATE START AND EXIT (0-indexed, ...):\n" line that ends the injected prompt).
        last_marker_pos = text_after_think.rfind(self._structured_marker)
        if last_marker_pos >= 0:
            # Model output starts right after the marker line
            model_output_start = last_marker_pos + len(self._structured_marker)
            text_after_think = text_after_think[model_output_start:]
            text_start_offset = think_end_pos + model_output_start
        else:
            text_start_offset = think_end_pos

        # Skip past feedback blocks
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        text = text_after_think[last_feedback_end:]
        text_start = text_start_offset + last_feedback_end

        # For turn-counting questions, check for structured steps
        if self.question_type in ("right_turns", "total_turns"):
            # Check for complete step (with Running count including Right=N, Left=N)
            step_pattern = re.compile(
                r'(>>>\s*STEP\s+(\d+):\s*Move\s+\w+\s+from\s+\([^)]+\)\s+to\s+\([^)]+\).*?'
                r'Running count:\s*Right\s*=\s*\d+\s*,\s*Left\s*=\s*\d+[^\n]*)',
                re.IGNORECASE | re.DOTALL
            )
            all_steps = list(step_pattern.finditer(text))

            if all_steps:
                last_step = all_steps[-1]
                # Check if next step started (current already verified)
                text_after = text[last_step.end():]
                next_step = re.search(r'>>>\s*STEP\s+\d+', text_after, re.IGNORECASE)
                if not next_step:
                    end_pos = text_start + last_step.end()
                    return True, generated_text[:end_pos]
                return False, None

            # Check LOCATE section
            locate_pattern = re.compile(
                r'(LOCATE START AND EXIT.*?E position:\s*\([^)]+\))',
                re.IGNORECASE | re.DOTALL
            )
            locate_match = locate_pattern.search(text)
            if locate_match:
                step1_start = re.search(r'>>>\s*STEP\s+1', text[locate_match.end():], re.IGNORECASE)
                if step1_start:
                    end_pos = text_start + locate_match.end()
                    return True, generated_text[:end_pos]

        # Check for boxed answer (any question type)
        boxed = re.search(r'\\boxed\{[^}]+\}', text)
        if boxed:
            end_pos = text_start + boxed.end()
            return True, generated_text[:end_pos]

        return False, None

    # ------------------------------------------------------------------
    #  verify
    # ------------------------------------------------------------------
    async def verify(self, step: str, token_index: int, event, event_info):
        """
        Case 1 -- still in thinking (no </think>):
            Fork side-stream to get traced path steps, verify each.
        Case 2 -- after </think>:
            Verify structured steps and/or final answer.
        """

        # ==================================================================
        # CASE 1: Thinking phase – side-stream path verification
        # ==================================================================
        if self.answer_start_token not in step:
            total_dn = step.count('\n\n')
            logger.info(
                f"[Maze Phase 1] Thinking-phase verification triggered\n"
                f"  \\n\\n count  : {total_dn}\n"
                f"  Thinking len : {len(step)} chars"
            )

            # Build text with injected prompt for step extraction
            # Uses the LLM's own voice: "Let me output the current steps..."
            text_with_prompt = step + self._thinking_phase_prompt

            # Side-stream: get path steps from the model
            side_output = await self._side_stream_maze_steps(
                text_with_prompt, max_new_tokens=300
            )

            if not side_output or len(side_output.strip()) < 20:
                logger.info(
                    "[Maze Phase 1] Insufficient output from side-stream. "
                    "Letting model continue thinking."
                )
                return step, None

            # Combine the prompt header with side output for parsing
            full_side_text = (
                ">>> LOCATE START AND EXIT (0-indexed, top-left is (0,0)):\n" + side_output
            )

            # First verify LOCATE section
            locate_valid, locate_errors = verify_locate_section(
                full_side_text, self.start_pos, self.exit_pos
            )

            if not locate_valid:
                self._think_phase_corrections += 1
                error_summary = "; ".join(locate_errors)
                logger.info(
                    f"[Maze Phase 1] LOCATE section errors: {error_summary}\n"
                    f"  Action: Inject feedback into thinking trace\n"
                    f"  Corrections: {self._think_phase_corrections}/{self.max_corrections}"
                )
                thinking_feedback = (
                    f"\n\nWait, I think I have the wrong positions. "
                    f"{error_summary}. "
                    f"Let me re-examine the maze grid carefully to find S and E.\n"
                )
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = thinking_feedback
                    event_info["correction_index"] = token_index
                    event_info["errors"] = locate_errors
                    event_info["phase"] = "rollback_to_thinking"
                    event.set()
                return step, thinking_feedback

            # Parse and verify steps
            steps = self._parse_steps_from_text(full_side_text)

            if not steps:
                logger.info(
                    "[Maze Phase 1] No structured steps found in side-stream. "
                    "Letting model continue thinking."
                )
                return step, None

            (all_valid, err_step_num, errors, final_pos,
             final_dir, r_count, l_count, t_count) = self._verify_all_steps(steps)

            if not all_valid:
                error_summary = "; ".join(errors)
                self._think_phase_corrections += 1
                logger.info(
                    f"[Maze Phase 1] INVALID step {err_step_num}\n"
                    f"  Error(s) : {error_summary}\n"
                    f"  Action   : Inject feedback into thinking trace\n"
                    f"  Corrections: {self._think_phase_corrections}/{self.max_corrections}"
                )
                thinking_feedback = (
                    f"\n\nWait, I made an error at Step {err_step_num}. "
                    f"{error_summary}. "
                    f"Let me re-trace the path more carefully from the correct position.\n"
                )
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = thinking_feedback
                    event_info["correction_index"] = token_index
                    event_info["errors"] = errors
                    event_info["phase"] = "rollback_to_thinking"
                    event.set()
                return step, thinking_feedback

            # All steps valid — check if path is complete (reached E)
            if final_pos == self.exit_pos:
                self._verified_path_complete = True
                logger.info(
                    f"[Maze Phase 1] VALID COMPLETE path to E={self.exit_pos}\n"
                    f"  Steps: {len(steps)}, Right={r_count}, Left={l_count}, Total={t_count}\n"
                    f"  Action: Inject early-stop + </think> + structured format."
                )
                # Include the structured prompt directly after </think>
                # so the model immediately starts filling in the answer format
                # (skips the separate Phase 2a injection round-trip).
                early_stop_msg = (
                    f"\n\nWait, I have successfully traced the path from "
                    f"S={self.start_pos} to E={self.exit_pos} with "
                    f"{len(steps)} steps. "
                    f"Right turns={r_count}, Left turns={l_count}, "
                    f"Total turns={t_count}. "
                    f"This path has been verified as correct. "
                    f"Let me give the final answer.\n"
                    f"{self.answer_start_token}"
                    f"{self._structured_prompt}"
                )
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = early_stop_msg
                    event_info["correction_index"] = token_index
                    event_info["phase"] = "early_stop_answer"
                    event_info["verified_counts"] = {
                        "right": r_count,
                        "left": l_count,
                        "total": t_count,
                        "steps": len(steps),
                    }
                    event.set()
                return step, early_stop_msg

            else:
                logger.info(
                    f"[Maze Phase 1] VALID PARTIAL path\n"
                    f"  Current pos: {final_pos}, Target: {self.exit_pos}\n"
                    f"  Steps so far: {len(steps)}\n"
                    f"  Action: No error, let model keep thinking."
                )
                return step, None

        # ==================================================================
        # CASE 2a: </think> present but structured prompt not yet injected
        # ==================================================================
        if not self._structured_prompt_injected(step):
            logger.info(
                "[Maze Phase 2a] </think> detected. "
                "Injecting structured step format."
            )
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = self._structured_prompt
                event_info["correction_index"] = token_index
                event_info["phase"] = "inject_structured_prompt"
                event.set()
            return step, self._structured_prompt

        # ==================================================================
        # CASE 2b: Structured prompt injected — verify output
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

        # Strip the injected <format>...</format> template — only look at
        # actual model output starting from the last ">>> LOCATE START AND EXIT (0-indexed, ...)" marker.
        last_marker_pos = text_after_think.rfind(self._structured_marker)
        if last_marker_pos >= 0:
            text_after_think = text_after_think[last_marker_pos:]

        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        recent_text = text_after_think[last_feedback_end:]

        # --- Verify LOCATE section ---
        locate_match = re.search(r'LOCATE START AND EXIT', recent_text, re.IGNORECASE)
        if locate_match:
            step1_start = re.search(r'>>>\s*STEP\s+1', recent_text, re.IGNORECASE)
            if step1_start or '\\boxed' in recent_text:
                if step1_start:
                    locate_text = recent_text[locate_match.start():step1_start.start()]
                else:
                    locate_text = recent_text[locate_match.start():]
                is_valid, loc_errors = verify_locate_section(
                    locate_text, self.start_pos, self.exit_pos
                )
                if not is_valid:
                    feedback = format_locate_feedback(loc_errors)
                    if not event.is_set():
                        event_info["generated_text"] = step
                        event_info["feedback"] = feedback
                        event_info["correction_index"] = token_index
                        event_info["errors"] = loc_errors
                        event_info["phase"] = "standard_verify"
                        event.set()
                    return step, feedback

        # --- Verify structured steps ---
        if self.question_type in ("right_turns", "total_turns"):
            step_pattern = re.compile(
                r'(>>>\s*STEP\s+(\d+):\s*Move\s+\w+\s+from\s+\([^)]+\)\s+to\s+\([^)]+\).*?'
                r'Running count:[^\n]+)',
                re.IGNORECASE | re.DOTALL
            )
            # Find steps in recent_text (after last feedback) to know what to verify
            recent_step_matches = list(step_pattern.finditer(recent_text))

            if recent_step_matches:
                last_match = recent_step_matches[-1]
                last_step_text = last_match.group(0)
                last_step_num = int(last_match.group(2))
                parsed = parse_maze_step(last_step_text)

                if parsed:
                    # For state reconstruction, gather ALL steps from the
                    # full text (not just recent_text).  When a step number
                    # appears multiple times (original + corrections), only
                    # the LAST occurrence before the target step is used.
                    all_full_matches = list(step_pattern.finditer(text_after_think))
                    state = self._get_state_before_step_phase2(
                        text_after_think, last_step_num, all_full_matches
                    )

                    is_valid, errors, new_state = verify_maze_step(
                        step=parsed,
                        grid=self.grid,
                        expected_from_pos=state['position'],
                        prev_direction=state['direction'],
                        expected_right_count=state['right_count'],
                        expected_left_count=state['left_count'],
                        expected_total_count=state['total_count'],
                    )

                    if not is_valid:
                        feedback = format_maze_feedback(errors, last_step_num)
                        if not event.is_set():
                            event_info["generated_text"] = step
                            event_info["feedback"] = feedback
                            event_info["correction_index"] = token_index
                            event_info["errors"] = errors
                            event_info["phase"] = "standard_verify"
                            event.set()
                        return step, feedback

        # --- Check for boxed answer ---
        boxed_answer = self._extract_boxed_answer(recent_text)
        if boxed_answer is not None:
            logger.info(f"[Maze Phase 2b] Extracted boxed answer: {boxed_answer}")

            # For relative_position questions, verify the Yes/No answer
            if self.question_type == "relative_position":
                is_correct, rp_feedback = self._verify_relative_position_answer(boxed_answer)
                if not is_correct and rp_feedback:
                    logger.info(
                        f"[Maze Phase 2b] Relative position answer '{boxed_answer}' is INCORRECT."
                    )
                    if not event.is_set():
                        event_info["generated_text"] = step
                        event_info["feedback"] = rp_feedback
                        event_info["correction_index"] = token_index
                        event_info["errors"] = [f"Wrong relative position answer: {boxed_answer}"]
                        event_info["phase"] = "standard_verify"
                        event.set()
                    return step, rp_feedback

            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = ""
                event_info["correction_index"] = token_index
                event_info["phase"] = "final_answer_correct"
                event.set()
            return step, None

        return step, None

    # ------------------------------------------------------------------
    #  _get_state_before_step_phase2 – reconstruct state for Phase 2
    # ------------------------------------------------------------------
    def _get_state_before_step_phase2(self, text: str, target_step_num: int,
                                       all_step_matches: list) -> dict:
        """Reconstruct state before a given step from Phase 2 structured output.
        
        When a step number appears multiple times (original + corrections after
        verifier feedback), only the LAST occurrence of each step number is used,
        so that corrected steps override earlier invalid ones.
        """
        state = {
            'position': self.start_pos,
            'direction': Direction.NONE,
            'right_count': 0,
            'left_count': 0,
            'total_count': 0,
        }

        # Collect the last occurrence of each step number before the target
        last_by_num = {}
        for match in all_step_matches:
            step_num = int(match.group(2))
            if step_num >= target_step_num:
                continue
            last_by_num[step_num] = match  # later occurrences overwrite earlier

        # Replay in step-number order
        for step_num in sorted(last_by_num.keys()):
            parsed = parse_maze_step(last_by_num[step_num].group(0))
            if not parsed:
                continue

            direction = parsed['direction']
            to_pos = parsed['to_pos']

            turn_type = get_expected_turn_type(state['direction'], direction)
            if turn_type == 'RIGHT_TURN':
                state['right_count'] += 1
                state['total_count'] += 1
            elif turn_type == 'LEFT_TURN':
                state['left_count'] += 1
                state['total_count'] += 1

            state['position'] = to_pos
            state['direction'] = direction

        return state

    # ------------------------------------------------------------------
    #  fix
    # ------------------------------------------------------------------
    async def fix(self, generated_text: str, event_info: dict, fix_method=None):
        """Apply the appropriate fix depending on the phase."""
        phase = event_info.get("phase", "standard_verify")

        if phase == "rollback_to_thinking":
            base_text = event_info["generated_text"]
            result = base_text.rstrip() + event_info["feedback"]
            logger.info(
                f"[Maze fix] Phase: rollback_to_thinking\n"
                f"  -> Appended error feedback into <think> trace.\n"
                f"  -> Think-phase corrections: {self._think_phase_corrections}/{self.max_corrections}"
            )
            return result

        if phase == "early_stop_answer":
            base_text = event_info["generated_text"]
            result = base_text.rstrip() + event_info["feedback"]
            counts = event_info.get("verified_counts", {})
            logger.info(
                f"[Maze fix] Phase: early_stop_answer\n"
                f"  -> Path verified: {counts.get('steps', '?')} steps, "
                f"R={counts.get('right', '?')}, L={counts.get('left', '?')}, "
                f"T={counts.get('total', '?')}\n"
                f"  -> Injecting early-stop + </think> + structured format."
            )
            return result

        if phase == "inject_structured_prompt":
            logger.info(
                "[Maze fix] Phase: inject_structured_prompt\n"
                "  -> Appending structured step format after </think>."
            )
            return event_info["generated_text"] + event_info["feedback"]

        if phase == "final_answer_correct":
            logger.info(
                f"[Maze fix] Phase: final_answer_correct\n"
                f"  -> Stopping generation."
            )
            return event_info["generated_text"]

        # standard_verify
        errors = event_info.get("errors", [])
        error_summary = "; ".join(errors) if errors else "unknown"
        logger.info(
            f"[Maze fix] Phase: standard_verify\n"
            f"  -> Error: {error_summary}\n"
            f"  -> Appending [VERIFIER FEEDBACK] so model retries."
        )
        return event_info["generated_text"] + event_info["feedback"]


# =====================================================================
#  SpatialMap Thinking-Phase Prompts
# =====================================================================


def _build_spatialmap_format_block() -> str:
    """
    Build the ``<format>...</format>`` block that describes the structured
    output template for SpatialMap tasks.

    Re-used by both the side-stream (Phase 1) and the post-``</think>``
    injection (Phase 2a).
    """
    return (
        "<format>\n"
        ">>> STEP 1: PARSE RELATIONSHIPS\n"
        "    - A is to the DIRECTION of B\n"
        "    [... list all given relationships ...]\n"
        "\n"
        ">>> STEP 2: ANALYZE SPATIAL RELATIONSHIPS\n"
        "    - Looking for: [target relationship / direction / count]\n"
        "    - [reasoning about the relationships]\n"
        "    - [use reversibility and transitivity as needed]\n"
        "\n"
        ">>> STEP 3: ANSWER\n"
        "    - [state conclusion]\n"
        "\n"
        ">>> FINAL ANSWER: [answer text]\n"
        "    \\boxed{LETTER}\n"
        "</format>"
    )


def _build_spatialmap_thinking_phase_prompt(
    parsed_relations: List[Dict],
) -> str:
    """
    Build the side-stream prompt injected during the thinking phase.

    Pre-fills STEP 1 with the known parsed relations (from the Z3 solver)
    so the model jumps directly to STEP 2 analysis, maximising the chance
    of producing verifiable directional claims within the token budget.

    Written in the LLM's own first-person thinking voice so it blends
    naturally with the ``<think>`` trace.
    """
    # Pre-fill STEP 1 from the ground-truth parsed relations
    step1_lines = []
    for rel in parsed_relations:
        step1_lines.append(
            f"    - {rel['A']} is to the {rel['direction']} of {rel['B']}"
        )
    step1_body = "\n".join(step1_lines) if step1_lines else "    (none)"

    return (
        "\n\nLet me organize what I have so far. I will list the given "
        "relationships in STEP 1, then in STEP 2 I will state every "
        "spatial claim I have derived using FULL object names (no "
        "abbreviations) in exactly this form:\n"
        "    - [Full Name A] is to the [direction] of [Full Name B]\n"
        "For direction I will use the full word: northeast, northwest, "
        "southeast, southwest, north, south, east, or west.\n\n"
        ">>> STEP 1: PARSE RELATIONSHIPS (given)\n"
        f"{step1_body}\n\n"
        ">>> STEP 2: ANALYZE SPATIAL RELATIONSHIPS (derived)\n"
        "Based on my analysis so far, the derived relationships are:\n"
    )


def _build_spatialmap_structured_prompt() -> str:
    """
    Build the structured format prompt injected after ``</think>``.

    Analogous to the maze's structured format injection — gives the
    model a template to fill in so we can parse and verify each step.
    """
    format_block = _build_spatialmap_format_block()
    return (
        "\nLet me solve this step by step using the structured format:\n"
        f"{format_block}\n"
        ">>> STEP 1: PARSE RELATIONSHIPS\n"
    )


# =====================================================================
#  ThinkingPhaseStepVerifierSpatialMapMonitor
# =====================================================================


class ThinkingPhaseStepVerifierSpatialMapMonitor(VerifyMonitor):
    """
    Monitor that verifies spatial-map directional claims during and after
    thinking.

    **No meta-prompt required** — works with a plain user prompt containing
    just the map description and question.  Structure is injected by this
    monitor after ``</think>`` (natural or early-stop), exactly like the
    Maze monitor injects its step format.

    Phase 1 – During ``<think>...</think>``:
        Every N double-newlines (after warmup), fork a side-stream that
        injects a structured step prompt, stream tokens, parse directional
        claims from STEP 2, and verify them against Z3.

    Phase 2a – ``</think>`` detected, structured prompt not yet injected:
        Inject the structured step-by-step format template so the model
        fills it in (STEP 1 → STEP 2 → STEP 3 → FINAL ANSWER → ``\\boxed{}``).

    Phase 2b – Structured prompt injected, model is generating:
        Verify directional claims in STEP 2 as they appear.  Once
        ``\\boxed{}`` appears, signal completion.
    """

    def __init__(
        self,
        name: str,
        problem_text: str,
        llm_server: dict,
        prompt: str,
        newline_threshold: int = 15,
        max_corrections: int = 5,
        answer_start_token: str = "</think>",
        async_execution: bool = True,
        warmup_newlines: int = 0,
    ):
        super().__init__(name)
        self.problem_text = problem_text
        self.llm_server = llm_server
        self.prompt = prompt
        self.newline_threshold = newline_threshold
        self.max_corrections = max_corrections
        self.answer_start_token = answer_start_token
        self.async_execution = async_execution
        self.warmup_newlines = warmup_newlines

        # Initialize Z3 solver with problem constraints
        self.z3_solver = SpatialMapZ3Solver(problem_text)

        # Build prompts for injection
        self._structured_prompt = _build_spatialmap_structured_prompt()
        self._thinking_phase_prompt = _build_spatialmap_thinking_phase_prompt(
            self.z3_solver.parsed_relations,
        )
        # Marker to detect if structured prompt was already injected
        self._structured_marker = ">>> STEP 1: PARSE RELATIONSHIPS"

        # ---- state ----
        self._think_phase_corrections = 0
        self.verified_claims: Set[Tuple[str, str, str]] = set()

        # ---- counting-question verification ----
        self._counting_question = parse_counting_question(problem_text)
        self._counting_options: Dict[str, str] = {}
        # Strip trailing instruction paragraph for clean option parsing
        _opts_text = re.split(r'\nFirst,', problem_text, maxsplit=1)[0]
        if self._counting_question:
            # Parse MCQ options from problem text (e.g., "A. 5\nB. 3\nC. 0\nD. 1")
            raw_opts = re.findall(
                r'([A-D])\.\s*(.+?)\s*(?=[A-D]\.|$)',
                _opts_text, flags=re.DOTALL,
            )
            self._counting_options = {
                k: v.strip().rstrip(".") for k, v in raw_opts
            }
            logger.info(
                f"[SpatialMap] Counting question detected: "
                f"direction={self._counting_question['direction']}, "
                f"reference={self._counting_question['reference']}, "
                f"options={self._counting_options}"
            )
        self._count_feedback_given = False
        self._count_feedback_blocks_count = 0  # tracks cardinal count retry attempts

        # ---- direction-question verification ----
        self._direction_question = parse_direction_question(problem_text)
        if self._direction_question:
            logger.info(
                f"[SpatialMap] Direction question detected: "
                f"entity_a={self._direction_question['entity_a']}, "
                f"entity_b={self._direction_question['entity_b']}"
            )

        # ---- object-question verification ----
        self._object_question = parse_object_question(problem_text)
        if self._object_question:
            logger.info(
                f"[SpatialMap] Object question detected: "
                f"direction={self._object_question['direction']}, "
                f"reference={self._object_question['reference']}"
            )

        # ---- Generic MCQ options (for direction & object Qs too) ----
        if not self._counting_options:
            raw_opts = re.findall(
                r'([A-D])\.\s*(.+?)\s*(?=[A-D]\.|$)',
                _opts_text, flags=re.DOTALL,
            )
            self._mcq_options: Dict[str, str] = {
                k: v.strip().rstrip(".") for k, v in raw_opts
            }
        else:
            self._mcq_options = dict(self._counting_options)

        # Allow multiple retries for final-answer verification
        self._max_final_answer_retries = 3
        self._direction_feedback_count = 0
        self._object_feedback_count = 0
        self._diag_count_feedback_count = 0

    @classmethod
    def from_prompt(
        cls,
        problem_text: str,
        llm_server: dict,
        prompt: str,
        newline_threshold: int = 15,
        max_corrections: int = 5,
        warmup_newlines: int = 0,
        name: str = "spatialmap_thinking_verifier",
    ) -> "ThinkingPhaseStepVerifierSpatialMapMonitor":
        """
        Convenience factory method.
        """
        return cls(
            name=name,
            problem_text=problem_text,
            llm_server=llm_server,
            prompt=prompt,
            newline_threshold=newline_threshold,
            max_corrections=max_corrections,
            warmup_newlines=warmup_newlines,
        )

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------
    def _count_feedback_blocks(self, text: str) -> int:
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))

    def _is_in_thinking_phase(self, generated_text: str) -> bool:
        return self.answer_start_token not in generated_text

    def _structured_prompt_injected(self, generated_text: str) -> bool:
        """Check if structured format was already injected after </think>."""
        if self.answer_start_token not in generated_text:
            return False
        after_think = generated_text.split(self.answer_start_token, 1)[1]
        return self._structured_marker in after_think

    def _extract_new_claims(self, text: str) -> List[Dict]:
        """
        Extract new (not yet verified) directional claims from STEP 2 of
        the most recent attempt (after last feedback block).
        """
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text):
            last_feedback_end = match.end()

        text_to_check = text[last_feedback_end:]

        all_claims = extract_step2_claims(text_to_check)

        new_claims = []
        for claim in all_claims:
            claim_key = (claim['A'], claim['direction'], claim['B'])
            if claim_key not in self.verified_claims:
                new_claims.append(claim)

        return new_claims

    # ------------------------------------------------------------------
    #  _side_stream_spatialmap – streams tokens to get analysis
    # ------------------------------------------------------------------
    async def _side_stream_spatialmap(self, text_so_far: str, max_new_tokens: int = 400) -> str:
        """
        Send ``prompt + text_so_far`` to vLLM, stream at most
        *max_new_tokens* tokens, and return the generated text.

        ``text_so_far`` is expected to end with the structured spatial map
        prompt so the model outputs its analysis steps.
        """
        logger.info(
            f"[SpatialMap Side-stream] Starting analysis extraction\n"
            f"  Relations: {len(self.z3_solver.parsed_relations)}\n"
            f"  Max new tokens: {max_new_tokens}"
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
                        logger.debug(f"[SpatialMap Side-stream] chunk: {chunk!r}")

                        # Stop if we see FINAL ANSWER or \boxed
                        if '\\boxed' in generated or '>>> FINAL ANSWER' in generated:
                            break

        logger.info(
            f"[SpatialMap Side-stream] Generated {len(generated)} chars"
        )
        return generated

    # ------------------------------------------------------------------
    #  step_extractor
    # ------------------------------------------------------------------
    def step_extractor(self, chunk: str, generated_text: str):
        """
        Phase 1 (thinking): trigger at every newline_threshold multiple
            (after warmup).
        Phase 2 (after </think>): trigger on structured steps or boxed
            answer.
        """
        # ===== PHASE 1: still inside <think> =====
        if self._is_in_thinking_phase(generated_text):
            if self._think_phase_corrections >= self.max_corrections:
                return False, None

            total_double_newlines = generated_text.count('\n\n')

            if total_double_newlines < self.warmup_newlines:
                return False, None

            past_warmup = total_double_newlines - self.warmup_newlines
            if (generated_text.endswith('\n\n')
                    and past_warmup >= 0
                    and past_warmup % self.newline_threshold == 0):
                logger.info(
                    f"[SpatialMap step_extractor] Phase 1 trigger: \\n\\n count={total_double_newlines} "
                    f"(warmup={self.warmup_newlines}, past_warmup={past_warmup}, "
                    f"threshold={self.newline_threshold})"
                )
                return True, generated_text

            return False, None

        # ===== PHASE 2: after </think> =====

        # 2a: structured prompt not yet injected → trigger immediately
        if not self._structured_prompt_injected(generated_text):
            logger.info(
                "[SpatialMap step_extractor] Phase 2a: </think> detected, "
                "structured prompt not yet injected."
            )
            return True, generated_text

        # 2b: structured prompt injected — verify STEP 2 claims / boxed answer
        think_end_pos = generated_text.find(self.answer_start_token) + len(self.answer_start_token)
        text_after_think = generated_text[think_end_pos:]

        # Strip out the injected <format>...</format> template so we only
        # look at actual model output (which starts after the last marker).
        last_marker_pos = text_after_think.rfind(self._structured_marker)
        if last_marker_pos >= 0:
            model_output_start = last_marker_pos + len(self._structured_marker)
            text_after_think = text_after_think[model_output_start:]
            text_start_offset = think_end_pos + model_output_start
        else:
            text_start_offset = think_end_pos

        # Skip past feedback blocks
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        text = text_after_think[last_feedback_end:]
        text_start = text_start_offset + last_feedback_end

        # Check for STEP 2 section with claims
        step2_pattern = re.compile(
            r'>>>\s*STEP\s*2[:\s].*?(?=>>>\s*STEP\s*3|>>>\s*FINAL|\\boxed|$)',
            re.DOTALL | re.IGNORECASE
        )
        step2_match = step2_pattern.search(text)

        if step2_match:
            # Check if STEP 3 or FINAL has started (STEP 2 is complete)
            text_after_step2 = text[step2_match.end():]
            step3_or_final = re.search(
                r'>>>\s*(STEP\s*3|FINAL)',
                text_after_step2,
                re.IGNORECASE
            )

            if step3_or_final:
                new_claims = self._extract_new_claims(text)
                if new_claims:
                    end_pos = text_start + step2_match.end()
                    return True, generated_text[:end_pos]

        # Check for boxed answer (trigger final verification)
        boxed_match = re.search(r'\\boxed\{[^}]+\}', text)
        if boxed_match:
            new_claims = self._extract_new_claims(text)
            if new_claims:
                end_pos = text_start + boxed_match.end()
                return True, generated_text[:end_pos]
            # Even if no new claims, boxed answer signals completion
            end_pos = text_start + boxed_match.end()
            return True, generated_text[:end_pos]

        return False, None

    # ------------------------------------------------------------------
    #  verify
    # ------------------------------------------------------------------
    async def verify(self, step: str, token_index: int, event, event_info):
        """
        Case 1 -- still in thinking (no </think>):
            Fork side-stream, parse claims, verify with Z3.
        Case 2 -- after </think>:
            2a: Inject structured prompt.
            2b: Verify STEP 2 claims and/or final answer.
        """

        # ==================================================================
        # CASE 1: Thinking phase – side-stream verification
        # ==================================================================
        if self.answer_start_token not in step:
            total_dn = step.count('\n\n')
            logger.info(
                f"[SpatialMap Phase 1] Thinking-phase verification triggered\n"
                f"  \\n\\n count  : {total_dn}\n"
                f"  Thinking len : {len(step)} chars"
            )

            # Build text with injected prompt for analysis extraction
            text_with_prompt = step + self._thinking_phase_prompt

            # Side-stream: get analysis from the model
            side_output = await self._side_stream_spatialmap(
                text_with_prompt, max_new_tokens=800
            )

            if not side_output or len(side_output.strip()) < 20:
                logger.info(
                    "[SpatialMap Phase 1] Insufficient output from side-stream. "
                    "Letting model continue thinking."
                )
                return step, None

            # Parse directional claims directly from the side-stream output.
            # The prompt pre-fills STEP 1 and ends at ">>> STEP 2:", so the
            # model's output is already STEP 2 content — no header to search for.
            claims = parse_directional_claims_from_text(side_output)

            logger.info(
                f"[SpatialMap Phase 1] Parsed {len(claims)} claims from side-stream.\n"
                f"  Side-stream output (first 500 chars): {side_output[:500]!r}"
            )

            if not claims:
                logger.info(
                    "[SpatialMap Phase 1] No directional claims found in side-stream. "
                    "Letting model continue thinking."
                )
                return step, None

            # Verify each claim against Z3
            for claim in claims:
                claim_key = (claim['A'], claim['direction'], claim['B'])
                if claim_key in self.verified_claims:
                    continue

                is_valid, errors = verify_spatialmap_step(
                    claim=claim,
                    z3_solver=self.z3_solver,
                    add_if_valid=True,
                )
                self.verified_claims.add(claim_key)

                if not is_valid:
                    self._think_phase_corrections += 1
                    error_summary = "; ".join(errors)
                    logger.info(
                        f"[SpatialMap Phase 1] INVALID claim: "
                        f"{claim['A']} is {claim['direction']} of {claim['B']}\n"
                        f"  Error(s) : {error_summary}\n"
                        f"  Corrections: {self._think_phase_corrections}/{self.max_corrections}"
                    )
                    thinking_feedback = (
                        f"\n\nWait, I think I made an error in my spatial reasoning. "
                        f"{error_summary}. "
                        f"Let me re-examine the relationships more carefully.\n"
                    )
                    if not event.is_set():
                        event_info["generated_text"] = step
                        event_info["feedback"] = thinking_feedback
                        event_info["correction_index"] = token_index
                        event_info["errors"] = errors
                        event_info["phase"] = "rollback_to_thinking"
                        event.set()
                    return step, thinking_feedback

            # All claims valid
            logger.info(
                f"[SpatialMap Phase 1] All {len(claims)} claims valid. "
                f"Letting model continue thinking."
            )
            return step, None

        # ==================================================================
        # CASE 2a: </think> present but structured prompt not yet injected
        # ==================================================================
        if not self._structured_prompt_injected(step):
            logger.info(
                "[SpatialMap Phase 2a] </think> detected. "
                "Injecting structured step format."
            )
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = self._structured_prompt
                event_info["correction_index"] = token_index
                event_info["phase"] = "inject_structured_prompt"
                event.set()
            return step, self._structured_prompt

        # ==================================================================
        # CASE 2b: Structured prompt injected — verify output
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

        # Strip the injected template — only look at model output after marker
        last_marker_pos = text_after_think.rfind(self._structured_marker)
        if last_marker_pos >= 0:
            text_after_think = text_after_think[last_marker_pos:]

        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        recent_text = text_after_think[last_feedback_end:]

        # --- Verify STEP 2 claims ---
        new_claims = self._extract_new_claims(recent_text)

        for claim in new_claims:
            claim_key = (claim['A'], claim['direction'], claim['B'])

            is_valid, errors = verify_spatialmap_step(
                claim=claim,
                z3_solver=self.z3_solver,
                add_if_valid=True,
            )
            self.verified_claims.add(claim_key)

            if not is_valid:
                feedback = format_spatialmap_feedback(errors, claim)
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = feedback
                    event_info["correction_index"] = token_index
                    event_info["errors"] = errors
                    event_info["failed_step"] = claim
                    event_info["phase"] = "standard_verify"
                    event.set()
                return step, feedback

        # --- Check for boxed answer ---
        boxed_match = re.search(r'\\boxed\{[^}]+\}', recent_text)
        if boxed_match:

            # ==========================================================
            # Direction-question verification
            # ==========================================================
            if (
                self._direction_question
                and num_corrections < self.max_corrections
                and self._direction_feedback_count < self._max_final_answer_retries
            ):
                model_dir_text = parse_model_boxed_answer(
                    recent_text, self._mcq_options
                )
                if model_dir_text:
                    possible = get_possible_directions(
                        self.z3_solver,
                        self._direction_question["entity_a"],
                        self._direction_question["entity_b"],
                    )
                    logger.info(
                        f"[SpatialMap Phase 2b] Direction check: "
                        f"model={model_dir_text}, possible={possible}"
                    )
                    if model_dir_text not in possible:
                        self._direction_feedback_count += 1
                        # Find which MCQ options are consistent
                        valid_options = [
                            letter for letter, val in self._mcq_options.items()
                            if val.strip().lower().rstrip(".") in possible
                        ]
                        if len(valid_options) == 1:
                            # Force correct answer
                            feedback = (
                                f"\n\n[VERIFIER FEEDBACK: Direction error!\n"
                                f"  '{model_dir_text.title()}' is "
                                f"impossible for "
                                f"{self._direction_question['entity_a']} "
                                f"relative to "
                                f"{self._direction_question['entity_b']} "
                                f"based on the given constraints.\n"
                                f"  The only consistent direction is "
                                f"'{possible[0].title()}'.\n"
                                f"  Please select option "
                                f"{valid_options[0]}.]\n\n"
                                f">>> STEP 3: ANSWER\n"
                            )
                        else:
                            possible_str = ", ".join(
                                d.title() for d in possible
                            )
                            feedback = (
                                f"\n\n[VERIFIER FEEDBACK: Direction error!\n"
                                f"  '{model_dir_text.title()}' is "
                                f"impossible for "
                                f"{self._direction_question['entity_a']} "
                                f"relative to "
                                f"{self._direction_question['entity_b']} "
                                f"based on the given constraints.\n"
                                f"  The possible directions are: "
                                f"{possible_str}.\n"
                                f"  Please reconsider and choose the "
                                f"correct option.]\n\n"
                                f">>> STEP 3: ANSWER\n"
                            )
                        if not event.is_set():
                            event_info["generated_text"] = step
                            event_info["feedback"] = feedback
                            event_info["correction_index"] = token_index
                            event_info["errors"] = [
                                f"Direction '{model_dir_text}' impossible; "
                                f"possible: {possible}"
                            ]
                            event_info["phase"] = "standard_verify"
                            event.set()
                        return step, feedback

            # ==========================================================
            # Object-question verification
            # ==========================================================
            if (
                self._object_question
                and num_corrections < self.max_corrections
                and self._object_feedback_count < self._max_final_answer_retries
            ):
                model_obj_text = parse_model_boxed_answer(
                    recent_text, self._mcq_options
                )
                boxed_raw = re.findall(
                    r'\\boxed\{([^}]*)\}', recent_text
                )
                model_letter = (
                    boxed_raw[-1].strip().upper() if boxed_raw else None
                )

                if model_letter:
                    consistent = get_consistent_object_options(
                        self.z3_solver,
                        self._object_question["direction"],
                        self._object_question["reference"],
                        self._mcq_options,
                    )
                    logger.info(
                        f"[SpatialMap Phase 2b] Object check: "
                        f"model={model_letter}, "
                        f"consistent_options={consistent}"
                    )
                    if model_letter not in consistent:
                        self._object_feedback_count += 1
                        odir = self._object_question["direction"]
                        oref = self._object_question["reference"]
                        if len(consistent) == 1:
                            correct_name = self._mcq_options.get(
                                consistent[0], consistent[0]
                            )
                            feedback = (
                                f"\n\n[VERIFIER FEEDBACK: Object error!\n"
                                f"  '{model_obj_text}' cannot be "
                                f"{odir} of {oref} based on the "
                                f"given constraints.\n"
                                f"  The only consistent option is "
                                f"{consistent[0]}. {correct_name}.\n"
                                f"  Please select option "
                                f"{consistent[0]}.]\n\n"
                                f">>> STEP 3: ANSWER\n"
                            )
                        else:
                            valid_names = [
                                f"{l}. {self._mcq_options.get(l, l)}"
                                for l in consistent
                            ]
                            feedback = (
                                f"\n\n[VERIFIER FEEDBACK: Object error!\n"
                                f"  '{model_obj_text}' cannot be "
                                f"{odir} of {oref} based on the "
                                f"given constraints.\n"
                                f"  The consistent options are: "
                                f"{', '.join(valid_names)}.\n"
                                f"  Please reconsider and choose the "
                                f"correct option.]\n\n"
                                f">>> STEP 3: ANSWER\n"
                            )
                        if not event.is_set():
                            event_info["generated_text"] = step
                            event_info["feedback"] = feedback
                            event_info["correction_index"] = token_index
                            event_info["errors"] = [
                                f"Object '{model_obj_text}' impossible "
                                f"in {odir} of {oref}; "
                                f"consistent: {consistent}"
                            ]
                            event_info["phase"] = "standard_verify"
                            event.set()
                        return step, feedback

            # ==========================================================
            # Counting-question verification (cardinal + diagonal)
            # ==========================================================
            if (
                self._counting_question
                and num_corrections < self.max_corrections
            ):
                direction = self._counting_question["direction"]
                reference = self._counting_question["reference"]
                is_cardinal = direction in (
                    "north", "south", "east", "west"
                )

                if is_cardinal:
                    # --- Cardinal: GT is always 0 ---
                    model_count = parse_model_count_from_answer(
                        recent_text, self._counting_options
                    )
                    z3_count = 0

                    logger.info(
                        f"[SpatialMap Phase 2b] Cardinal count check: "
                        f"model={model_count}, expected={z3_count}, "
                        f"direction={direction}, reference={reference}"
                    )

                    if (
                        model_count is not None
                        and model_count != z3_count
                    ):
                        self._count_feedback_given = True
                        count_corrections = self._count_feedback_blocks_count
                        self._count_feedback_blocks_count = count_corrections + 1

                        if count_corrections == 0:
                            # First attempt: explain why cardinal = 0
                            if direction in ("north", "south"):
                                diag_examples = "northeast or northwest"
                            elif direction == "west":
                                diag_examples = "northwest or southwest"
                            else:  # east
                                diag_examples = "northeast or southeast"

                            feedback = (
                                f"\n\n[VERIFIER FEEDBACK: Count mismatch!\n"
                                f"  You answered {model_count} objects "
                                f"'{direction}' of {reference}, but the "
                                f"correct count is {z3_count}.\n"
                                f"  IMPORTANT: '{direction.title()}' means "
                                f"STRICTLY and EXACTLY {direction} — it "
                                f"does NOT include diagonal directions "
                                f"like {diag_examples}.\n"
                                f"  An object that is Northwest of "
                                f"{reference} is NOT North of {reference}"
                                f" and NOT West of {reference}.\n"
                                f"  Since all given relationships in this "
                                f"problem are diagonal (NE/NW/SE/SW), no "
                                f"object can be strictly "
                                f"'{direction.title()}' of {reference}.\n"
                                f"  The correct count is {z3_count}. "
                                f"Please select the option for 0.]\n\n"
                                f">>> STEP 3: ANSWER\n"
                            )
                        else:
                            # Subsequent attempts: force the correct answer directly
                            correct_option = None
                            for opt, val in self._counting_options.items():
                                if val == "0":
                                    correct_option = opt
                                    break
                            if correct_option:
                                feedback = (
                                    f"\nThe correct answer is 0. "
                                    f"\\boxed{{{correct_option}}}"
                                )
                            else:
                                feedback = (
                                    f"\nThe correct answer is 0. "
                                    f"\\boxed{{0}}"
                                )

                        logger.info(
                            f"[SpatialMap Phase 2b] Cardinal count "
                            f"mismatch: model={model_count}, "
                            f"expected=0. Injecting feedback "
                            f"(attempt={'1st' if not self._count_feedback_given else '2nd'})."
                        )
                        if not event.is_set():
                            event_info["generated_text"] = step
                            event_info["feedback"] = feedback
                            event_info["correction_index"] = token_index
                            event_info["errors"] = [
                                f"Cardinal count mismatch: expected 0, "
                                f"got {model_count}"
                            ]
                            event_info["phase"] = "standard_verify"
                            event.set()
                        return step, feedback

                else:
                    # --- Diagonal: use Z3 range check ---
                    if self._diag_count_feedback_count < self._max_final_answer_retries:
                        model_count = parse_model_count_from_answer(
                            recent_text, self._counting_options
                        )
                        count_range = get_possible_count_range(
                            self.z3_solver, reference, direction
                        )

                        if (
                            model_count is not None
                            and count_range is not None
                        ):
                            min_c, max_c = count_range
                            logger.info(
                                f"[SpatialMap Phase 2b] Diagonal count "
                                f"check: model={model_count}, "
                                f"range=[{min_c}, {max_c}], "
                                f"direction={direction}, "
                                f"reference={reference}"
                            )

                            if not (min_c <= model_count <= max_c):
                                self._diag_count_feedback_count += 1
                                # Find valid MCQ options
                                valid_opts = []
                                for opt, val in (
                                    self._counting_options.items()
                                ):
                                    try:
                                        v = int(val)
                                        if min_c <= v <= max_c:
                                            valid_opts.append(
                                                (opt, v)
                                            )
                                    except (ValueError, TypeError):
                                        pass

                                if len(valid_opts) == 1:
                                    feedback = (
                                        f"\n\n[VERIFIER FEEDBACK: "
                                        f"Count error!\n"
                                        f"  {model_count} objects "
                                        f"'{direction}' of {reference}"
                                        f" is impossible.\n"
                                        f"  The valid count is "
                                        f"{valid_opts[0][1]}.\n"
                                        f"  Please select option "
                                        f"{valid_opts[0][0]}.]\n\n"
                                        f">>> STEP 3: ANSWER\n"
                                    )
                                else:
                                    feedback = (
                                        f"\n\n[VERIFIER FEEDBACK: "
                                        f"Count error!\n"
                                        f"  {model_count} objects "
                                        f"'{direction}' of {reference}"
                                        f" is impossible.\n"
                                        f"  The possible count range "
                                        f"is [{min_c}, {max_c}].\n"
                                        f"  Please reconsider and "
                                        f"choose the correct "
                                        f"option.]\n\n"
                                        f">>> STEP 3: ANSWER\n"
                                    )

                                if not event.is_set():
                                    event_info["generated_text"] = step
                                    event_info["feedback"] = feedback
                                    event_info["correction_index"] = (
                                        token_index
                                    )
                                    event_info["errors"] = [
                                        f"Diagonal count "
                                        f"{model_count} outside "
                                        f"range [{min_c}, {max_c}]"
                                    ]
                                    event_info["phase"] = (
                                        "standard_verify"
                                    )
                                    event.set()
                                return step, feedback

            logger.info(
                f"[SpatialMap Phase 2b] Boxed answer found. Stopping."
            )
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = ""
                event_info["correction_index"] = token_index
                event_info["phase"] = "final_answer_correct"
                event.set()
            return step, None

        # All claims valid, no boxed yet
        return step, None

    # ------------------------------------------------------------------
    #  fix
    # ------------------------------------------------------------------
    async def fix(self, generated_text: str, event_info: dict, fix_method=None):
        """Apply the appropriate fix depending on the phase."""
        phase = event_info.get("phase", "standard_verify")

        if phase == "rollback_to_thinking":
            base_text = event_info["generated_text"]
            result = base_text.rstrip() + event_info["feedback"]
            logger.info(
                f"[SpatialMap fix] Phase: rollback_to_thinking\n"
                f"  -> Appended error feedback into <think> trace.\n"
                f"  -> Think-phase corrections: {self._think_phase_corrections}/{self.max_corrections}"
            )
            return result

        if phase == "inject_structured_prompt":
            logger.info(
                "[SpatialMap fix] Phase: inject_structured_prompt\n"
                "  -> Appending structured step format after </think>."
            )
            return event_info["generated_text"] + event_info["feedback"]

        if phase == "final_answer_correct":
            logger.info(
                "[SpatialMap fix] Phase: final_answer_correct\n"
                "  -> Stopping generation."
            )
            return event_info["generated_text"]

        # standard_verify
        errors = event_info.get("errors", [])
        error_summary = "; ".join(errors) if errors else "unknown"
        logger.info(
            f"[SpatialMap fix] Phase: standard_verify\n"
            f"  -> Error: {error_summary}\n"
            f"  -> Appending [VERIFIER FEEDBACK] so model retries."
        )
        return event_info["generated_text"] + event_info["feedback"]
