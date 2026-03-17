"""
Thinking-phase verifier for Maze tasks.

Verifies maze path-tracing by forking a side-stream during the
thinking phase to ask the model about its current traced path.

Workflow
--------
A) **DURING the thinking phase** (inside ``<think>...</think>``):
   After a warmup period, every *N* newlines in the thinking trace:
   1. Inject a first-person prompt to extract the traced path steps.
   2. Parse and verify each step against the maze grid.
   3. If **errors** -> inject feedback into thinking trace.
   4. If **path reaches E** -> inject early-stop + ``</think>`` +
      structured format.
   5. If **partial but correct** -> no feedback, keep thinking.

B) **AFTER ``</think>``**:
   Phase 2a: Inject structured step format template.
   Phase 2b: Verify each step as the model fills in the template.
   Once ``\\boxed{}`` appears, stop generation.
"""

import re
import json
import logging
from typing import Tuple, Optional
from copy import deepcopy

import httpx

from .base import VerifyMonitor
from ._common import find_complete_boxed
from ..utils.maze_verifier import (
    Direction, parse_direction, get_expected_turn_type,
    parse_maze_from_prompt, parse_maze_step, verify_maze_step,
    verify_locate_section, format_maze_feedback, format_locate_feedback,
    DIRECTION_DELTAS, compute_relative_direction,
)

logger = logging.getLogger(__name__)


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
        Every N newlines (after warmup), fork a side-stream that
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
        answer_map = {"A": "Yes", "B": "No"}
        model_yn = answer_map.get(boxed_answer.strip().upper())
        if model_yn is None:
            return True, None

        m = re.search(
            r'Is the exit \(E\)\s+(.*?)\s+(?:of\s+)?the starting point \(S\)',
            self.prompt, re.IGNORECASE,
        )
        if not m:
            return True, None

        asked_raw = m.group(1).strip().lower()
        asked_raw = re.sub(r',.*', '', asked_raw).strip()

        actual = compute_relative_direction(self.start_pos, self.exit_pos)

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
            return True, None

        expected_yn = "Yes" if actual in expected_dirs else "No"

        if model_yn == expected_yn:
            return True, None

        sr, sc = self.start_pos
        er, ec = self.exit_pos
        correct_letter = 'A' if expected_yn == 'Yes' else 'B'
        feedback = (
            f"\n\n[VERIFIER FEEDBACK for relative position:\n"
            f"  ✗ Your answer {boxed_answer} ({model_yn}) is incorrect.\n"
            f"  IMPORTANT: In this task, \"{asked_raw}\" means the GENERAL "
            f"COMPASS DIRECTION, NOT immediate adjacency. It asks whether E "
            f"is in the {actual} direction from S, regardless of distance or "
            f"walls between them.]\n\n"
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
                    f"[Maze step_extractor] Phase 1 trigger: \\n count={total_newlines} "
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

        last_marker_pos = text_after_think.rfind(self._structured_marker)
        if last_marker_pos >= 0:
            model_output_start = last_marker_pos + len(self._structured_marker)
            text_after_think = text_after_think[model_output_start:]
            text_start_offset = think_end_pos + model_output_start
        else:
            text_start_offset = think_end_pos

        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        text = text_after_think[last_feedback_end:]
        text_start = text_start_offset + last_feedback_end

        if self.question_type in ("right_turns", "total_turns"):
            step_pattern = re.compile(
                r'(>>>\s*STEP\s+(\d+):\s*Move\s+\w+\s+from\s+\([^)]+\)\s+to\s+\([^)]+\).*?'
                r'Running count:\s*Right\s*=\s*\d+\s*,\s*Left\s*=\s*\d+[^\n]*)',
                re.IGNORECASE | re.DOTALL
            )
            all_steps = list(step_pattern.finditer(text))

            if all_steps:
                last_step = all_steps[-1]
                text_after = text[last_step.end():]
                next_step = re.search(r'>>>\s*STEP\s+\d+', text_after, re.IGNORECASE)
                if not next_step:
                    end_pos = text_start + last_step.end()
                    return True, generated_text[:end_pos]
                return False, None

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

        boxed = find_complete_boxed(text)
        if boxed:
            end_pos = text_start + boxed.end()
            return True, generated_text[:end_pos]

        return False, None

    # ------------------------------------------------------------------
    #  verify
    # ------------------------------------------------------------------
    async def verify(self, step: str, token_index: int, event, event_info):
        # ==================================================================
        # CASE 1: Thinking phase – side-stream path verification
        # ==================================================================
        if self.answer_start_token not in step:
            total_dn = step.count('\n')
            logger.info(
                f"[Maze Phase 1] Thinking-phase verification triggered\n"
                f"  \\n count    : {total_dn}\n"
                f"  Thinking len : {len(step)} chars"
            )

            text_with_prompt = step + self._thinking_phase_prompt

            side_output = await self._side_stream_maze_steps(
                text_with_prompt, max_new_tokens=300
            )

            if not side_output or len(side_output.strip()) < 20:
                logger.info(
                    "[Maze Phase 1] Insufficient output from side-stream. "
                    "Letting model continue thinking."
                )
                return step, None

            full_side_text = (
                ">>> LOCATE START AND EXIT (0-indexed, top-left is (0,0)):\n" + side_output
            )

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
            recent_step_matches = list(step_pattern.finditer(recent_text))

            if recent_step_matches:
                last_match = recent_step_matches[-1]
                last_step_text = last_match.group(0)
                last_step_num = int(last_match.group(2))
                parsed = parse_maze_step(last_step_text)

                if parsed:
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

        last_by_num = {}
        for match in all_step_matches:
            step_num = int(match.group(2))
            if step_num >= target_step_num:
                continue
            last_by_num[step_num] = match

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
