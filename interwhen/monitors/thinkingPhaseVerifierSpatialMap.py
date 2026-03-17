"""
Thinking-phase verifier for SpatialMap tasks.

Verifies spatial-map directional claims by forking a side-stream during
the thinking phase.  Uses Z3 constraint solving to check whether
directional claims (e.g. "A is northeast of B") are consistent with
the stated problem constraints.

Workflow
--------
A) **DURING the thinking phase** (inside ``<think>...</think>``):
   After a warmup period, every *N* newlines in the thinking trace:
   1. Inject a first-person prompt to extract parsed and derived
      spatial relationships (STEP 1 pre-filled, STEP 2 generated).
   2. Parse directional claims from STEP 2 output.
   3. Verify each claim using a Z3 solver.
   4. If **errors** -> inject feedback into thinking trace.
   5. If **all valid** -> no feedback, keep thinking.

B) **AFTER ``</think>``**:
   Phase 2a: Inject structured step format template.
   Phase 2b: Verify directional claims and final answer
   (direction / object / counting questions) as model fills template.
   Once ``\\boxed{}`` appears, stop generation.
"""

import re
import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from copy import deepcopy

import httpx

from .base import VerifyMonitor
from ._common import find_complete_boxed
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
        "    - [Full Name A] is to the [direction] of [Full Name B]\n"
        "    - [Full Name C] is to the [direction] of [Full Name D]\n"
        "    [... list ALL given relationships using FULL names exactly as in the question ...]\n"
        "    (NO abbreviations, NO short forms, NO parenthetical aliases like 'Police Supply Store (PSS)')\n"
        "\n"
        ">>> STEP 2: ANALYZE SPATIAL RELATIONSHIPS\n"
        "    - Looking for: [target relationship / direction / count]\n"
        "    - [Full Name A] is to the [direction] of [Full Name B]\n"
        "    - [Full Name C] is to the [direction] of [Full Name D]\n"
        "    [... list each derived relationship as a structured claim using FULL names ...]\n"
        "    (Each claim MUST be in the form: '[Full Name] is to the [direction] of [Full Name]')\n"
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
        "spatial claim I have derived.\n"
        "IMPORTANT: I must use the FULL object names exactly as given in the question "
        "(no abbreviations, no short forms, no aliases, no partial names, no parenthetical aliases like 'Store (S)').\n"
        "Every claim must be in the form: '[Full Name] is to the [direction] of [Full Name]'\n"
        "For direction I will use the full word: northeast, northwest, southeast, southwest, north, south, east, or west.\n\n"
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
        "\nLet me solve this step by step using the structured format.\n"
        "IMPORTANT: I must use the FULL names of all objects exactly as they appear in the question. "
        "NO abbreviations, NO short forms, NO parenthetical aliases.\n\n"
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
        Every N newlines (after warmup), fork a side-stream that
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

        # Get full entity names from Z3 solver for abbreviation resolution
        entity_names = list({
            k[:-2] for k in self.z3_solver.entities if k.endswith('_x')
        })

        all_claims = extract_step2_claims(text_to_check, entity_names=entity_names)

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

            total_newlines = generated_text.count('\n')

            if total_newlines < self.warmup_newlines:
                return False, None

            past_warmup = total_newlines - self.warmup_newlines
            if (generated_text.endswith('\n')
                    and past_warmup >= 0
                    and past_warmup % self.newline_threshold == 0):
                logger.info(
                    f"[SpatialMap step_extractor] Phase 1 trigger: \\n count={total_newlines} "
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
        boxed_match = find_complete_boxed(text)
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
            total_dn = step.count('\n')
            logger.info(
                f"[SpatialMap Phase 1] Thinking-phase verification triggered\n"
                f"  \\n count    : {total_dn}\n"
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
            entity_names = list({
                k[:-2] for k in self.z3_solver.entities if k.endswith('_x')
            })
            claims = parse_directional_claims_from_text(
                side_output, entity_names=entity_names
            )

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
        boxed_match = find_complete_boxed(recent_text)
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
                            feedback = (
                                f"\n\n[VERIFIER FEEDBACK: Direction error!\n"
                                f"  '{model_dir_text.title()}' is "
                                f"impossible for "
                                f"{self._direction_question['entity_a']} "
                                f"relative to "
                                f"{self._direction_question['entity_b']} "
                                f"based on the given constraints.]\n\n"
                                f">>> STEP 3: ANSWER\n"
                            )
                        else:
                            feedback = (
                                f"\n\n[VERIFIER FEEDBACK: Direction error!\n"
                                f"  '{model_dir_text.title()}' is "
                                f"impossible for "
                                f"{self._direction_question['entity_a']} "
                                f"relative to "
                                f"{self._direction_question['entity_b']} "
                                f"based on the given constraints.\n"
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
                    # All spatial constraints in this dataset are diagonal
                    # (NE, NW, SE, SW), so no object can be strictly
                    # north/south/east/west of another. The answer is
                    # always 0.
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

                        # Build direction-specific examples of what does NOT count
                        if direction in ("north", "south"):
                            diag_examples = "northeast or northwest"
                        elif direction == "west":
                            diag_examples = "northwest or southwest"
                        else:  # east
                            diag_examples = "northeast or southeast"

                        feedback = (
                            f"\n\n[VERIFIER FEEDBACK: Count mismatch!\n"
                            f"  You answered {model_count} objects "
                            f"'{direction}' of {reference}, but this "
                            f"count is incorrect.\n"
                            f"  IMPORTANT: '{direction}' is a strict "
                            f"cardinal direction — it means ONLY "
                            f"exactly {direction}, NOT {diag_examples}."
                            f"\n"
                            f"  An object that is {diag_examples.split(' or ')[0]} of "
                            f"{reference} is NOT {direction} of "
                            f"{reference}.\n"
                            f"  Re-examine each object: is it described "
                            f"as being strictly '{direction} of' "
                            f"{reference}, or is the relationship "
                            f"actually a diagonal direction like "
                            f"{diag_examples}? Only count objects that "
                            f"are strictly {direction}.]\n\n"
                            f">>> STEP 3: ANSWER\n"
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
