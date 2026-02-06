import re
from typing import List, Tuple, Optional, Set, Dict
from .base import VerifyMonitor
from ..utils.game24_verifier import parse_step, verify_step, format_feedback
from ..utils.maze_verifier import (
    Direction, parse_direction, parse_maze_step, verify_maze_step,
    verify_locate_section, format_maze_feedback, format_locate_feedback,
    parse_maze_from_prompt
)
from ..utils.spatialmap_verifier import (
    SpatialMapZ3Solver, parse_directional_claims_from_text,
    extract_step2_claims, verify_spatialmap_step, format_spatialmap_feedback
)

    
class StepVerifierGame24Monitor(VerifyMonitor):
    """
    Step-by-step Game of 24 verifier monitor.
    
    Completely stateless - all state is derived from the generated_text (chunk) on each call.
    """
    
    def __init__(self, name, answer_start_token, original_numbers, max_corrections=5, async_execution=True):
        super().__init__(name)
        self.async_execution = async_execution
        self.answer_start_token = answer_start_token
        self.original_numbers = [float(x) for x in original_numbers]
        self.max_corrections = max_corrections

    def _count_feedback_blocks(self, text):
        """Count how many [VERIFIER FEEDBACK...] blocks are in the text."""
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))

    def _get_current_available(self, generated_text):
        """
        Reconstruct the current available numbers from the generated text.
        
        Stateless - parses the text to figure out what numbers should be available
        for the step we're about to verify.
        
        Logic:
        1. Find the step number of the last step (the one we're verifying)
        2. If it's Step 1, return original_numbers
        3. If it's Step N (N > 1), find the most recent VALID Step N-1 and return its remaining numbers
        
        The key insight is that after feedback, we need to look BACK through all sections
        to find the last valid step at (N-1) level.
        """
        if '</think>' not in generated_text:
            return self.original_numbers.copy()
        
        text_after_think = generated_text.split("</think>")[-1]
        
        # Pattern to find complete steps
        step_pattern = re.compile(
            r'>\s*Step\s*(\d+)\s*\n'
            r'available\s+numbers?\s*:\s*\[([^\]]+)\]\s*\n'
            r'suggested\s+operation\s*:\s*([^\n]+?)\s*\n'
            r'remaining\s+numbers?\s*:\s*\[([^\]]+)\]',
            re.IGNORECASE
        )
        
        # Split by feedback to get sections
        sections = re.split(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', text_after_think, flags=re.DOTALL)
        
        # Find the step we're verifying (last step in the last section)
        last_section = sections[-1]
        steps_in_last_section = list(step_pattern.finditer(last_section))
        
        if not steps_in_last_section:
            return self.original_numbers.copy()
        
        # Get the step number of the step we're verifying
        last_step = steps_in_last_section[-1]
        step_num_to_verify = int(last_step.group(1))
        
        # If verifying Step 1, always use original numbers
        if step_num_to_verify == 1:
            return self.original_numbers.copy()
        
        # For Step N (N > 1), we need the remaining numbers from the last valid Step N-1
        # This could be:
        # 1. In the same section (e.g., Step1 then Step2 in same section)
        # 2. In a previous section (e.g., Step1 passed, Step2 failed, now retrying Step2)
        
        target_step = step_num_to_verify - 1
        
        # First check in the last section (before the step we're verifying)
        for step_match in steps_in_last_section[:-1]:  # All steps except the last one
            if int(step_match.group(1)) == target_step:
                # Found the target step in the same section
                remaining_nums_str = step_match.group(4).strip()
                try:
                    return [float(x.strip()) for x in remaining_nums_str.split(',') if x.strip()]
                except:
                    pass
        
        # Look in previous sections (going backwards to find the most recent valid one)
        for section in reversed(sections[:-1]):  # All sections except the last
            steps_in_section = list(step_pattern.finditer(section))
            # Look for the target step number in this section
            for step_match in reversed(steps_in_section):
                if int(step_match.group(1)) == target_step:
                    # Found a Step N-1 - this is the valid one (since it wasn't rejected)
                    remaining_nums_str = step_match.group(4).strip()
                    try:
                        return [float(x.strip()) for x in remaining_nums_str.split(',') if x.strip()]
                    except:
                        pass
        
        # Fallback to original numbers
        return self.original_numbers.copy()

    def _extract_last_step_info(self, generated_text):
        """
        Extract the last complete step's info from the generated text.
        
        Returns:
            (step_num, parsed_step) or (None, None) if no step found
        """
        if '</think>' not in generated_text:
            return None, None
        
        text_after_think = generated_text.split("</think>")[-1]
        
        # Get text after last feedback
        sections = re.split(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', text_after_think, flags=re.DOTALL)
        text = sections[-1]
        
        # Find complete steps
        step_pattern = re.compile(
            r'(>\s*Step\s*(\d+)\s*\n'
            r'available\s+numbers?\s*:\s*\[([^\]]+)\]\s*\n'
            r'suggested\s+operation\s*:\s*([^\n]+?)\s*\n'
            r'remaining\s+numbers?\s*:\s*\[([^\]]+)\])',
            re.IGNORECASE
        )
        
        all_steps = list(step_pattern.finditer(text))
        
        if not all_steps:
            return None, None
        
        last_step = all_steps[-1]
        step_num = int(last_step.group(2))
        
        # Reconstruct step text for parse_step
        step_text = (
            f">Step{step_num}\n"
            f"available numbers: [{last_step.group(3).strip()}]\n"
            f"suggested operation: {last_step.group(4).strip()}\n"
            f"remaining numbers: [{last_step.group(5).strip()}]"
        )
        
        parsed = parse_step(step_text)
        return step_num, parsed

    async def verify(self, chunk, token_index, event, event_info):
        """
        Verify the latest step in the generated text.
        
        Args:
            chunk: the step extracted text
            token_index: Current token index (not used)
            event: asyncio.Event to signal when verification fails
            event_info: Dict to store feedback info
            
        Returns:
            (chunk, feedback) - feedback is None if valid, or str if invalid
        """
    
        # Check if max corrections reached
        num_corrections = self._count_feedback_blocks(chunk)
        if num_corrections >= self.max_corrections:
            max_feedback = "\nthe answer is \\boxed{no solution}"
            if not event.is_set():
                event_info["generated_text"] = chunk
                event_info["feedback"] = max_feedback
                event_info["correction_index"] = token_index
                event_info["errors"] = ["Max corrections reached"]
                event_info["failed_step"] = None
                event.set()
            
            return chunk, max_feedback
        
        # Extract the last step info
        step_num, parsed = self._extract_last_step_info(chunk)
        
        if step_num is None or parsed is None or parsed.get('available_numbers') is None:
            return chunk, None
        
        # Get the current available numbers for this step
        current_available = self._get_current_available(chunk)
        
        # Verify the step
        is_valid, errors, new_available = verify_step(
            parsed,
            current_available,
            self.original_numbers,
            step_num
        )
        
        if is_valid:
            return chunk, None
        
        # Step has errors - generate feedback
        feedback = format_feedback(errors, step_num, current_available)
        
        if not event.is_set():
            event_info["generated_text"] = chunk
            event_info["feedback"] = feedback
            event_info["correction_index"] = token_index
            event_info["errors"] = errors
            event_info["failed_step"] = step_num
            event.set()
        
        return chunk, feedback

    async def fix(self, generated_text, event_info, fix_method=None):
        """Append feedback to the generated text."""
        return event_info["generated_text"] + event_info["feedback"]


    def step_extractor(self, chunk, generated_text):
        # 1. Return False if </think> not present
        if '</think>' not in generated_text:
            return False, None

        # 2. Find where </think> ends
        think_end_pos = generated_text.find("</think>") + len("</think>")
        text_after_think = generated_text[think_end_pos:]
        
        # 3. Skip past any previous [VERIFIER FEEDBACK...] blocks
        # Find the position after the last feedback block
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        
        text = text_after_think[last_feedback_end:]
        # Position in generated_text where 'text' starts
        text_start_in_generated = think_end_pos + last_feedback_end

        # 4. Find complete steps using regex
        step_pattern = re.compile(
            r'(>\s*Step\s*(\d+)\s*\n'                              # >Step1 (group 1=full, 2=num)
            r'available\s+numbers?\s*:\s*\[([^\]]+)\]\s*\n'       # available numbers: [...]  (group 3)
            r'suggested\s+operation\s*:\s*([^\n]+?)\s*\n'         # suggested operation: ...  (group 4)
            r'remaining\s+numbers?\s*:\s*\[([^\]]+)\])',          # remaining numbers: [...]  (group 5)
            re.IGNORECASE
        )
        
        all_steps = list(step_pattern.finditer(text))
        
        if not all_steps:
            return False, None
        
        # Get the last complete step
        last_complete_step = all_steps[-1]
        
        # 5. Check if the next step has started after this complete step
        text_after_last_step = text[last_complete_step.end():]
        next_step_header = re.search(r'>\s*Step\s*\d+', text_after_last_step, re.IGNORECASE)
        
        if next_step_header:
            # Next step has started - current step was already verified, skip
            return False, None
        
        # 6. Return generated_text sliced up to end of the complete step
        end_pos = text_start_in_generated + last_complete_step.end()
        text_slice = generated_text[:end_pos]
        
        return True, text_slice


class StepVerifierMazeMonitor(VerifyMonitor):
    """
    Step-by-step Maze verifier monitor.
    
    Completely stateless - all state is derived from the generated_text (chunk) on each call.
    Verifies:
    1. Move direction validity (from_pos to to_pos matches claimed direction)
    2. from_pos matches expected position (from previous step or start)
    3. to_pos is walkable (not a wall)
    4. Turn type is correct based on direction change
    5. Running counts (Right, Left, Total) are correct
    """
    
    def __init__(
        self,
        name: str,
        answer_start_token: str,
        grid: List[List[str]],
        start_pos: Tuple[int, int],
        exit_pos: Tuple[int, int],
        max_corrections: int = 5,
        question_type: str = "right_turns",  # "right_turns", "total_turns", "relative_position"
        async_execution: bool = True
    ):
        super().__init__(name)
        self.async_execution = async_execution
        self.answer_start_token = answer_start_token
        self.grid = grid
        self.start_pos = start_pos
        self.exit_pos = exit_pos
        self.max_corrections = max_corrections
        self.question_type = question_type

    @staticmethod
    def detect_question_type(prompt: str) -> str:
        """
        Auto-detect the question type from the prompt text.
        
        Rules:
        - If contains "right turn" → "right_turns"
        - If contains "left turn" → "total_turns" (left turns still need step tracking)
        - If contains "total" AND "turn" → "total_turns"
        - If contains "turn" (generic) → "right_turns" (default for turn questions)
        - Otherwise → "relative_position"
        
        Returns:
            One of: "right_turns", "total_turns", "relative_position"
        """
        prompt_lower = prompt.lower()
        
        # Check for specific turn types first
        if "right turn" in prompt_lower or "right-turn" in prompt_lower:
            return "right_turns"
        if "left turn" in prompt_lower or "left-turn" in prompt_lower:
            return "total_turns"  # Left turn questions also need step-by-step tracking
        
        # Check for "total" + "turn" (even if not adjacent)
        if "total" in prompt_lower and "turn" in prompt_lower:
            return "total_turns"
        
        # Generic "turn" mention - default to right_turns
        if "turn" in prompt_lower:
            return "right_turns"
        
        # No turn keywords found - assume relative position question
        return "relative_position"

    @classmethod
    def from_prompt(
        cls,
        name: str,
        answer_start_token: str,
        prompt: str,
        max_corrections: int = 5,
        question_type: str = None,  # Auto-detect if None
        async_execution: bool = True
    ):
        """
        Create a StepVerifierMazeMonitor by parsing the maze from a prompt.
        
        Args:
            name: Monitor name
            answer_start_token: Token that marks start of answer section
            prompt: The full prompt containing the maze
            max_corrections: Maximum number of feedbacks before giving up
            question_type: Type of question. If None, auto-detects from prompt.
            async_execution: Whether to run asynchronously
        """
        grid, start_pos, exit_pos = parse_maze_from_prompt(prompt)
        
        # Auto-detect question type if not provided
        if question_type is None:
            question_type = cls.detect_question_type(prompt)
        
        return cls(
            name=name,
            answer_start_token=answer_start_token,
            grid=grid,
            start_pos=start_pos,
            exit_pos=exit_pos,
            max_corrections=max_corrections,
            question_type=question_type,
            async_execution=async_execution
        )

    def _count_feedback_blocks(self, text: str) -> int:
        """Count how many [VERIFIER FEEDBACK...] blocks are in the text."""
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))

    def _get_state_before_step(self, generated_text: str, target_step_num: int) -> dict:
        """
        Reconstruct the state (position, direction, counts) before a given step.
        
        Stateless - parses the text to figure out what state we should be in
        before the step we're verifying.
        
        Returns:
            dict with keys: position, direction, right_count, left_count, total_count
        """
        state = {
            'position': self.start_pos,
            'direction': Direction.NONE,
            'right_count': 0,
            'left_count': 0,
            'total_count': 0,
        }
        
        if '</think>' not in generated_text:
            return state
        
        text_after_think = generated_text.split("</think>")[-1]
        
        # Split by feedback to get sections
        sections = re.split(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', text_after_think, flags=re.DOTALL)
        
        # We need to find all valid steps with step_num < target_step_num
        # A valid step is one that wasn't followed by feedback (or was in a previous section)
        
        step_pattern = re.compile(
            r'>>>\s*STEP\s+(\d+):\s*Move\s+(\w+)\s+from\s+\((\d+)\s*,\s*(\d+)\)\s+to\s+\((\d+)\s*,\s*(\d+)\)',
            re.IGNORECASE
        )
        
        # Process each section to build up state
        for section_idx, section in enumerate(sections):
            is_last_section = (section_idx == len(sections) - 1)
            steps_in_section = list(step_pattern.finditer(section))
            
            for i, step_match in enumerate(steps_in_section):
                step_num = int(step_match.group(1))
                
                # Only process steps before our target
                if step_num >= target_step_num:
                    continue
                
                # In the last section, all steps before target are valid
                # In previous sections, only steps that weren't invalidated are valid
                # (the last step of a non-last section might have been invalid if feedback followed)
                if not is_last_section and i == len(steps_in_section) - 1:
                    # This is the last step before feedback - it was invalid, skip it
                    continue
                
                # Parse and apply this valid step
                direction = parse_direction(step_match.group(2))
                to_pos = (int(step_match.group(5)), int(step_match.group(6)))
                
                # Calculate turn type
                from ..verifiers.maze_verifier import get_expected_turn_type
                turn_type = get_expected_turn_type(state['direction'], direction)
                
                # Update counts based on turn
                if turn_type == 'RIGHT_TURN':
                    state['right_count'] += 1
                    state['total_count'] += 1
                elif turn_type == 'LEFT_TURN':
                    state['left_count'] += 1
                    state['total_count'] += 1
                
                # Update position and direction
                state['position'] = to_pos
                state['direction'] = direction
        
        return state

    def _extract_last_step_info(self, generated_text: str) -> Tuple[Optional[int], Optional[dict], Optional[str]]:
        """
        Extract the last complete step's info from the generated text.
        
        Returns:
            (step_num, parsed_step, step_text) or (None, None, None) if no step found
        """
        if '</think>' not in generated_text:
            return None, None, None
        
        text_after_think = generated_text.split("</think>")[-1]
        
        # Get text after last feedback
        sections = re.split(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', text_after_think, flags=re.DOTALL)
        text = sections[-1]
        
        # Find complete steps (must have at least the header line and running count)
        # Full pattern for a complete step
        step_pattern = re.compile(
            r'(>>>\s*STEP\s+(\d+):\s*Move\s+\w+\s+from\s+\([^)]+\)\s+to\s+\([^)]+\).*?'
            r'Running count:[^\n]+)',
            re.IGNORECASE | re.DOTALL
        )
        
        all_steps = list(step_pattern.finditer(text))
        
        if not all_steps:
            return None, None, None
        
        last_step = all_steps[-1]
        step_text = last_step.group(1)
        step_num = int(last_step.group(2))
        
        # Parse the step
        parsed = parse_maze_step(step_text)
        
        return step_num, parsed, step_text

    def _check_locate_section(self, generated_text: str) -> Tuple[bool, List[str], bool]:
        """
        Check if LOCATE section exists and is correct.
        
        Returns:
            (is_valid, errors, section_found)
        """
        if '</think>' not in generated_text:
            return True, [], False
        
        text_after_think = generated_text.split("</think>")[-1]
        
        # Check if LOCATE section is present
        locate_match = re.search(r'LOCATE START AND EXIT', text_after_think, re.IGNORECASE)
        if not locate_match:
            return True, [], False
        
        # Get the section text
        section_end = re.search(r'>>>\s*STEP\s+1', text_after_think, re.IGNORECASE)
        if section_end:
            locate_text = text_after_think[locate_match.start():section_end.start()]
        else:
            locate_text = text_after_think[locate_match.start():]
        
        is_valid, errors = verify_locate_section(locate_text, self.start_pos, self.exit_pos)
        return is_valid, errors, True

    async def verify(self, chunk: str, token_index: int, event, event_info: dict):
        """
        Verify the latest step in the generated text.
        
        Args:
            chunk: the step extracted text
            token_index: Current token index (not used)
            event: asyncio.Event to signal when verification fails
            event_info: Dict to store feedback info
            
        Returns:
            (chunk, feedback) - feedback is None if valid, or str if invalid
        """
        
        # Check if max corrections reached
        num_corrections = self._count_feedback_blocks(chunk)
        if num_corrections >= self.max_corrections:
            max_feedback = "\nthe answer is \\boxed{no solution}"
            if not event.is_set():
                event_info["generated_text"] = chunk
                event_info["feedback"] = max_feedback
                event_info["correction_index"] = token_index
                event_info["errors"] = ["Max corrections reached"]
                event_info["failed_step"] = None
                event.set()
            
            return chunk, max_feedback
        
        # For relative_position questions, use different verification
        if self.question_type == "relative_position":
            return await self._verify_relative_position(chunk, token_index, event, event_info)
        
        # For turn-counting questions (right_turns, total_turns), verify steps
        # Extract the last step info
        step_num, parsed, step_text = self._extract_last_step_info(chunk)
        
        if step_num is None or parsed is None:
            # No complete step found - check LOCATE section instead
            locate_valid, locate_errors, locate_found = self._check_locate_section(chunk)
            if locate_found and not locate_valid:
                feedback = format_locate_feedback(locate_errors)
                if not event.is_set():
                    event_info["generated_text"] = chunk
                    event_info["feedback"] = feedback
                    event_info["correction_index"] = token_index
                    event_info["errors"] = locate_errors
                    event_info["failed_step"] = 0  # LOCATE section
                    event.set()
                return chunk, feedback
            return chunk, None
        
        # Get the state before this step
        state = self._get_state_before_step(chunk, step_num)
        
        # Verify the step
        is_valid, errors, new_state = verify_maze_step(
            step=parsed,
            grid=self.grid,
            expected_from_pos=state['position'],
            prev_direction=state['direction'],
            expected_right_count=state['right_count'],
            expected_left_count=state['left_count'],
            expected_total_count=state['total_count']
        )
        
        if is_valid:
            return chunk, None
        
        # Step has errors - generate feedback
        feedback = format_maze_feedback(errors, step_num)
        
        if not event.is_set():
            event_info["generated_text"] = chunk
            event_info["feedback"] = feedback
            event_info["correction_index"] = token_index
            event_info["errors"] = errors
            event_info["failed_step"] = step_num
            event.set()
        
        return chunk, feedback

    async def _verify_relative_position(self, chunk: str, token_index: int, event, event_info: dict):
        """
        Verify relative position answer.
        
        For relative_position questions (Yes/No format), we only verify:
        1. The LOCATE section (S and E positions are correctly identified)
        
        We do NOT verify the final Yes/No answer because:
        - The question asks "Is E to the [direction] of S?" 
        - We don't have the question text here to know what direction was asked
        - The comparison logic (row/col arithmetic) is straightforward
        - If LOCATE is correct, the model should get the answer right
        """
        # Check LOCATE section for correct S and E positions
        locate_valid, locate_errors, locate_found = self._check_locate_section(chunk)
        if locate_found and not locate_valid:
            feedback = format_locate_feedback(locate_errors)
            if not event.is_set():
                event_info["generated_text"] = chunk
                event_info["feedback"] = feedback
                event_info["correction_index"] = token_index
                event_info["errors"] = locate_errors
                event_info["failed_step"] = 0
                event.set()
            return chunk, feedback
        
        # For relative_position, we don't verify the final Yes/No answer
        # Just let it complete once LOCATE is verified
        return chunk, None

    async def fix(self, generated_text: str, event_info: dict, fix_method=None) -> str:
        """Append feedback to the generated text."""
        return event_info["generated_text"] + event_info["feedback"]

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a complete maze step has been generated.
        
        For turn-counting questions (right_turns, total_turns):
        - >>> STEP N: Move DIRECTION from (r1,c1) to (r2,c2)
        - Current position: (r,c)
        - Previous direction: ...
        - Current direction: ...
        - Turn type: ...
        - Running count: Right = X, Left = Y, Total = Z
        
        For relative_position questions:
        - LOCATE section OR
        - \\boxed{} answer OR
        - >>> ANSWER: line
        
        Returns:
            (found_complete_step, text_slice) - text_slice includes all text up to step end
        """
        # 1. Return False if </think> not present
        if '</think>' not in generated_text:
            return False, None

        # 2. Find where </think> ends
        think_end_pos = generated_text.find("</think>") + len("</think>")
        text_after_think = generated_text[think_end_pos:]
        
        # 3. Skip past any previous [VERIFIER FEEDBACK...] blocks
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        
        text = text_after_think[last_feedback_end:]
        text_start_in_generated = think_end_pos + last_feedback_end

        # For relative_position questions, different extraction logic
        if self.question_type == "relative_position":
            return self._step_extractor_relative_position(text, text_start_in_generated, generated_text)
        
        # For turn-counting questions (right_turns, total_turns)
        # 4. Check for LOCATE section first (if no steps yet)
        locate_pattern = re.compile(
            r'(LOCATE START AND EXIT.*?E position:\s*\([^)]+\))',
            re.IGNORECASE | re.DOTALL
        )
        
        # 5. Find complete steps using regex
        # Pattern for complete step (with Running count line)
        step_pattern = re.compile(
            r'(>>>\s*STEP\s+(\d+):\s*Move\s+\w+\s+from\s+\([^)]+\)\s+to\s+\([^)]+\).*?'
            r'Running count:[^\n]+)',
            re.IGNORECASE | re.DOTALL
        )
        
        all_steps = list(step_pattern.finditer(text))
        
        # Check for LOCATE section if no steps yet
        if not all_steps:
            locate_match = locate_pattern.search(text)
            if locate_match:
                # Check if Step 1 has started after LOCATE
                step1_start = re.search(r'>>>\s*STEP\s+1', text[locate_match.end():], re.IGNORECASE)
                if step1_start:
                    # Step 1 started - LOCATE section is complete, verify it
                    end_pos = text_start_in_generated + locate_match.end()
                    return True, generated_text[:end_pos]
            return False, None
        
        # Get the last complete step
        last_complete_step = all_steps[-1]
        
        # 6. Check if the next step has started after this complete step
        text_after_last_step = text[last_complete_step.end():]
        next_step_header = re.search(r'>>>\s*STEP\s+\d+', text_after_last_step, re.IGNORECASE)
        
        if next_step_header:
            # Next step has started - current step was already verified, skip
            return False, None
        
        # 7. Return generated_text sliced up to end of the complete step
        end_pos = text_start_in_generated + last_complete_step.end()
        text_slice = generated_text[:end_pos]
        
        return True, text_slice

    def _step_extractor_relative_position(
        self,
        text: str,
        text_start_in_generated: int,
        generated_text: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Step extractor for relative_position questions.
        
        Triggers verification when:
        1. A boxed answer is found (highest priority - verify final answer), OR
        2. >>> ANSWER: line is found, OR
        3. LOCATE section is complete and analysis has started (verify LOCATE)
        """
        # Check for boxed answer first (highest priority)
        boxed_match = re.search(r'\\boxed\{[^}]+\}', text)
        if boxed_match:
            # Found answer, verify it (include full text up to boxed answer)
            end_pos = text_start_in_generated + boxed_match.end()
            return True, generated_text[:end_pos]
        
        # Check for >>> ANSWER: line
        answer_match = re.search(r'>>>\s*ANSWER:\s*\w+', text, re.IGNORECASE)
        if answer_match:
            end_pos = text_start_in_generated + answer_match.end()
            return True, generated_text[:end_pos]
        
        # Check for LOCATE section (only if no answer yet)
        locate_pattern = re.compile(
            r'(LOCATE START AND EXIT.*?E position:\s*\([^)]+\))',
            re.IGNORECASE | re.DOTALL
        )
        locate_match = locate_pattern.search(text)
        
        if locate_match:
            # Check if analysis/comparison has started after LOCATE
            text_after_locate = text[locate_match.end():]
            analysis_started = re.search(
                r'>>>\s*(COMPARE|ANALYSIS)',
                text_after_locate,
                re.IGNORECASE
            )
            if analysis_started:
                # LOCATE is complete and analysis started, verify LOCATE section
                end_pos = text_start_in_generated + locate_match.end()
                return True, generated_text[:end_pos]
        
        return False, None


class StepVerifierSpatialMapMonitor(VerifyMonitor):
    """
    Step-by-step SpatialMap verifier monitor using Z3 constraint solving.
    
    Verifies directional claims made by the model against Z3 constraints.
    Claims are checked as they appear in STEP 2 of the model's reasoning.
    """
    
    def __init__(
        self,
        name: str,
        answer_start_token: str,
        problem_text: str,
        max_corrections: int = 5,
        async_execution: bool = True
    ):
        """
        Initialize the SpatialMap step verifier monitor.
        
        Args:
            name: Monitor name
            answer_start_token: Token that starts the answer section (e.g., "</think>")
            problem_text: The problem description text for Z3 parsing
            max_corrections: Maximum number of correction attempts
            async_execution: Whether to run verification asynchronously
        """
        super().__init__(name)
        self.async_execution = async_execution
        self.answer_start_token = answer_start_token
        self.problem_text = problem_text
        self.max_corrections = max_corrections
        
        # Initialize Z3 solver with problem constraints
        self.z3_solver = SpatialMapZ3Solver(problem_text)
        
        # Track verified claims to avoid re-checking
        self.verified_claims: Set[Tuple[str, str, str]] = set()
    
    @classmethod
    def from_prompt(
        cls,
        problem_text: str,
        max_corrections: int = 5,
        name: str = "spatialmap_step_verifier"
    ) -> "StepVerifierSpatialMapMonitor":
        """
        Create a SpatialMap monitor from just the problem text.
        
        Args:
            problem_text: The problem description containing spatial constraints
            max_corrections: Maximum correction attempts
            name: Monitor name
        
        Returns:
            StepVerifierSpatialMapMonitor instance
        """
        return cls(
            name=name,
            answer_start_token="</think>",
            problem_text=problem_text,
            max_corrections=max_corrections
        )
    
    def _count_feedback_blocks(self, text: str) -> int:
        """Count how many [VERIFIER FEEDBACK...] blocks are in the text."""
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))
    
    def _extract_new_claims(self, chunk: str) -> List[Dict]:
        """
        Extract new (not yet verified) directional claims from the text.
        
        Only extracts claims from the MOST RECENT STEP 2 section (after any feedback).
        """
        if '</think>' not in chunk:
            return []
        
        text_after_think = chunk.split("</think>")[-1]
        
        # Skip past any previous [VERIFIER FEEDBACK...] blocks to get only the latest attempt
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        
        # Only look at text after the last feedback
        text_to_check = text_after_think[last_feedback_end:]
        
        # Extract claims from STEP 2 in the latest attempt only
        all_claims = extract_step2_claims(text_to_check)
        
        # Filter to only new claims (not yet verified)
        new_claims = []
        for claim in all_claims:
            claim_key = (claim['A'], claim['direction'], claim['B'])
            if claim_key not in self.verified_claims:
                new_claims.append(claim)
        
        return new_claims

    async def verify(self, chunk: str, token_index: int, event, event_info: dict):
        """
        Verify directional claims in the generated text against Z3 constraints.
        
        Args:
            chunk: The step extracted text
            token_index: Current token index
            event: asyncio.Event to signal when verification fails
            event_info: Dict to store feedback info
            
        Returns:
            (chunk, feedback) - feedback is None if valid, or str if invalid
        """
        # Check if max corrections reached
        num_corrections = self._count_feedback_blocks(chunk)
        if num_corrections >= self.max_corrections:
            max_feedback = "\nthe answer is \\boxed{no solution}"
            if not event.is_set():
                event_info["generated_text"] = chunk
                event_info["feedback"] = max_feedback
                event_info["correction_index"] = token_index
                event_info["errors"] = ["Max corrections reached"]
                event_info["failed_step"] = None
                event.set()
            return chunk, max_feedback
        
        # Extract new claims to verify
        new_claims = self._extract_new_claims(chunk)
        
        for claim in new_claims:
            claim_key = (claim['A'], claim['direction'], claim['B'])
            
            # Verify the claim
            is_valid, errors = verify_spatialmap_step(
                claim=claim,
                z3_solver=self.z3_solver,
                add_if_valid=True  # Add valid claims to solver for future checks
            )
            
            # Mark as verified (whether valid or not)
            self.verified_claims.add(claim_key)
            
            if not is_valid:
                # Contradiction found - generate feedback
                feedback = format_spatialmap_feedback(errors, claim)
                
                if not event.is_set():
                    event_info["generated_text"] = chunk
                    event_info["feedback"] = feedback
                    event_info["correction_index"] = token_index
                    event_info["errors"] = errors
                    event_info["failed_step"] = claim
                    event.set()
                
                return chunk, feedback
        
        # All claims valid
        return chunk, None

    async def fix(self, generated_text: str, event_info: dict, fix_method=None) -> str:
        """Append feedback to the generated text."""
        return event_info["generated_text"] + event_info["feedback"]

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if new directional claims have been generated in STEP 2.
        
        For SpatialMap, we trigger verification when:
        1. STEP 2 section exists and has new claims, OR
        2. A boxed answer is found (verify any remaining claims)
        
        Returns:
            (found_complete_step, text_slice) - text_slice includes all text up to verification point
        """
        # Must have answer section started
        if '</think>' not in generated_text:
            return False, None
        
        # Find where </think> ends
        think_end_pos = generated_text.find("</think>") + len("</think>")
        text_after_think = generated_text[think_end_pos:]
        
        # Skip past any previous [VERIFIER FEEDBACK...] blocks
        feedback_pattern = re.compile(r'\[VERIFIER FEEDBACK[^\]]*\]\s*', re.DOTALL)
        last_feedback_end = 0
        for match in feedback_pattern.finditer(text_after_think):
            last_feedback_end = match.end()
        
        text = text_after_think[last_feedback_end:]
        text_start_in_generated = think_end_pos + last_feedback_end
        
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
                # STEP 2 is complete, extract claims for verification
                new_claims = self._extract_new_claims(generated_text)
                if new_claims:
                    # Found new claims to verify
                    end_pos = text_start_in_generated + step2_match.end()
                    return True, generated_text[:end_pos]
        
        # Check for boxed answer (trigger final verification)
        boxed_match = re.search(r'\\boxed\{[^}]+\}', text)
        if boxed_match:
            # Verify any remaining claims before final answer
            new_claims = self._extract_new_claims(generated_text)
            if new_claims:
                end_pos = text_start_in_generated + boxed_match.end()
                return True, generated_text[:end_pos]
        
        return False, None