"""
Maze Step Verifier

Verifies each step of a Maze trace solution:
1. Check if the move direction is valid (UP, DOWN, LEFT, RIGHT)
2. Check if the from_pos matches expected position
3. Check if the to_pos is walkable (not a wall)
4. Check if the turn type is correct based on direction change
5. Check if running counts (Right, Left, Total) are correct
"""

import re
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class Direction(Enum):
    NONE = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


# Turn rules - which direction changes constitute right/left turns
RIGHT_TURNS = {
    (Direction.UP, Direction.RIGHT),
    (Direction.RIGHT, Direction.DOWN),
    (Direction.DOWN, Direction.LEFT),
    (Direction.LEFT, Direction.UP),
}

LEFT_TURNS = {
    (Direction.UP, Direction.LEFT),
    (Direction.LEFT, Direction.DOWN),
    (Direction.DOWN, Direction.RIGHT),
    (Direction.RIGHT, Direction.UP),
}

# Movement deltas for each direction
DIRECTION_DELTAS = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


def parse_direction(dir_str: str) -> Direction:
    """Parse direction string to Direction enum."""
    dir_str = dir_str.strip().upper()
    if dir_str in ['—', '-', 'NONE', '']:
        return Direction.NONE
    try:
        return Direction[dir_str]
    except KeyError:
        return Direction.NONE


def get_expected_turn_type(prev_dir: Direction, curr_dir: Direction) -> str:
    """Get expected turn type based on direction change."""
    if prev_dir == Direction.NONE or prev_dir == curr_dir:
        return 'STRAIGHT'
    if (prev_dir, curr_dir) in RIGHT_TURNS:
        return 'RIGHT_TURN'
    if (prev_dir, curr_dir) in LEFT_TURNS:
        return 'LEFT_TURN'
    return 'UNKNOWN'


def parse_maze_from_prompt(prompt: str) -> Tuple[List[List[str]], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Parse maze from prompt. Returns (grid, start_pos, exit_pos).
    Finds the LAST maze in the prompt (the actual one being solved).
    """
    lines = prompt.split('\n')
    all_mazes = []
    current_maze = []
    in_maze = False
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') and all(c in '#XSEX ' for c in stripped):
            in_maze = True
            current_maze.append(stripped)
        elif in_maze:
            if current_maze:
                all_mazes.append(current_maze)
            current_maze = []
            in_maze = False
    
    if current_maze:
        all_mazes.append(current_maze)
    
    if not all_mazes:
        return [], None, None
    
    maze_lines = all_mazes[-1]
    grid = [list(row) for row in maze_lines]
    start_pos = None
    exit_pos = None
    
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell == 'S':
                start_pos = (r, c)
            elif cell == 'E':
                exit_pos = (r, c)
    
    return grid, start_pos, exit_pos


def parse_maze_step(step_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single maze step from the LLM output.
    
    Expected format:
    >>> STEP N: Move DIRECTION from (r1, c1) to (r2, c2)
    Current position: (r2, c2)
    Previous direction: DIRECTION
    Current direction: DIRECTION
    Turn type: STRAIGHT/RIGHT_TURN/LEFT_TURN
    Running count: Right = X, Left = Y, Total = Z
    
    Returns a dict with parsed values or None if parsing fails.
    """
    result = {}
    
    # Extract step number and move info
    step_match = re.search(
        r'>>>\s*STEP\s+(\d+):\s*Move\s+(\w+)\s+from\s+\((\d+)\s*,\s*(\d+)\)\s+to\s+\((\d+)\s*,\s*(\d+)\)',
        step_text,
        re.IGNORECASE
    )
    
    if not step_match:
        return None
    
    result['step_num'] = int(step_match.group(1))
    result['direction'] = parse_direction(step_match.group(2))
    result['from_pos'] = (int(step_match.group(3)), int(step_match.group(4)))
    result['to_pos'] = (int(step_match.group(5)), int(step_match.group(6)))
    
    # Extract current position
    curr_pos_match = re.search(r'Current position:\s*\((\d+)\s*,\s*(\d+)\)', step_text)
    if curr_pos_match:
        result['claimed_current_pos'] = (int(curr_pos_match.group(1)), int(curr_pos_match.group(2)))
    else:
        result['claimed_current_pos'] = None
    
    # Extract previous direction
    prev_dir_match = re.search(r'Previous direction:\s*(\S+)', step_text)
    if prev_dir_match:
        result['claimed_prev_dir'] = prev_dir_match.group(1)
    else:
        result['claimed_prev_dir'] = None
    
    # Extract current direction
    curr_dir_match = re.search(r'Current direction:\s*(\S+)', step_text)
    if curr_dir_match:
        result['claimed_curr_dir'] = curr_dir_match.group(1)
    else:
        result['claimed_curr_dir'] = None
    
    # Extract turn type
    turn_match = re.search(r'Turn type:\s*(\S+)', step_text)
    if turn_match:
        result['claimed_turn'] = turn_match.group(1).upper()
    else:
        result['claimed_turn'] = None
    
    # Extract running counts
    count_match = re.search(
        r'Running count:\s*Right\s*=\s*(\d+)\s*,\s*Left\s*=\s*(\d+)(?:\s*,\s*Total\s*=\s*(\d+))?',
        step_text,
        re.IGNORECASE
    )
    if count_match:
        result['claimed_right'] = int(count_match.group(1))
        result['claimed_left'] = int(count_match.group(2))
        result['claimed_total'] = int(count_match.group(3)) if count_match.group(3) else None
    else:
        result['claimed_right'] = None
        result['claimed_left'] = None
        result['claimed_total'] = None
    
    return result


def verify_maze_step(
    step: Dict[str, Any],
    grid: List[List[str]],
    expected_from_pos: Tuple[int, int],
    prev_direction: Direction,
    expected_right_count: int,
    expected_left_count: int,
    expected_total_count: int
) -> Tuple[bool, List[str], Dict]:
    """
    Verify a single maze step.
    
    Args:
        step: Parsed step dictionary
        grid: The maze grid
        expected_from_pos: Expected starting position for this step
        prev_direction: Previous movement direction
        expected_right_count: Expected right turn count before this step
        expected_left_count: Expected left turn count before this step
        expected_total_count: Expected total turn count before this step
    
    Returns:
        (is_valid, errors, state_updates)
        state_updates contains: new_pos, new_direction, new_right, new_left, new_total
    """
    errors = []
    state = {
        'new_pos': expected_from_pos,
        'new_direction': prev_direction,
        'new_right': expected_right_count,
        'new_left': expected_left_count,
        'new_total': expected_total_count,
    }
    
    direction = step['direction']
    from_pos = step['from_pos']
    to_pos = step['to_pos']
    claimed_current = step['claimed_current_pos']
    claimed_turn = step['claimed_turn']
    claimed_right = step['claimed_right']
    claimed_left = step['claimed_left']
    claimed_total = step['claimed_total']
    
    # 1. Verify from_pos matches expected
    if from_pos != expected_from_pos:
        errors.append(f"from_pos {from_pos} should be {expected_from_pos}")
    
    # 2. Verify direction delta
    expected_delta = DIRECTION_DELTAS.get(direction)
    if expected_delta:
        actual_delta = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        if actual_delta != expected_delta:
            errors.append(f"Move {direction.name} doesn't match delta {actual_delta}, expected {expected_delta}")
    
    # 3. Verify to_pos is walkable (not a wall)
    if 0 <= to_pos[0] < len(grid) and 0 <= to_pos[1] < len(grid[0]):
        cell = grid[to_pos[0]][to_pos[1]]
        if cell == '#':
            errors.append(f"to_pos {to_pos} is a wall ('#')")
    else:
        errors.append(f"to_pos {to_pos} is out of bounds")
    
    # 4. Verify current_pos matches to_pos
    if claimed_current is not None and claimed_current != to_pos:
        errors.append(f"Current position {claimed_current} should be {to_pos}")
    
    # 5. Verify turn type
    expected_turn = get_expected_turn_type(prev_direction, direction)
    if claimed_turn is not None and claimed_turn != expected_turn:
        errors.append(f"Turn type {claimed_turn} should be {expected_turn} (prev={prev_direction.name}, curr={direction.name})")
    
    # 6. Calculate expected counts after this step
    new_right = expected_right_count
    new_left = expected_left_count
    new_total = expected_total_count
    
    if expected_turn == 'RIGHT_TURN':
        new_right += 1
        new_total += 1
    elif expected_turn == 'LEFT_TURN':
        new_left += 1
        new_total += 1
    
    # 7. Verify running counts
    if claimed_right is not None and claimed_right != new_right:
        errors.append(f"Right count {claimed_right} should be {new_right}")
    if claimed_left is not None and claimed_left != new_left:
        errors.append(f"Left count {claimed_left} should be {new_left}")
    if claimed_total is not None and claimed_total != new_total:
        errors.append(f"Total count {claimed_total} should be {new_total}")
    
    # Update state for next step
    state['new_pos'] = to_pos
    state['new_direction'] = direction
    state['new_right'] = new_right
    state['new_left'] = new_left
    state['new_total'] = new_total
    
    is_valid = len(errors) == 0
    return is_valid, errors, state


def verify_locate_section(
    text: str,
    actual_start: Tuple[int, int],
    actual_exit: Tuple[int, int]
) -> Tuple[bool, List[str]]:
    """Verify the LOCATE START AND EXIT section."""
    errors = []
    
    s_match = re.search(r'S position:\s*\((\d+)\s*,\s*(\d+)\)', text)
    e_match = re.search(r'E position:\s*\((\d+)\s*,\s*(\d+)\)', text)
    
    if s_match:
        claimed_s = (int(s_match.group(1)), int(s_match.group(2)))
        if claimed_s != actual_start:
            errors.append(f"S position {claimed_s} should be {actual_start}")
    
    if e_match:
        claimed_e = (int(e_match.group(1)), int(e_match.group(2)))
        if claimed_e != actual_exit:
            errors.append(f"E position {claimed_e} should be {actual_exit}")
    
    return len(errors) == 0, errors


def format_maze_feedback(errors: List[str], step_num: int) -> str:
    """Format verification errors as feedback to append."""
    if not errors:
        return ""
    
    feedback = f"\n\n[VERIFIER FEEDBACK for Step {step_num}:\n"
    for err in errors:
        feedback += f"  ✗ {err}\n"
    feedback += "Please correct this step and continue.]\n\n"
    return feedback


def format_locate_feedback(errors: List[str]) -> str:
    """Format LOCATE section errors as feedback."""
    if not errors:
        return ""
    
    feedback = "\n\n[VERIFIER FEEDBACK for LOCATE section:\n"
    for err in errors:
        feedback += f"  ✗ {err}\n"
    feedback += "Please correct the start/exit positions and continue.]\n\n"
    return feedback


# =============================================================================
# Relative Position Verification
# =============================================================================

def compute_relative_direction(
    start_pos: Tuple[int, int],
    exit_pos: Tuple[int, int]
) -> str:
    """
    Compute the actual relative direction of exit from start.
    
    Uses standard grid coordinates where:
    - Row increases downward (so lower row = north, higher row = south)
    - Column increases rightward (so lower col = west, higher col = east)
    
    Returns one of: "north", "south", "east", "west", 
                   "northeast", "northwest", "southeast", "southwest",
                   "same" (if positions are identical)
    """
    start_row, start_col = start_pos
    exit_row, exit_col = exit_pos
    
    row_diff = exit_row - start_row  # positive = south, negative = north
    col_diff = exit_col - start_col  # positive = east, negative = west
    
    if row_diff == 0 and col_diff == 0:
        return "same"
    
    # Determine north/south component
    ns_component = ""
    if row_diff < 0:
        ns_component = "north"
    elif row_diff > 0:
        ns_component = "south"
    
    # Determine east/west component
    ew_component = ""
    if col_diff > 0:
        ew_component = "east"
    elif col_diff < 0:
        ew_component = "west"
    
    # Combine components
    if ns_component and ew_component:
        return ns_component + ew_component  # e.g., "northeast"
    elif ns_component:
        return ns_component
    else:
        return ew_component


def normalize_direction(direction: str) -> str:
    """Normalize direction string to standard form."""
    direction = direction.strip().lower()
    
    # Handle common variations
    direction = direction.replace("-", "").replace(" ", "")
    direction = direction.replace("_", "")
    
    # Map variations to standard form
    mappings = {
        "n": "north",
        "s": "south",
        "e": "east",
        "w": "west",
        "ne": "northeast",
        "nw": "northwest",
        "se": "southeast",
        "sw": "southwest",
        "northeasterly": "northeast",
        "northwesterly": "northwest",
        "southeasterly": "southeast",
        "southwesterly": "southwest",
    }
    
    return mappings.get(direction, direction)


def parse_relative_position_answer(text: str) -> Optional[str]:
    """
    Parse the claimed relative direction from the LLM's answer.
    
    Looks for patterns like:
    - "E is to the northeast of S"
    - "the exit is northeast"
    - "\\boxed{northeast}"
    - ">>> ANSWER: northeast"
    """
    # Try boxed answer first
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return normalize_direction(boxed_match.group(1))
    
    # Try >>> ANSWER pattern
    answer_match = re.search(r'>>>\s*ANSWER:\s*(\w+)', text, re.IGNORECASE)
    if answer_match:
        return normalize_direction(answer_match.group(1))
    
    # Try "E is [direction] of S" pattern
    relative_match = re.search(
        r'(?:E|exit|end)\s+is\s+(?:to\s+the\s+)?(\w+)\s+(?:of|from)\s+(?:S|start)',
        text,
        re.IGNORECASE
    )
    if relative_match:
        return normalize_direction(relative_match.group(1))
    
    # Try "direction is [direction]" pattern
    direction_match = re.search(
        r'(?:direction|answer)\s+(?:is|:)\s*(\w+)',
        text,
        re.IGNORECASE
    )
    if direction_match:
        return normalize_direction(direction_match.group(1))
    
    return None


def verify_relative_position(
    text: str,
    start_pos: Tuple[int, int],
    exit_pos: Tuple[int, int]
) -> Tuple[bool, List[str], Optional[str]]:
    """
    Verify the LLM's relative position answer.
    
    Args:
        text: The generated text containing the answer
        start_pos: (row, col) of S
        exit_pos: (row, col) of E
    
    Returns:
        (is_valid, errors, claimed_direction)
    """
    errors = []
    
    # Compute actual direction
    actual_direction = compute_relative_direction(start_pos, exit_pos)
    
    # Parse claimed direction
    claimed_direction = parse_relative_position_answer(text)
    
    if claimed_direction is None:
        errors.append("Could not parse relative direction from answer")
        return False, errors, None
    
    if claimed_direction != actual_direction:
        errors.append(f"Claimed direction '{claimed_direction}' is incorrect. E is {actual_direction} of S.")
    
    is_valid = len(errors) == 0
    return is_valid, errors, claimed_direction


def format_relative_position_feedback(errors: List[str]) -> str:
    """Format relative position errors as feedback."""
    if not errors:
        return ""
    
    feedback = "\n\n[VERIFIER FEEDBACK for relative position:\n"
    for err in errors:
        feedback += f"  ✗ {err}\n"
    feedback += "Please reconsider and correct your answer.]\n\n"
    return feedback


def check_format_compliance(answer_text: str, question_type: str = "right_turns") -> Tuple[bool, List[str]]:
    """
    Check if the answer follows the expected format based on question type.
    
    Args:
        answer_text: The answer text to check
        question_type: One of "right_turns", "total_turns", or "relative_position"
    
    Returns (is_compliant, errors).
    """
    errors = []
    
    # For relative position questions, different format is expected
    if question_type == "relative_position":
        # Check for the expected COMPARE POSITIONS format or direct answer
        if '>>> ANSWER' in answer_text or '\\boxed' in answer_text:
            has_compare = '>>> COMPARE POSITIONS:' in answer_text or '>>> COMPARE' in answer_text
            has_analysis = '>>> ANALYSIS:' in answer_text
            
            if not has_compare and not has_analysis:
                # Still OK if they have a boxed answer
                if '\\boxed' not in answer_text:
                    errors.append("Missing >>> COMPARE POSITIONS: or >>> ANALYSIS: section for relative position question")
                    return False, errors
        
        # Relative position questions don't need STEP format
        return True, []
    
    # For turn-counting questions (right_turns or total_turns)
    # Check for wrong format patterns
    wrong_patterns = [
        (r'>>>\s*TRACE\s+(THE\s+)?PATH', "Uses '>>> TRACE PATH' instead of individual '>>> STEP X:' format"),
        (r'>>>\s*DETERMINE\s+DIRECTIONS', "Uses '>>> DETERMINE DIRECTIONS' instead of individual steps"),
        (r'>>>\s*ANALYZE\s+DIRECTION\s+CHANGES', "Uses '>>> ANALYZE DIRECTION CHANGES' instead of per-step Running count"),
        (r'>>>\s*COUNT\s+RIGHT\s+TURNS', "Uses summary counting instead of per-step Running count"),
    ]
    
    for pattern, error_msg in wrong_patterns:
        if re.search(pattern, answer_text, re.IGNORECASE):
            errors.append(error_msg)
    
    if errors:
        return False, errors
    
    # Check if we have final answer without steps
    if '>>> FINAL ANSWER' in answer_text or '\\boxed' in answer_text:
        has_steps = re.search(r'>>>\s*STEP\s+\d+:', answer_text)
        has_running_count = re.search(r'Running count:\s*Right\s*=\s*\d+', answer_text)
        
        if not has_steps and not has_running_count:
            errors.append("Missing step-by-step trace. Must show each step with >>> STEP X: and Running count.")
            return False, errors
    
    return True, []
