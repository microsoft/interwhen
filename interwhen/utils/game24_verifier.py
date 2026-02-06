"""
Game of 24 Step Verifier

Verifies each step of a Game of 24 solution:
1. Check if the suggested operation is mathematically correct
2. Check if the operation uses numbers from the available numbers
3. Check if remaining numbers are correct (unused + result)
4. Check if 24 is reachable from the remaining numbers (exhaustive search)
5. Check if single remaining number equals 24
6. Check that all numbers are used exactly once

Also provides feedback for LLM to continue generation.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from itertools import permutations, product

logger = logging.getLogger(__name__)


# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def safe_eval(expr: str) -> Optional[float]:
    """Safely evaluate a mathematical expression."""
    try:
        # Only allow basic math operations
        result = eval(expr, {"__builtins__": None}, {})
        return float(result)
    except:
        return None


def is_close(a: float, b: float, tol: float = 1e-3) -> bool:
    """Check if two floats are close enough.
    
    Default tolerance of 1e-3 to handle LLM outputs that are rounded
    to 4 decimal places (e.g., 11/7 = 1.5714 vs actual 1.5714285714).
    """
    return abs(a - b) < tol


def format_number(n: float) -> str:
    """Format number for display (avoid ugly floats)."""
    if is_close(n, round(n)):
        return str(int(round(n)))
    return f"{n:.4f}".rstrip('0').rstrip('.')


# ============================================================================
# EXHAUSTIVE 24 REACHABILITY CHECK
# ============================================================================

OPERATIONS = ['+', '-', '*', '/']


def apply_op(a: float, b: float, op: str) -> Optional[float]:
    """Apply operation and return result, or None if invalid."""
    try:
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if is_close(b, 0):
                return None
            return a / b
    except:
        return None
    return None


def can_reach_24_from_two(nums: List[float]) -> Tuple[bool, Optional[str]]:
    """
    Check if 24 can be reached from exactly 2 numbers.
    Returns (can_reach, example_expression).
    
    With 2 numbers [a, b], we have:
    - 2 orderings: (a, b) and (b, a)
    - 4 operations: +, -, *, /
    = 8 total combinations (but + and * are commutative, so effectively fewer)
    """
    if len(nums) != 2:
        return False, None
    
    a, b = nums[0], nums[1]
    
    for op in OPERATIONS:
        # Try a op b
        result = apply_op(a, b, op)
        if result is not None and is_close(result, 24):
            return True, f"{format_number(a)} {op} {format_number(b)} = 24"
        
        # Try b op a (for non-commutative ops)
        if op in ['-', '/']:
            result = apply_op(b, a, op)
            if result is not None and is_close(result, 24):
                return True, f"{format_number(b)} {op} {format_number(a)} = 24"
    
    return False, None


def can_reach_24_from_three(nums: List[float]) -> Tuple[bool, Optional[str]]:
    """
    Check if 24 can be reached from exactly 3 numbers.
    Returns (can_reach, example_expression).
    
    With 3 numbers [a, b, c], we have:
    - Two bracket structures: ((a op b) op c) or (a op (b op c))
    - 3! = 6 permutations
    - 4^2 = 16 operation combinations
    = 6 * 2 * 16 = 192 combinations (with some duplicates due to commutativity)
    """
    if len(nums) != 3:
        return False, None
    
    # Try all permutations of the 3 numbers
    for perm in permutations(nums):
        a, b, c = perm
        
        # Try all operation combinations
        for op1, op2 in product(OPERATIONS, repeat=2):
            
            # Structure 1: (a op1 b) op2 c
            intermediate = apply_op(a, b, op1)
            if intermediate is not None:
                result = apply_op(intermediate, c, op2)
                if result is not None and is_close(result, 24):
                    expr = f"({format_number(a)} {op1} {format_number(b)}) {op2} {format_number(c)} = 24"
                    return True, expr
            
            # Structure 2: a op1 (b op2 c)
            intermediate = apply_op(b, c, op2)
            if intermediate is not None:
                result = apply_op(a, intermediate, op1)
                if result is not None and is_close(result, 24):
                    expr = f"{format_number(a)} {op1} ({format_number(b)} {op2} {format_number(c)}) = 24"
                    return True, expr
    
    return False, None


def can_reach_24_from_four(nums: List[float]) -> Tuple[bool, Optional[str]]:
    """
    Check if 24 can be reached from exactly 4 numbers.
    Returns (can_reach, example_expression).
    
    With 4 numbers, there are 5 bracket structures:
    1. ((a op b) op c) op d
    2. (a op (b op c)) op d
    3. (a op b) op (c op d)
    4. a op ((b op c) op d)
    5. a op (b op (c op d))
    
    4! = 24 permutations, 4^3 = 64 operation combinations
    = 24 * 5 * 64 = 7680 combinations (with duplicates)
    """
    if len(nums) != 4:
        return False, None
    
    for perm in permutations(nums):
        a, b, c, d = perm
        
        for op1, op2, op3 in product(OPERATIONS, repeat=3):
            
            # Structure 1: ((a op1 b) op2 c) op3 d
            r1 = apply_op(a, b, op1)
            if r1 is not None:
                r2 = apply_op(r1, c, op2)
                if r2 is not None:
                    r3 = apply_op(r2, d, op3)
                    if r3 is not None and is_close(r3, 24):
                        expr = f"(({format_number(a)} {op1} {format_number(b)}) {op2} {format_number(c)}) {op3} {format_number(d)} = 24"
                        return True, expr
            
            # Structure 2: (a op1 (b op2 c)) op3 d
            r1 = apply_op(b, c, op2)
            if r1 is not None:
                r2 = apply_op(a, r1, op1)
                if r2 is not None:
                    r3 = apply_op(r2, d, op3)
                    if r3 is not None and is_close(r3, 24):
                        expr = f"({format_number(a)} {op1} ({format_number(b)} {op2} {format_number(c)})) {op3} {format_number(d)} = 24"
                        return True, expr
            
            # Structure 3: (a op1 b) op2 (c op3 d)
            r1 = apply_op(a, b, op1)
            r2 = apply_op(c, d, op3)
            if r1 is not None and r2 is not None:
                r3 = apply_op(r1, r2, op2)
                if r3 is not None and is_close(r3, 24):
                    expr = f"({format_number(a)} {op1} {format_number(b)}) {op2} ({format_number(c)} {op3} {format_number(d)}) = 24"
                    return True, expr
            
            # Structure 4: a op1 ((b op2 c) op3 d)
            r1 = apply_op(b, c, op2)
            if r1 is not None:
                r2 = apply_op(r1, d, op3)
                if r2 is not None:
                    r3 = apply_op(a, r2, op1)
                    if r3 is not None and is_close(r3, 24):
                        expr = f"{format_number(a)} {op1} (({format_number(b)} {op2} {format_number(c)}) {op3} {format_number(d)}) = 24"
                        return True, expr
            
            # Structure 5: a op1 (b op2 (c op3 d))
            r1 = apply_op(c, d, op3)
            if r1 is not None:
                r2 = apply_op(b, r1, op2)
                if r2 is not None:
                    r3 = apply_op(a, r2, op1)
                    if r3 is not None and is_close(r3, 24):
                        expr = f"{format_number(a)} {op1} ({format_number(b)} {op2} ({format_number(c)} {op3} {format_number(d)})) = 24"
                        return True, expr
    
    return False, None


def can_reach_24(nums: List[float]) -> Tuple[bool, Optional[str]]:
    """
    Check if 24 can be reached from the given numbers.
    Returns (can_reach, example_expression).
    """
    n = len(nums)
    
    if n == 1:
        if is_close(nums[0], 24):
            return True, f"{format_number(nums[0])} = 24"
        return False, None
    elif n == 2:
        return can_reach_24_from_two(nums)
    elif n == 3:
        return can_reach_24_from_three(nums)
    elif n == 4:
        return can_reach_24_from_four(nums)
    else:
        # For more than 4, we'd need more complex logic
        # This shouldn't happen in Game of 24
        logger.warning(f"can_reach_24 called with {n} numbers, expected <= 4")
        return True, None  # Assume reachable


# ============================================================================
# STEP PARSING
# ============================================================================

def parse_step(step_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single step from the LLM output.
    
    Expected format:
    >StepN
    available numbers: [a, b, c, d]
    suggested operation: a * b = result
    remaining numbers: [result, c, d]
    
    Returns a dict with:
    - step_num: int
    - available_numbers: List[float]
    - operand1: float
    - operand2: float  
    - operator: str
    - claimed_result: float
    - remaining_numbers: List[float]
    
    Or None if parsing fails.
    """
    result = {}
    
    # Extract step number
    step_match = re.search(r'>?\s*Step\s*(\d+)', step_text, re.IGNORECASE)
    if step_match:
        result['step_num'] = int(step_match.group(1))
    else:
        result['step_num'] = None
    
    # Extract available numbers
    avail_match = re.search(r'available\s+numbers?\s*:\s*\[([^\]]+)\]', step_text, re.IGNORECASE)
    if avail_match:
        nums_str = avail_match.group(1)
        try:
            result['available_numbers'] = [float(x.strip()) for x in nums_str.split(',') if x.strip()]
        except:
            result['available_numbers'] = None
    else:
        result['available_numbers'] = None
    
    # Extract suggested operation
    # Patterns like: "a * b = c", "a + b = c", "a / b = c", "a - b = c"
    # Also handle fractions in result like "11 / 7 = 11/7"
    op_match = re.search(
        r'suggested\s+operation\s*:\s*([\d.]+)\s*([+\-*/×÷])\s*([\d.]+)\s*=\s*([\d./]+)',
        step_text,
        re.IGNORECASE
    )
    if op_match:
        result['operand1'] = float(op_match.group(1))
        op_str = op_match.group(2)
        # Normalize operators
        if op_str in ['×', 'x', 'X']:
            op_str = '*'
        elif op_str == '÷':
            op_str = '/'
        result['operator'] = op_str
        result['operand2'] = float(op_match.group(3))
        # Handle fraction results like "11/7"
        result_str = op_match.group(4)
        if '/' in result_str:
            parts = result_str.split('/')
            if len(parts) == 2:
                try:
                    result['claimed_result'] = float(parts[0]) / float(parts[1])
                except:
                    result['claimed_result'] = None
            else:
                result['claimed_result'] = None
        else:
            result['claimed_result'] = float(result_str)
    else:
        result['operand1'] = None
        result['operator'] = None
        result['operand2'] = None
        result['claimed_result'] = None
    
    # Extract remaining numbers (also handle fractions like "11/7")
    remain_match = re.search(r'remaining\s+numbers?\s*:\s*\[([^\]]+)\]', step_text, re.IGNORECASE)
    if remain_match:
        nums_str = remain_match.group(1)
        try:
            remaining_nums = []
            for x in nums_str.split(','):
                x = x.strip()
                if not x:
                    continue
                if '/' in x:
                    parts = x.split('/')
                    if len(parts) == 2:
                        remaining_nums.append(float(parts[0]) / float(parts[1]))
                else:
                    remaining_nums.append(float(x))
            result['remaining_numbers'] = remaining_nums
        except:
            result['remaining_numbers'] = None
    else:
        result['remaining_numbers'] = None
    
    return result


def parse_all_steps(text: str, ignore_before_feedback: bool = True) -> List[Dict[str, Any]]:
    """
    Parse all steps from the LLM output.
    
    If ignore_before_feedback is True, only parse steps after the last
    [VERIFIER FEEDBACK block (i.e., the corrected steps).
    
    Only returns COMPLETE steps that have all required fields.
    
    Returns a list of parsed step dictionaries.
    """
    # If there's feedback in the text, only look at the part after the last feedback
    if ignore_before_feedback and "[VERIFIER FEEDBACK" in text:
        # Find the last feedback block
        last_feedback_idx = text.rfind("[VERIFIER FEEDBACK")
        # Find the end of the feedback block (closing bracket followed by newlines)
        feedback_end = text.find("]\n", last_feedback_idx)
        if feedback_end != -1:
            text = text[feedback_end + 2:]  # Skip past the feedback
    
    steps = []
    
    # Split by >Step or >step pattern
    # This regex captures each step block
    step_pattern = re.compile(r'(>?\s*Step\s*\d+[^\n]*\n(?:.*?)(?=(?:>?\s*Step\s*\d+)|(?:>\s*Final)|$))', 
                              re.IGNORECASE | re.DOTALL)
    
    matches = step_pattern.findall(text)
    
    for match in matches:
        parsed = parse_step(match)
        if parsed:
            # Only include steps that are complete (have all required fields)
            # A complete step must have: available_numbers, operand1, operator, operand2, remaining_numbers
            if (parsed.get('available_numbers') is not None and
                parsed.get('operand1') is not None and
                parsed.get('operator') is not None and
                parsed.get('operand2') is not None and
                parsed.get('remaining_numbers') is not None):
                parsed['raw_text'] = match.strip()
                steps.append(parsed)
    
    return steps


def get_state_before_feedback(text: str, original_numbers: List[float]) -> Tuple[List[float], int]:
    """
    Get the valid state (available numbers) right before the last feedback was given.
    
    This handles multiple feedback blocks by recursively processing the text.
    
    Returns (available_numbers, next_step_num)
    """
    if "[VERIFIER FEEDBACK" not in text:
        return list(original_numbers), 1
    
    # Find the last feedback block
    last_feedback_idx = text.rfind("[VERIFIER FEEDBACK")
    text_before_last_feedback = text[:last_feedback_idx]
    
    # Check if there are earlier feedback blocks
    if "[VERIFIER FEEDBACK" in text_before_last_feedback:
        # Recursively get state up to the previous feedback
        prev_feedback_idx = text_before_last_feedback.rfind("[VERIFIER FEEDBACK")
        feedback_end = text_before_last_feedback.find("]\n", prev_feedback_idx)
        if feedback_end == -1:
            feedback_end = text_before_last_feedback.find("]", prev_feedback_idx)
        
        if feedback_end != -1:
            text_after_prev_feedback = text_before_last_feedback[feedback_end + 1:]
        else:
            text_after_prev_feedback = ""
        
        # Get state from before the previous feedback
        base_state, base_step = get_state_before_feedback(
            text_before_last_feedback[:prev_feedback_idx + 1], 
            original_numbers
        )
        
        # Parse and verify steps between the two feedbacks
        step_pattern = re.compile(
            r'(>?\s*Step\s*\d+[^\n]*\n(?:.*?)(?=(?:>?\s*Step\s*\d+)|(?:>\s*Final)|(?:\[VERIFIER)|$))', 
            re.IGNORECASE | re.DOTALL
        )
        matches = step_pattern.findall(text_after_prev_feedback)
        
        current_available = base_state
        current_step = base_step
        
        for match in matches:
            parsed = parse_step(match)
            if parsed:
                is_valid, errors, new_available = verify_step(
                    parsed,
                    current_available,
                    original_numbers,
                    current_step
                )
                
                if is_valid:
                    current_available = new_available
                    current_step += 1
                else:
                    # This step has an error - this is where the last feedback was triggered
                    break
        
        return current_available, current_step
    
    else:
        # Only one feedback block - parse steps before it
        step_pattern = re.compile(
            r'(>?\s*Step\s*\d+[^\n]*\n(?:.*?)(?=(?:>?\s*Step\s*\d+)|(?:>\s*Final)|(?:\[VERIFIER)|$))', 
            re.IGNORECASE | re.DOTALL
        )
        matches = step_pattern.findall(text_before_last_feedback)
        
        current_available = list(original_numbers)
        current_step = 1
        
        for match in matches:
            parsed = parse_step(match)
            if parsed:
                is_valid, errors, new_available = verify_step(
                    parsed,
                    current_available,
                    original_numbers,
                    current_step
                )
                
                if is_valid:
                    current_available = new_available
                    current_step += 1
                else:
                    # This step has an error
                    break
        
        return current_available, current_step


# ============================================================================
# STEP VERIFICATION
# ============================================================================

def verify_step(
    step: Dict[str, Any],
    expected_available: List[float],
    original_numbers: List[float],
    step_num: int
) -> Tuple[bool, List[str], List[float]]:
    """
    Verify a single step of the Game of 24 solution.
    
    Args:
        step: Parsed step dictionary
        expected_available: Expected available numbers at this step
        original_numbers: The original 4 numbers
        step_num: Expected step number
    
    Returns:
        (is_valid, errors, new_available_numbers)
    """
    errors = []
    new_available = expected_available.copy()
    
    # 0. Check step number
    if step.get('step_num') is not None and step['step_num'] != step_num:
        errors.append(f"Step number mismatch: expected Step{step_num}, got Step{step['step_num']}")
    
    # 1. Check if available numbers match expected
    claimed_available = step.get('available_numbers')
    if claimed_available is not None:
        # Compare as multisets (sorted lists)
        if sorted(claimed_available) != sorted(expected_available):
            errors.append(
                f"Available numbers mismatch: claimed {claimed_available}, "
                f"expected {[format_number(x) for x in expected_available]}"
            )
    else:
        errors.append("Could not parse available numbers")
        return False, errors, new_available
    
    # 2. Check if operands are from available numbers
    op1 = step.get('operand1')
    op2 = step.get('operand2')
    operator = step.get('operator')
    claimed_result = step.get('claimed_result')
    
    if op1 is None or op2 is None or operator is None:
        errors.append("Could not parse suggested operation")
        return False, errors, new_available
    
    # Check operand1 is in available numbers
    available_copy = expected_available.copy()
    found_op1 = False
    for i, num in enumerate(available_copy):
        if is_close(num, op1):
            available_copy.pop(i)
            found_op1 = True
            break
    
    if not found_op1:
        errors.append(f"Operand {format_number(op1)} is not in available numbers {[format_number(x) for x in expected_available]}")
    
    # Check operand2 is in remaining available numbers
    found_op2 = False
    for i, num in enumerate(available_copy):
        if is_close(num, op2):
            available_copy.pop(i)
            found_op2 = True
            break
    
    if not found_op2:
        errors.append(f"Operand {format_number(op2)} is not in available numbers after using {format_number(op1)}")
    
    # 3. Check if operation result is mathematically correct
    if claimed_result is not None:
        actual_result = apply_op(op1, op2, operator)
        if actual_result is None:
            errors.append(f"Invalid operation: {format_number(op1)} {operator} {format_number(op2)}")
        elif not is_close(actual_result, claimed_result):
            errors.append(
                f"Arithmetic error: {format_number(op1)} {operator} {format_number(op2)} = "
                f"{format_number(actual_result)}, not {format_number(claimed_result)}"
            )
            # Use the correct result for subsequent checks
            actual_result_to_use = actual_result
        else:
            actual_result_to_use = claimed_result
    else:
        errors.append("Could not parse claimed result")
        return False, errors, new_available
    
    # 4. Check remaining numbers
    claimed_remaining = step.get('remaining_numbers')
    if claimed_remaining is not None:
        # Expected remaining = unused numbers + result
        if 'actual_result_to_use' not in dir():
            actual_result_to_use = claimed_result
        expected_remaining = available_copy + [actual_result_to_use]
        
        # Round to 3 decimal places for comparison since LLM outputs ~4 decimal places
        if sorted([round(x, 3) for x in claimed_remaining]) != sorted([round(x, 3) for x in expected_remaining]):
            errors.append(
                f"Remaining numbers mismatch: claimed {[format_number(x) for x in claimed_remaining]}, "
                f"expected {[format_number(x) for x in expected_remaining]}"
            )
        
        new_available = claimed_remaining if not errors else expected_remaining
    else:
        errors.append("Could not parse remaining numbers")
        return False, errors, new_available
    
    # 5. Check if 24 is still reachable from remaining numbers
    actual_remaining = available_copy + [actual_result_to_use if 'actual_result_to_use' in dir() else claimed_result]
    
    if len(actual_remaining) == 1:
        # Final step - check if result is 24
        if not is_close(actual_remaining[0], 24):
            errors.append(
                f"Final result is {format_number(actual_remaining[0])}, not 24. "
                f"This path cannot reach 24."
            )
    else:
        # Check reachability
        can_reach, example = can_reach_24(actual_remaining)
        if not can_reach:
            errors.append(
                f"Cannot reach 24 from remaining numbers {[format_number(x) for x in actual_remaining]}. "
                f"This path is a dead end. Please try a different operation."
            )
    
    is_valid = len(errors) == 0
    return is_valid, errors, new_available


def verify_all_steps(
    text: str,
    original_numbers: List[float],
    handle_corrections: bool = True
) -> Tuple[bool, List[Dict], Dict[str, Any]]:
    """
    Verify all steps in the LLM output.
    
    If handle_corrections is True and there are [VERIFIER FEEDBACK blocks,
    only verify steps after the last feedback (the corrected steps).
    
    Args:
        text: Full LLM output text
        original_numbers: The original 4 numbers for this problem
        handle_corrections: Whether to handle correction feedback
    
    Returns:
        (all_valid, step_results, summary)
        
        step_results is a list of dicts with:
        - step_num: int
        - is_valid: bool
        - errors: List[str]
        - available_before: List[float]
        - available_after: List[float]
        
        summary contains:
        - total_steps: int
        - valid_steps: int
        - first_error_step: int or None
        - reached_24: bool
    """
    # If there are corrections, get the state before feedback and verify from there
    if handle_corrections and "[VERIFIER FEEDBACK" in text:
        current_available, start_step_num = get_state_before_feedback(text, original_numbers)
        steps = parse_all_steps(text, ignore_before_feedback=True)
    else:
        current_available = list(original_numbers)
        start_step_num = 1
        steps = parse_all_steps(text, ignore_before_feedback=False)
    
    step_results = []
    all_valid = True
    first_error_step = None
    reached_24 = False
    
    for i, step in enumerate(steps):
        # The step number should match what the LLM claims, or be sequential from start
        claimed_step_num = step.get('step_num')
        expected_step_num = start_step_num + i
        
        available_before = current_available.copy()
        
        is_valid, errors, new_available = verify_step(
            step,
            current_available,
            original_numbers,
            expected_step_num
        )
        
        step_results.append({
            'step_num': expected_step_num,
            'claimed_step_num': claimed_step_num,
            'is_valid': is_valid,
            'errors': errors,
            'available_before': available_before,
            'available_after': new_available,
            'raw_text': step.get('raw_text', '')
        })
        
        if not is_valid:
            all_valid = False
            if first_error_step is None:
                first_error_step = expected_step_num
        
        current_available = new_available
        
        # Check if we reached 24
        if len(new_available) == 1 and is_close(new_available[0], 24):
            reached_24 = True
    
    summary = {
        'total_steps': len(steps),
        'valid_steps': sum(1 for r in step_results if r['is_valid']),
        'first_error_step': first_error_step,
        'reached_24': reached_24,
        'final_numbers': current_available
    }
    
    return all_valid, step_results, summary


# ============================================================================
# FEEDBACK GENERATION
# ============================================================================

def format_feedback(errors: List[str], step_num: int = None, available_numbers: List[float] = None) -> str:
    """
    Format verification errors as feedback for the LLM.
    Forces the model to retry the same step by appending the step marker.
    """
    if not errors:
        return ""
    
    if step_num:
        feedback = f"\n\n[VERIFIER FEEDBACK for Step {step_num}:\n"
    else:
        feedback = "\n\n[VERIFIER FEEDBACK:\n"
    
    for err in errors:
        feedback += f"  ✗ {err}\n"
    
    feedback += "The previous steps are correct. Please provide a corrected Step {step_num} and continue.]\n\n".format(step_num=step_num) if step_num else "Please correct and continue.]\n\n"
    
    # Force the model to continue with the same step number
    if step_num:
        feedback += f">Step{step_num}\n"
    
    return feedback


def format_success_feedback(step_num: int, remaining: List[float]) -> str:
    """
    Format success feedback (optional, for debugging).
    """
    return f"[Step {step_num} verified ✓ - remaining: {[format_number(x) for x in remaining]}]\n"


# ============================================================================
# MAIN VERIFICATION INTERFACE
# ============================================================================

def verify_game24_output(
    text: str,
    original_numbers: List[int],
    verbose: bool = False
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Main verification interface for Game of 24 output.
    
    Args:
        text: The LLM output to verify
        original_numbers: The original 4 numbers
        verbose: Whether to log detailed info
    
    Returns:
        (is_valid, feedback, details)
        
        feedback: String to append to LLM output if errors found
        details: Dict with full verification results
    """
    original_floats = [float(x) for x in original_numbers]
    
    all_valid, step_results, summary = verify_all_steps(text, original_floats)
    
    if verbose:
        logger.info(f"Verification summary: {summary}")
        for result in step_results:
            status = "✓" if result['is_valid'] else "✗"
            logger.info(f"  Step {result['step_num']}: {status}")
            for err in result['errors']:
                logger.info(f"    - {err}")
    
    # Generate feedback for first error
    feedback = ""
    if not all_valid:
        for result in step_results:
            if not result['is_valid']:
                feedback = format_feedback(result['errors'], result['step_num'])
                break
    
    details = {
        'all_valid': all_valid,
        'step_results': step_results,
        'summary': summary
    }
    
    return all_valid, feedback, details


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Test the verifier with sample output
    
    logging.basicConfig(level=logging.INFO)
    
    # Sample correct output
    correct_output = """
>Step1
available numbers: [1, 1, 1, 8]
suggested operation: 1 + 1 = 2
remaining numbers: [2, 1, 8]

>Step2
available numbers: [2, 1, 8]
suggested operation: 2 + 1 = 3
remaining numbers: [3, 8]

>Step3
available numbers: [3, 8]
suggested operation: 3 * 8 = 24
remaining numbers: [24]

> Final expression: \\boxed{(1 + 1 + 1) * 8}
"""
    
    print("=" * 60)
    print("Testing CORRECT output:")
    print("=" * 60)
    is_valid, feedback, details = verify_game24_output(correct_output, [1, 1, 1, 8], verbose=True)
    print(f"\nIs valid: {is_valid}")
    print(f"Feedback: {feedback if feedback else 'None'}")
    print(f"Reached 24: {details['summary']['reached_24']}")
    
    # Sample output with arithmetic error
    arithmetic_error = """
>Step1
available numbers: [2, 3, 4, 5]
suggested operation: 2 * 3 = 7
remaining numbers: [7, 4, 5]
"""
    
    print("\n" + "=" * 60)
    print("Testing ARITHMETIC ERROR:")
    print("=" * 60)
    is_valid, feedback, details = verify_game24_output(arithmetic_error, [2, 3, 4, 5], verbose=True)
    print(f"\nIs valid: {is_valid}")
    print(f"Feedback: {feedback}")
    
    # Sample output with wrong operand
    wrong_operand = """
>Step1
available numbers: [2, 3, 4, 5]
suggested operation: 2 * 6 = 12
remaining numbers: [12, 4, 5]
"""
    
    print("\n" + "=" * 60)
    print("Testing WRONG OPERAND:")
    print("=" * 60)
    is_valid, feedback, details = verify_game24_output(wrong_operand, [2, 3, 4, 5], verbose=True)
    print(f"\nIs valid: {is_valid}")
    print(f"Feedback: {feedback}")
    
    # Sample output that leads to dead end
    dead_end = """
>Step1
available numbers: [1, 2, 3, 4]
suggested operation: 1 * 2 = 2
remaining numbers: [2, 3, 4]

>Step2
available numbers: [2, 3, 4]
suggested operation: 2 * 3 = 6
remaining numbers: [6, 4]

>Step3
available numbers: [6, 4]
suggested operation: 6 + 4 = 10
remaining numbers: [10]
"""
    
    print("\n" + "=" * 60)
    print("Testing DEAD END (doesn't reach 24):")
    print("=" * 60)
    is_valid, feedback, details = verify_game24_output(dead_end, [1, 2, 3, 4], verbose=True)
    print(f"\nIs valid: {is_valid}")
    print(f"Feedback: {feedback}")
    
    # Test reachability function
    print("\n" + "=" * 60)
    print("Testing REACHABILITY:")
    print("=" * 60)
    
    test_cases = [
        [24],  # Single 24
        [8, 3],  # 8 * 3 = 24
        [6, 4],  # 6 * 4 = 24
        [2, 3, 4],  # 2 * 3 * 4 = 24
        [1, 5, 5, 5],  # (5 - 1/5) * 5 = 24
        [10],  # Single 10 - unreachable
        [1, 1],  # Two 1s - unreachable
    ]
    
    for nums in test_cases:
        can_reach, expr = can_reach_24(nums)
        print(f"  {nums}: {can_reach} - {expr if expr else 'No solution'}")
