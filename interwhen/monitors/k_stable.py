import re
from .base import VerifyMonitor
import asyncio

NEGATION_WORDS = ["not", "isn't", "isnt", "no ", "cannot", "can't", "cant",
                  "doesn't", "doesnt", "never"]


class KstableAnswerMCQMonitor(VerifyMonitor):
    """
    Monitor that detects when the model has stabilized on the same answer k times.
    
    When the same normalized answer appears k consecutive times in lines containing
    "answer", the monitor signals to stop generation early.
    """
    
    def __init__(self, name, k, options, answer_start_token="</think>"):
        """
        Args:
            name: Monitor name
            k: Number of consecutive same answers required to trigger early stop
            options: Dictionary mapping option letters to values, e.g. {"A": "Yes", "B": "No", "C": "2", "D": "4"}
            answer_start_token: Token that signals end of thinking (default: "</think>")
        """
        super().__init__(name)
        self.k = k
        self.options = options
        self.answer_start_token = answer_start_token
        self.stabilized_answer = None
        # Instantiate Lock for safer async execution
        self.lock = asyncio.Lock()

    def _contains_negation(self, text: str) -> bool:
        """Check if text contains negation words indicating uncertainty."""
        t = text.lower()
        return any(neg in t for neg in NEGATION_WORDS)

    def _contains_all_options(self, line: str) -> bool:
        """Check if line lists all options (just restating the problem)."""
        # Check keys A,B,C,D exactly as they appear
        if all(k in line for k in self.options.keys()):
            return True
        # Check values exactly as they appear
        if all(v in line for v in self.options.values()):
            return True
        # Check full formats A.4, B.8, etc.
        fulls = [f"{k}.{v}" for k, v in self.options.items()]
        if all(full in line for full in fulls):
            return True
        return False

    def _normalize_answer(self, ans: str) -> str:
        """Convert extracted answer to normalized form for comparison."""
        raw_ans = ans.strip().replace('"', '').replace("'", "").strip()
        ans_lower = raw_ans.lower()

        # A.2 / C.0 / D.10 pattern
        m = re.match(r"^([a-d])[\.\d]*$", ans_lower)
        if m:
            letter = m.group(1).upper()
            return self.options.get(letter, letter)

        # Exact dictionary value match
        for key, val in self.options.items():
            if ans_lower == val.lower():
                return val

        # Substring match with dictionary value
        for key, val in self.options.items():
            if val.lower() in ans_lower:
                return val

        # Pure letter A/B/C/D
        if ans_lower in ["a", "b", "c", "d"]:
            return self.options.get(ans_lower.upper(), raw_ans)

        return raw_ans

    def _extract_answer_from_segment(self, segment: str, line: str):
        """Extract the answer candidate from segment after 'answer'."""
        seg_low = segment.lower()

        # Skip negation lines
        if self._contains_negation(segment):
            return None

        # Skip if listing all options
        if self._contains_all_options(line):
            return None

        tokens = re.split(r"[ ,;:\(\)]", segment)
        candidates = []

        # A.2 / C.0 type patterns
        for tok in tokens:
            if re.match(r"^[a-d][\.\d]+$", tok.lower()):
                candidates.append(tok)

        # Single letter A B C D
        for tok in tokens:
            if tok.lower() in ["a", "b", "c", "d"]:
                candidates.append(tok)

        # Dictionary values
        for key, val in self.options.items():
            if val.lower() in seg_low:
                candidates.append(val)

        candidates = list(set(candidates))

        # Normalize candidates before checking uniqueness
        normalized_candidates = [self._normalize_answer(c) for c in candidates]
        unique_normalized = list(set(normalized_candidates))

        # Only accept if exactly 1 unique normalized answer found
        if len(unique_normalized) == 1:
            return unique_normalized[0]

        return None

    def _extract_answer_from_line(self, line: str):
        """Extract answer from a line containing 'answer' keyword."""
        line_low = line.lower()

        if "answer" not in line_low:
            return None

        # Get text after the word "answer"
        if "answer" in line:
            after = line.split("answer", 1)[1].strip()
        elif "Answer" in line:
            after = line.split("Answer", 1)[1].strip()
        elif "ANSWER" in line:
            after = line.split("ANSWER", 1)[1].strip()
        else:
            return None

        # Slice until the first period
        idx = after.find(".")
        segment = after[:idx] if idx != -1 else after
        segment = segment.strip()

        if not segment:
            return None

        return self._extract_answer_from_segment(segment, line)

    def _check_answer_stability(self, generated_text: str):
        """
        Check if answer has stabilized k times in the generated text.
        
        Returns:
            (is_stable, cutoff_position, stabilized_answer)
        """
        lines = generated_text.splitlines()
        
        prev_normalized = None
        count = 0
        
        for i, line in enumerate(lines):
            raw_answer = self._extract_answer_from_line(line)
            
            if raw_answer is None:
                continue

            normalized = self._normalize_answer(raw_answer)

            if normalized == prev_normalized:
                count += 1
            else:
                prev_normalized = normalized
                count = 1

            if count >= self.k:
                cutoff_position = sum(len(l) + 1 for l in lines[:i + 1])  # +1 for \n
                self.stabilized_answer = normalized
                return True, cutoff_position, normalized

        return False, None, None

    async def _verify(self, step, token_index):
        """
        Check if answer has stabilized k times.
        
        Returns:
            (is_valid, sliced_text, cutoff_index) where is_valid=False means we should stop
        """
        is_stable, cutoff_pos, answer = self._check_answer_stability(step)
        
        if is_stable:
            # Slice text at cutoff position
            sliced_text = step[:cutoff_pos].rstrip()
            return (False, sliced_text, cutoff_pos)
        
        return (True, step, token_index)

    async def verify(self, step, token_index, event, event_info):
        """Verify if answer has stabilized and signal if we should stop."""
        is_valid, updated_text, correction_index = await self._verify(step, token_index)
        
        if is_valid:
            return step, None
        
        async with self.lock:
            if not event.is_set():
                event_info["generated_text"] = step
                event_info["feedback"] = "</think>"  # Sliced text up to k-stable point
                event_info["correction_index"] = correction_index
                event.set()
    
    async def fix(self, generated_text, event_info, fix_method=None):
        """Return the sliced text up to the k-stable point."""
        return event_info["generated_text"] + "\n\n" + event_info["feedback"]
        
    def step_extractor(self, chunk, generated_text):
        """
        When chunk contains \\n, check the line from this \\n to the previous \\n in generated_text.
        """
        if self.answer_start_token in generated_text:
            return False, None
        
        if "\n" not in chunk:
            return False, None
        
        newline_pos_in_chunk = chunk.find("\n")
        newline_pos_in_text = len(generated_text) - len(chunk) + newline_pos_in_chunk
        
        prev_newline_pos = generated_text.rfind("\n", 0, newline_pos_in_text)
        
        if prev_newline_pos == -1:
            latest_line = generated_text[:newline_pos_in_text]
        else:
            latest_line = generated_text[prev_newline_pos + 1:newline_pos_in_text]
        
        if "answer" in latest_line.lower():
            return True, generated_text[:newline_pos_in_text]
        
        return False, None


class KstableAnswerGame24Monitor(VerifyMonitor):
    """
    Monitor that detects when the model has stabilized on the same equation k times for Game of 24.
    
    When the same normalized arithmetic expression appears k consecutive times,
    the monitor signals to stop generation early.
    
    Optionally validates that the equation uses exactly the expected numbers.
    """
    
    def __init__(self, name, k, expected_nums=None, answer_start_token="</think>"):
        """
        Args:
            name: Monitor name
            k: Number of consecutive same equations required to trigger early stop
            expected_nums: Optional list of numbers that must be used in the equation (e.g., [1, 2, 3, 4])
            answer_start_token: Token that signals end of thinking (default: "</think>")
        """
        super().__init__(name)
        self.k = k
        self.expected_nums = expected_nums
        self.answer_start_token = answer_start_token
        self.stabilized_equation = None
        # Instantiate Lock for safer async execution
        self.lock = asyncio.Lock()

    def _extract_numbers_from_expr(self, expr):
        """Extract all numbers (including decimals) from an expression."""
        numbers = re.findall(r'\d+\.?\d*', expr)
        return [int(float(n)) if float(n).is_integer() else float(n) for n in numbers]

    def _validate_numbers_used(self, expr):
        """Check if the expression uses exactly the expected numbers (each exactly once)."""
        if self.expected_nums is None:
            return True
        used_nums = self._extract_numbers_from_expr(expr)
        return sorted(used_nums) == sorted(self.expected_nums)

    def _normalize_equation(self, eq_str):
        """
        Normalize an equation string for comparison.
        - Convert LaTeX operators to standard operators
        - Remove whitespace
        - Handle common variations
        """
        if not eq_str:
            return None
        
        # Convert LaTeX operators
        eq_str = eq_str.replace(r"\times", "*")
        eq_str = eq_str.replace(r"\cdot", "*")
        eq_str = eq_str.replace(r"\div", "/")
        eq_str = eq_str.replace("×", "*")
        eq_str = eq_str.replace("÷", "/")
        
        # Remove LaTeX spacing and regular whitespace
        eq_str = eq_str.replace(r"\,", "")
        eq_str = eq_str.replace(r"\ ", "")
        eq_str = re.sub(r'\s+', '', eq_str)
        
        return eq_str.lower()

    def _extract_equation_from_line(self, line):
        """
        Extract an arithmetic equation from a line of reasoning text.
        
        Looks for patterns like:
        - "expression is (1+2)*3*4"
        - "answer is (7+1)*(2+1)"
        - "= 24" patterns with preceding equation
        - Patterns with boxed content
        - LaTeX operators like \times, \div
        - Unicode operators like ×, ÷
        
        Returns the equation string or None if no equation found.
        """
        # First check for boxed content (highest priority - final answers)
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', line)
        if boxed_match:
            return boxed_match.group(1).strip()
        
        # Pattern for "So the expression is X", "expression is X", "answer is X", "solution is X"
        # Also handles "the expression would be X", "this gives us X"
        # Check this BEFORE the general "= 24" pattern to avoid partial matches
        answer_pattern = r'(?:So\s+)?(?:the\s+)?(?:expression|answer|solution|result)(?:\s+would\s+be|\s+is|\s*:)\s*([\d\(\)\+\-\*\/\×\÷\s\\times\\cdot\\div]+?)(?:\.|,|$|\s*=)'
        match = re.search(answer_pattern, line, re.IGNORECASE)
        if match:
            eq = match.group(1).strip()
            # Clean trailing punctuation
            eq = re.sub(r'[.,;:!?]+$', '', eq)
            if eq and re.search(r'[\+\-\*\/\×\÷]|\\times|\\cdot|\\div', eq):
                return eq
        
        # Pattern for "So X = 24" or "Then, X = 24" or "Thus X = 24"
        # Use word boundary to avoid partial word matches
        then_pattern = r'\b(?:So|Then|Thus|Therefore|Hence)\b[,]?\s+([\d\(\)\+\-\*\/\×\÷\s\\times\\cdot\\div]+?)\s*=\s*24'
        match = re.search(then_pattern, line, re.IGNORECASE)
        if match:
            eq = match.group(1).strip()
            if re.search(r'[\+\-\*\/\×\÷]|\\times|\\cdot|\\div', eq):
                return eq
        
        # Pattern for equations that evaluate to 24
        # Looking for patterns like "(7+1)*(2+1) = 24" or "8*3 = 24" or "8 × 3 = 24"
        # Also matches LaTeX: "8 \times 3 = 24"
        eq_24_pattern = r'([\d\(\)\+\-\*\/\×\÷\s\\times\\cdot\\div]+)\s*=\s*24'
        match = re.search(eq_24_pattern, line)
        if match:
            eq = match.group(1).strip()
            # Make sure it has operators (not just "24" or simple numbers)
            # Also clean up any leading non-equation characters
            eq = re.sub(r'^[^0-9\(]+', '', eq)
            if re.search(r'[\+\-\*\/\×\÷]|\\times|\\cdot|\\div', eq):
                return eq
        
        # Pattern for "X evaluates to 24" or "X equals 24"
        eval_pattern = r'([\d\(\)\+\-\*\/\×\÷\s\\times\\cdot\\div]+?)\s+(?:evaluates?\s+to|equals?|gives?)\s+24'
        match = re.search(eval_pattern, line, re.IGNORECASE)
        if match:
            eq = match.group(1).strip()
            # Clean up leading non-equation characters
            eq = re.sub(r'^[^0-9\(]+', '', eq)
            if re.search(r'[\+\-\*\/\×\÷]|\\times|\\cdot|\\div', eq):
                return eq
        
        # Pattern for standalone equations with parentheses that might equal 24
        # e.g., "(7 + 1) * (2 + 1)" or "((8-2)*4)" or "6 × (9 - 4 - 1)"
        paren_eq_pattern = r'(\([^)]+\)\s*[\+\-\*\/\×\÷]\s*\([^)]+\)|\([^)]+\)\s*[\+\-\*\/\×\÷]\s*\d+|\d+\s*[\+\-\*\/\×\÷]\s*\([^)]+\))'
        matches = re.findall(paren_eq_pattern, line)
        for m in matches:
            # Try to evaluate to check if it equals 24
            try:
                normalized = self._normalize_equation(m)
                if normalized:
                    val = eval(normalized, {"__builtins__": None}, {})
                    if abs(val - 24) < 1e-6:
                        return m
            except:
                pass
        
        return None

    def _check_equation_stability(self, generated_text: str):
        """
        Check if equation has stabilized k times in the generated text.
        
        If expected_nums is set, only counts equations that use exactly those numbers.
        
        Returns:
            (is_stable, cutoff_position, stabilized_equation)
        """
        lines = generated_text.splitlines()
        
        prev_normalized = None
        count = 0
        
        for i, line in enumerate(lines):
            eq = self._extract_equation_from_line(line)
            
            if eq is None:
                continue

            normalized = self._normalize_equation(eq)
            if normalized is None:
                continue
            
            # Validate that the equation uses the expected numbers (if specified)
            if self.expected_nums is not None:
                if not self._validate_numbers_used(normalized):
                    continue

            if normalized == prev_normalized:
                count += 1
            else:
                prev_normalized = normalized
                count = 1

            if count >= self.k:
                cutoff_position = sum(len(l) + 1 for l in lines[:i + 1])  # +1 for \n
                self.stabilized_equation = eq
                return True, cutoff_position, eq

        return False, None, None

    async def _verify(self, generated_text, token_index):
        """
        Check if equation has stabilized k times.
        
        Returns:
            (is_valid, sliced_text, cutoff_index) where is_valid=False means we should stop
        """
        is_stable, cutoff_pos, equation = self._check_equation_stability(generated_text)
        
        if is_stable:
            # Slice text at cutoff position
            sliced_text = generated_text[:cutoff_pos].rstrip()
            return (False, sliced_text, cutoff_pos)
        
        return (True, generated_text, token_index)

    async def verify(self, chunk, token_index, event, event_info):
        """Verify if equation has stabilized and signal if we should stop."""
        is_valid, updated_text, correction_index = await self._verify(chunk, token_index)
        
        if is_valid:
            return chunk, None
        
        async with self.lock:
            if not event.is_set():
                event_info["generated_text"] = chunk
                event_info["feedback"] = "</think>"
                event_info["correction_index"] = correction_index
                event.set()
    
    async def fix(self, generated_text, event_info, fix_method=None):
        """Return the sliced text up to the k-stable point."""
        return event_info["generated_text"] + "\n\n" + event_info["feedback"]
        
    def step_extractor(self, chunk, generated_text):
        """
        When chunk contains \\n, check if the line contains an equation pattern.
        
        Triggers verification when line contains:
        - "= 24" pattern
        - "answer/expression/solution is" pattern
        - boxed content
        - parenthesized equations
        """
        if self.answer_start_token in generated_text:
            return False, None
        
        if "\n" not in chunk:
            return False, None
        
        newline_pos_in_chunk = chunk.find("\n")
        newline_pos_in_text = len(generated_text) - len(chunk) + newline_pos_in_chunk
        
        prev_newline_pos = generated_text.rfind("\n", 0, newline_pos_in_text)
        
        if prev_newline_pos == -1:
            latest_line = generated_text[:newline_pos_in_text]
        else:
            latest_line = generated_text[prev_newline_pos + 1:newline_pos_in_text]
        
        # Check for equation patterns
        line_lower = latest_line.lower()
        has_equation_pattern = (
            "= 24" in latest_line or
            "=24" in latest_line or
            "answer" in line_lower or
            "expression" in line_lower or
            "solution" in line_lower or
            r"\boxed" in latest_line or
            # Check for parenthesized expressions with operators
            (re.search(r'\([^)]+\)\s*[\+\-\*\/\×\÷]', latest_line) is not None)
        )
        
        if has_equation_pattern:
            return True, generated_text[:newline_pos_in_text]
        
        return False, None