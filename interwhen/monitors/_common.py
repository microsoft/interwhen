"""
Shared utilities for thinking-phase verifier monitors.
"""

import re
from typing import Optional


def find_complete_boxed(text: str) -> Optional[object]:
    """Find a complete \\boxed{...} in text, handling nested braces.

    Unlike ``re.search(r'\\boxed\\{[^}]+\\}', text)`` this correctly
    handles LaTeX like ``\\boxed{12\\frac{1}{2}}`` where the naive
    ``[^}]+`` pattern would stop at the first ``}``.

    Returns a match-like object with ``.start()`` and ``.end()``
    spanning the full ``\\boxed{...}`` (including the outer braces),
    or ``None`` if no complete boxed expression is found.
    """
    idx = 0
    while idx < len(text):
        pos = text.find(r'\boxed{', idx)
        if pos == -1:
            return None
        # Start counting braces from after '\boxed{'
        brace_start = pos + len(r'\boxed{')
        depth = 1
        i = brace_start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            match_start = pos
            match_end = i  # i is right after the closing '}'
            content = text[brace_start:i - 1].strip()
            if content:
                class _BoxedMatch:
                    def __init__(self, s, e):
                        self._start, self._end = s, e
                    def start(self):
                        return self._start
                    def end(self):
                        return self._end
                    def group(self, n=0):
                        return text[self._start:self._end]
                return _BoxedMatch(match_start, match_end)
        idx = pos + 1
    return None
