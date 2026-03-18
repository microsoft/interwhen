"""
Step Verifier for Verina (Lean 4 code generation).

This monitor:
1. Counts reasoning steps (newlines) during generation
2. After K newlines, forces the model to output code by closing </think> and prompting [CODE]
3. Extracts code from [CODE]...[/CODE] tags
4. If verification fails, injects feedback and lets the model retry
"""

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional
from interwhen.utils.verina_verifier_utils import *
import shutil
import httpx
import copy

from .base import VerifyMonitor

# Verina Paths
VERINA_ROOT = (Path(__file__).parent.resolve() / ".." / ".." / ".." / "verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

# STEP VERIFIER VERINA MONITOR
class StepVerifierVerinaMonitor(VerifyMonitor):
    """
    Step-by-step verifier monitor for Verina (Lean 4 code generation).
    
    This monitor:
    1. Counts reasoning steps (newlines) during generation
    2. After K newlines, forces the model to output code by streaming with </think> + [CODE]
    3. When [CODE]...[/CODE] is detected, extracts and verifies via Lean compilation
    4. If verification fails, injects feedback for retry
    """
    
    def __init__(
        self,
        name: str,
        task_data: dict,  # Contains signature, lean_data, description, data_id, tests, etc.
        llm_server: dict,  # LLM server config for forcing code output
        prompt: str,  # The original prompt (needed for continuation)
        k_steps: int = 10,  # Number of newlines before forcing code output
        max_corrections: int = 5,
        compile_timeout: int = 120,
        async_execution: bool = True,
    ):
        super().__init__(name)
        self.task_data = task_data
        self.llm_server = llm_server
        self.prompt = prompt
        self.k_steps = k_steps
        self.max_corrections = max_corrections
        self.compile_timeout = compile_timeout
        self.async_execution = async_execution
        self.max_corrections = max_corrections
        self.num_corrections = 0
        
        # State tracking
        self.verification_count = 0
        self.force_count = 0  # Track how many times code was forced
        self.last_newline_count = 0
        self.last_verified_code_end = 0  # Position of last [/CODE] we verified
        self.success_found = False  # Once True, block all future failure feedback
        self.verified_code = None  # Store the code that compiled successfully
        self.last_triggered_think_start = -1  # Position of <think> block we're tracking
        self.last_triggered_think_newlines = 0  # Newlines in think block at last trigger
        
        # Diversity tracking: remember what failed so we can steer away from it
        self.failed_attempts = []  # List of (code_snippet, error_summary) tuples
        self.base_k_steps = k_steps  # Original k_steps before progressive scaling
        
        # Lock for thread safety
        self.lock = asyncio.Lock()

    def reset(self):
        """Reset state for a new problem."""
        self.verification_count = 0
        self.force_count = 0
        self.last_newline_count = 0
        self.last_verified_code_end = 0
        self.success_found = False
        self.verified_code = None
        self.last_triggered_think_start = -1
        self.last_triggered_think_newlines = 0
        self.failed_attempts = []
        self.num_corrections = 0
        self.k_steps = self.base_k_steps
    
    def get_force_count(self) -> int:
        """Return the number of times code was forced for this sample."""
        return self.force_count

    async def verify_final_code(
        self, 
        code: str, 
        prompt_with_answer: str, 
        max_retries: int = 3
    ) -> Tuple[str, bool, str, int]:
        """
        Verify extracted final code and retry with LLM if compilation fails.
        Allows model to think before outputting corrected code.
        
        Args:
            code: The extracted code to verify
            prompt_with_answer: The full prompt + model answer (for continuing conversation)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (final_code, compiles, compile_output, num_retries)
        """
        
        current_code = code
        current_prompt = prompt_with_answer
        num_retries = 0
        
        for attempt in range(max_retries):
            # Verify compilation
            compiles, compile_output = await self._verify_compilation(current_code)
            
            if compiles:
                print(f"[Verina Final] Code compiles successfully (attempt {attempt + 1})")
                return current_code, True, compile_output, num_retries
            
            num_retries += 1
            print(f"[Verina Final] Compilation failed (attempt {attempt + 1}), retrying...")
            
            # Build retry prompt with error feedback
            error_feedback = f"""
<|im_end|>
<|im_start|>user
The code you gave failed with error:
{clean_compile_output(compile_output)}

Please fix the error and provide the corrected code. Think through the problem carefully, then output your solution wrapped in [CODE]...[/CODE] tags.

{LEAN4_API_REFERENCE}
<|im_end|>
<|im_start|>assistant
<think>
"""
            retry_prompt = current_prompt + error_feedback
            
            # Call LLM for retry
            try:
                payload = copy.deepcopy(self.llm_server.get("payload", {}))
                payload["prompt"] = retry_prompt
                payload["max_tokens"] = 2048  # More tokens to allow thinking
                payload["stream"] = False
                payload["stop"] = ["[/CODE]", "</s>", "<|im_end|>"]
                
                headers = copy.deepcopy(self.llm_server.get("headers", {}))
                url = self.llm_server["url"]
                
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    result = response.json()
                    full_response = result["choices"][0]["text"].strip()
                    
                    # Extract code from [CODE]...[/CODE] block
                    new_code = extract_code_from_response(full_response)
                    
                    if new_code:
                        current_code = new_code
                        current_prompt = retry_prompt + full_response + "[/CODE]"
                        print(f"[Verina Final] Got new code: ...")
                    else:
                        print(f"[Verina Final] No code block found in response, keeping previous code")
                        
            except Exception as e:
                print(f"[Verina Final] LLM retry failed: {e}")
                break
        
        return current_code, False, compile_output, num_retries

    def _count_feedback_blocks(self, text: str) -> int:
        """Count how many [VERIFIER FEEDBACK...] blocks are in the text."""
        return len(re.findall(r'\[VERIFIER FEEDBACK[^\]]*\]', text))

    def _count_newlines(self, text: str) -> int:
        """Count the number of newlines in text."""
        return text.count('\n')

    def _has_complete_code_block(self, text: str) -> bool:
        """Check if text contains a complete [CODE]...[/CODE] block."""
        return bool(re.search(r'\[CODE\].*?\[/CODE\]', text, re.DOTALL | re.IGNORECASE))
    
    def _has_think_end(self, text: str) -> bool:
        """Check if </think> is present."""
        return "</think>" in text.lower()

    _FORCE_CODE_VARIANTS = [
        """\n\nWait, let me now output the full code as per my current understanding, so that the user can give feedback.
</think> The final code is:
[CODE]""",
        """\n\nLet me try writing my solution now.
</think> Here is my implementation:
[CODE]""",
        """\n\nOk let me take a step back and write the code using a different strategy than before.
</think> My approach:
[CODE]""",
        """\n\nI should try a completely different algorithm this time.
</think> Alternative solution:
[CODE]""",
    ]

    def _build_force_code_feedback(self) -> str:
        """Build the feedback string to force code output. Rotates prompts for diversity."""
        idx = self.force_count % len(self._FORCE_CODE_VARIANTS)
        return self._FORCE_CODE_VARIANTS[idx]

    def _build_diversity_feedback(self, compile_output: str) -> str:
        """
        Build feedback that escalates diversity with each failed attempt.
        
        - Attempt 1: Just show the error and ask to fix
        - Attempt 2+: Show error + all previous failed approaches, explicitly
          ask for a fundamentally different strategy
        """
        cleaned_error = clean_compile_output(compile_output)
        n_failures = len(self.failed_attempts)

        if n_failures <= 1:
            # First failure: straightforward fix request
            return f"""
<|im_end|>
<|im_start|>user
The code you gave failed with error:
{cleaned_error}

Please fix the error. Your code should compile.

{LEAN4_API_REFERENCE}"""

        # 2+ failures: show history and demand a different approach
        prev_section = ""
        for i, (code_snip, err_snip) in enumerate(self.failed_attempts[:-1], 1):
            prev_section += f"\n--- Failed Attempt {i} ---\n{code_snip}\nError: {err_snip}\n"

        return f"""
<|im_end|>
<|im_start|>user
The code you gave failed with error:
{cleaned_error}

You have now failed {n_failures} time(s). Here are your previous failed attempts:
{prev_section}
Do NOT repeat a similar approach. Use a fundamentally different algorithm or data structure. Think step-by-step about a fresh solution before writing code.

{LEAN4_API_REFERENCE}"""

    async def _stream_force_code(self, current_text: str, max_tokens: int = 2048) -> str:
        """
        Stream from LLM to force code output.
        
        Appends the force-code prompt and continues streaming until [/CODE] is found
        or max_tokens reached.
        
        Args:
            current_text: The text generated so far (reasoning only, not including original prompt)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Complete text with forced code (the new generated portion only)
        """
        force_prompt = self._build_force_code_feedback()
        # Include original prompt + reasoning so far + force code trigger
        full_prompt = self.prompt + current_text + force_prompt
        
        payload = copy.deepcopy(self.llm_server.get("payload", {}))
        payload["prompt"] = full_prompt
        payload["max_tokens"] = max_tokens
        
        # Copy headers
        headers = copy.deepcopy(self.llm_server.get("headers", {}))
        url = self.llm_server["url"]
    
        
        generated_text = ""  # Initialize the accumulator for streamed text
        
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                url,
                headers=headers,
                json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[len("data: "):].strip()
                        if data == "[DONE]":
                            break
                        
                        chunk = json.loads(data)["choices"][0]["text"]
                        generated_text += chunk
                        
                        # Stop when we see [/CODE]
                        if "[/CODE]" in generated_text:
                            break
        
        return generated_text

    async def _verify_compilation(self, code: str) -> Tuple[bool, str]:
        """Run Lean compilation check using existing helpers."""
        lean_content = build_lean_content(self.task_data, code)
        task_id = self.task_data.get("data_id", "unknown")
        lean_file = create_lean_file(f"verify_{task_id}_{self.verification_count}", lean_content)
        # Run compilation
        compiles, output = await asyncio.to_thread(
            check_lean_compile, lean_file, self.compile_timeout
        )
        return compiles, output

    def sync_verify_compilation(self, code: str) -> Tuple[bool, str]:
        """Run Lean compilation check using existing helpers."""
        lean_content = build_lean_content(self.task_data, code)
        task_id = self.task_data.get("data_id", "unknown")
        lean_file = create_lean_file(f"verify_{task_id}_{self.verification_count}", lean_content)
        # Run compilation
        compiles, output = check_lean_compile(lean_file, self.compile_timeout)
        return compiles, output

    async def verify(self, step: str, token_index: int, event: asyncio.Event, event_info: dict):
        """
        Verify the generated text.
        
        Args:
            step: The text to verify (from step_extractor)
            token_index: Current token index
            event: asyncio.Event to signal when intervention needed
            event_info: Dict to store feedback info
        """
        async with self.lock:
            if self.num_corrections >= self.max_corrections:
                max_feedback = "\nthe answer is \\boxed{no solution}"
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = max_feedback
                    event_info["correction_index"] = token_index
                    event.set()
                return step, max_feedback

        # Force code output (triggered by step_extractor)
        async with self.lock:
            self.force_count += 1
            print(f"[Verina] Forcing code output after {self._count_newlines(step)} newlines (force #{self.force_count})")
        
        full_text = await self._stream_force_code(step)
        full_text_with_code = "[CODE]" + full_text
        code = extract_code_from_response(full_text_with_code)
        
        if not code:
            print("[Verina] Forced code generation but could not extract code")
            async with self.lock:
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = None
            return step, None
        
        async with self.lock:
            self.verification_count += 1
            current_verification = self.verification_count

        # Verify forced code compilation
        compiles, compile_output = await self._verify_compilation(code)

        if not compiles:
            # Track this failure for diversity
            code_preview = code[:200] + ('...' if len(code) > 200 else '')
            error_summary = clean_compile_output(compile_output)[:300]
            async with self.lock:
                self.failed_attempts.append((code_preview, error_summary))
            
            # Build the complete text
            force_prompt = self._build_force_code_feedback()
            complete_generated = force_prompt + full_text
            
            # Build escalating feedback based on how many times we've failed
            feedback = self._build_diversity_feedback(compile_output)
            feedback += """
<|im_end|>
<|im_start|>assistant
<think> It seems my code failed to compile. I should analyze the errror and try to fix it using a different approach.
"""
            async with self.lock:
                print(f"[Verina] Verification #{current_verification}: Compilation FAILED after forcing code (attempt {len(self.failed_attempts)})")
                self.num_corrections += 1
                # Progressive thinking budget: give more room on retries
                self.k_steps = self.base_k_steps #+ (self.num_corrections * self.base_k_steps)
                if not event.is_set():
                    event_info["generated_text"] = step + complete_generated
                    event_info["feedback"] = feedback
                    event_info["correction_index"] = len(full_text)
                    event.set()
            return full_text, feedback
        
        # Compilation succeeded
        async with self.lock:
            print(f"[Verina] Verification #{current_verification}: Compilation SUCCESS after forcing code")
            self.success_found = True
            self.verified_code = code
        
        # Build complete generated text for success/judge cases too
        force_prompt = self._build_force_code_feedback()
        complete_generated = force_prompt + full_text

        success_feedback = f"""
<|im_end|>
<|im_start|>user
Your code compiled successfully! Now give the final answer
<|im_end|>
<|im_start|>assistant
<think> Good, the code I gave compiled successfully. Now I am confident in my answer, so I should output it in the required format.
"""
        print(f"[Verina] Injecting success feedback with verified code in user message")
        async with self.lock:
            if not event.is_set():
                event_info["generated_text"] = step + complete_generated
                event_info["feedback"] = success_feedback
                event.set()
        return full_text, success_feedback
    
        # No verification triggered
        return step, None

    async def fix(self, generated_text: str, event_info: dict, fix_method=None) -> str:
        """Append feedback to the generated text for retry.
        
        If feedback is None (successful verification), return the generated text as-is.
        """
        feedback = event_info.get("feedback")
        gen_text = event_info.get("generated_text", generated_text)
        
        if feedback is None:
            return gen_text
        return gen_text + feedback

    def step_extractor(self, chunk: str, generated_text: str) -> Tuple[bool, Optional[str]]:
        """
        Determine when to trigger verification.
        
        Triggers:
        1. Every K newlines without </think> → force code output
        
        Note: Final code verification is handled by verify_final_code() after generation completes.
        
        Returns: (should_verify, text_to_verify)
        """

        last_think_start = generated_text.rfind('<think>')
        last_think_end = generated_text.rfind('</think>')
        in_think_block = last_think_start > last_think_end

        if in_think_block:
            # Count newlines in the current think block
            current_think_content = generated_text[last_think_start + 7:]
            think_newlines = current_think_content.count('\n')
            
            # If this is a new think block, reset the tracking
            if last_think_start != self.last_triggered_think_start:
                self.last_triggered_think_start = last_think_start
                self.last_triggered_think_newlines = 0
            
            # Trigger if we've accumulated k_steps more newlines since last trigger
            if think_newlines >= self.last_triggered_think_newlines + self.k_steps:
                self.last_triggered_think_newlines = think_newlines  # Update to current count
                return True, generated_text

        return False, None
