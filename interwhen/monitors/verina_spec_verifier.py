"""
Step Verifier for Verina Specification Generation (Lean 4 precondition/postcondition generation).
"""

import asyncio
import json
import logging
import re
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import copy
import httpx
from interwhen.utils.verina_utils import *
from .base import VerifyMonitor

#Paths
VERINA_ROOT = (Path(__file__).parent.resolve() / ".." / ".." / ".." / "verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"


# Monitor
class StepVerifierVerinaSpecMonitor(VerifyMonitor):
    """
    Step-by-step verifier monitor for Verina Specification Generation.
    
    This monitor:
    1. Counts reasoning steps (newlines) during generation
    2. After K newlines, forces the model to output spec by streaming with </think> + [PRECOND]/[POSTCOND]
    3. When [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] are detected, extracts and verifies via Lean compilation
    4. If verification fails, injects feedback for retry
    """
    
    def __init__(
        self,
        name: str,
        task_data: dict,  # Contains signature, lean_data, description, data_id, tests, etc.
        llm_server: dict,  # LLM server config for forcing spec output
        prompt: str,  # The original prompt (needed for continuation)
        k_steps: int = 40,  # Number of newlines before forcing spec output
        max_corrections: int = 3,
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
        self.force_count = 0  # Track how many times spec was forced
        self.last_newline_count = 0
        self.last_verified_spec_end = 0  # Position of last [/POSTCOND] we verified
        self.success_found = False  # Once True, block all future failure feedback
        self.verified_spec = None  # Store the spec that compiled successfully
        self.last_triggered_think_start = -1
        self.last_triggered_think_newlines = 0
        
        # Diversity tracking: remember what failed so we can steer away from it
        self.failed_attempts = []  # List of (spec_snippet, error_summary) tuples
        self.base_k_steps = k_steps  # Original k_steps before progressive scaling

        # Lock for thread safety
        self.lock = asyncio.Lock()

    def reset(self):
        """Reset state for a new problem."""
        self.verification_count = 0
        self.force_count = 0
        self.last_newline_count = 0
        self.last_verified_spec_end = 0
        self.success_found = False
        self.verified_spec = None
        self.num_corrections = 0
        self.last_triggered_think_start = -1
        self.last_triggered_think_newlines = 0
        self.failed_attempts = []
        self.k_steps = self.base_k_steps
    
    def get_force_count(self) -> int:
        """Return the number of times spec was forced for this sample."""
        return self.force_count

    async def verify_final_spec(
        self, 
        spec: Dict[str, str], 
        prompt_with_answer: str, 
        max_retries: int = 3
    ) -> Tuple[Dict[str, str], bool, str, int]:
        """
        Verify extracted final spec and retry with LLM if compilation fails.
        Allows model to think before outputting corrected spec.
        
        Args:
            spec: Dict with keys: precond, postcond, precond_aux, postcond_aux
            prompt_with_answer: The full prompt + model answer (for continuing conversation)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (final_spec, compiles, compile_output, num_retries)
        """

        current_spec = spec.copy()
        current_prompt = prompt_with_answer
        num_retries = 0
        
        for attempt in range(max_retries):
            # Verify compilation
            compiles, compile_output = await self._verify_compilation(current_spec)
            
            if compiles:
                print(f"[VerinaSpec Final] Spec compiles successfully (attempt {attempt + 1})")
                return current_spec, True, compile_output, num_retries
            
            if attempt == max_retries:
                print(f"[VerinaSpec Final] Spec failed after {max_retries} retries, giving up")
                break
            
            num_retries += 1
            print(f"[VerinaSpec Final] Compilation failed (attempt {attempt + 1}), retrying...")
            
            # Build retry prompt with error feedback
            error_feedback = f"""
<|im_end|>
<|im_start|>user
The specification you gave failed with error:
{clean_compile_output(compile_output)}

Please fix the error and provide the corrected specification. Think through the problem carefully, then output your solution with [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] tags.

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
                payload["max_tokens"] = 2048
                payload["stream"] = False
                payload["stop"] = ["[/POSTCOND]", "</s>", "<|im_end|>"]
                
                headers = copy.deepcopy(self.llm_server.get("headers", {}))
                url = self.llm_server["url"]
                
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(url, headers=headers, json=payload)
                    result = response.json()
                    full_response = result["choices"][0]["text"].strip()
                    
                    # Extract spec from response
                    new_spec = extract_spec_from_response(full_response)
                    
                    if new_spec and new_spec.get("postcond"):
                        current_spec = new_spec
                        current_prompt = retry_prompt + full_response + "[/POSTCOND]"
                        print(f"[VerinaSpec Final] Got new spec...")
                    else:
                        print(f"[VerinaSpec Final] No spec found in response, keeping previous spec")
                        
            except Exception as e:
                print(f"[VerinaSpec Final] LLM retry failed: {e}")
                break
        
        return current_spec, False, compile_output, num_retries

    def _count_newlines(self, text: str) -> int:
        """Count the number of newlines in text."""
        return text.count('\n')

    _FORCE_SPEC_VARIANTS = [
        """\n\nWait, let me now output the specification as per my current understanding, so that the user can give feedback.
</think> The final specification is:
[PRECOND]""",
        """\n\nLet me try writing the specification now.
</think> Here is my specification:
[PRECOND]""",
        """\n\nOk let me take a step back and write the specification using a different approach than before.
</think> My approach:
[PRECOND]""",
        """\n\nI should try a completely different formulation this time.
</think> Alternative specification:
[PRECOND]""",
    ]

    def _build_force_spec_feedback(self) -> str:
        """Build the feedback string to force spec output. Rotates prompts for diversity."""
        idx = self.force_count % len(self._FORCE_SPEC_VARIANTS)
        return self._FORCE_SPEC_VARIANTS[idx]

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
The specification you gave failed with error:
{cleaned_error}

Please fix the error. Your specification should compile.

{LEAN4_API_REFERENCE}"""

        # 2+ failures: show history and demand a different approach
        prev_section = ""
        for i, (spec_snip, err_snip) in enumerate(self.failed_attempts[:-1], 1):
            prev_section += f"\n--- Failed Attempt {i} ---\n{spec_snip}\nError: {err_snip}\n"

        return f"""
<|im_end|>
<|im_start|>user
The specification you gave failed with error:
{cleaned_error}

You have now failed {n_failures} time(s). Here are your previous failed attempts:
{prev_section}
Do NOT repeat a similar approach. Use a fundamentally different formulation. Think step-by-step about a fresh solution before writing the specification.

{LEAN4_API_REFERENCE}"""

    async def _stream_force_spec(self, current_text: str, max_tokens: int = 2048) -> str:
        """
        Stream from LLM to force spec output.
        
        Appends the force-spec prompt and continues streaming until [/POSTCOND] is found
        or max_tokens reached.
        
        Args:
            current_text: The text generated so far (reasoning only, not including original prompt)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Complete text with forced spec (the new generated portion only)
        """
        force_prompt = self._build_force_spec_feedback()
        # Include original prompt + reasoning so far + force spec trigger
        full_prompt = self.prompt + current_text + force_prompt
        
        # Build the payload for the LLM call
        payload = copy.deepcopy(self.llm_server.get("payload", {}))
        payload["prompt"] = full_prompt
        payload["max_tokens"] = max_tokens
        
        headers = copy.deepcopy(self.llm_server.get("headers", {}))
        url = self.llm_server["url"]
        
        generated_text = ""
        
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
                        
                        # Stop when we see [/POSTCOND]
                        if "[/POSTCOND]" in generated_text.upper():
                            break
        
        return generated_text

    async def _verify_compilation(self, generated_spec: Dict[str, str]) -> Tuple[bool, str]:
        """Run Lean compilation check using existing helpers."""
        lean_content = build_spec_lean_content(self.task_data, generated_spec)
        task_id = self.task_data.get("data_id", "unknown")
        lean_file = create_lean_file(f"verify_spec_{task_id}_{self.verification_count}", lean_content)
        compiles, output = await asyncio.to_thread(
            check_lean_compile, lean_file, self.compile_timeout
        )
        return compiles, output

    def sync_verify_compilation(self, generated_spec: Dict[str, str]) -> Tuple[bool, str]:
        """Run Lean compilation check using existing helpers."""
        lean_content = build_spec_lean_content(self.task_data, generated_spec)
        task_id = self.task_data.get("data_id", "unknown")
        lean_file = create_lean_file(f"verify_spec_{task_id}_{self.verification_count}", lean_content)
        compiles, output = check_lean_compile(lean_file, self.compile_timeout)
        return compiles, output


    async def verify(self, step: str, token_index: int, event: asyncio.Event, event_info: dict):
        """
        Verify the generated text.
        
        Two modes:
        1. Force spec output: If K newlines reached and no spec yet, inject </think> + [PRECOND] prompt
        2. Verify spec: If [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] present, extract and compile
        
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

        async with self.lock:
            self.force_count += 1
        print(f"[VerinaSpec] Forcing spec output after {self._count_newlines(step)} newlines (force #{self.force_count})")
        
        # Stream from LLM to force spec output
        full_text = await self._stream_force_spec(step)
        full_text_with_precond = "[PRECOND]" + full_text
        generated_spec = extract_spec_from_response(full_text_with_precond)
        
        if not generated_spec.get("precond") and not generated_spec.get("postcond"):
            print("[VerinaSpec] Forced spec generation but could not extract spec")
            async with self.lock:
                if not event.is_set():
                    event_info["generated_text"] = step
                    event_info["feedback"] = None
            return step, None
        
        async with self.lock:
            self.verification_count += 1
            current_verification = self.verification_count
        
        # Verify the generated spec
        compiles, compile_output = await self._verify_compilation(generated_spec)
        
        if not compiles:
            # Track this failure for diversity
            spec_preview = f"precond: {generated_spec.get('precond', '')[:100]}... postcond: {generated_spec.get('postcond', '')[:100]}..."
            error_summary = clean_compile_output(compile_output)[:300]
            async with self.lock:
                self.failed_attempts.append((spec_preview, error_summary))
            
            # Build the complete text: original reasoning + force prompt + generated spec
            force_prompt = self._build_force_spec_feedback()
            complete_generated = force_prompt + full_text
            
            # Build escalating feedback based on how many times we've failed
            feedback = self._build_diversity_feedback(compile_output)
            feedback += """
<|im_end|>
<|im_start|>assistant
<think> It seems my specification failed to compile. I should analyze the error and try to fix it using a different approach.
"""
            async with self.lock:
                print(f"[VerinaSpec] Verification #{current_verification}: Compilation FAILED after forcing spec (attempt {len(self.failed_attempts)})")
                self.num_corrections += 1
                # Progressive thinking budget: give more room on retries
                self.k_steps = self.base_k_steps
                if not event.is_set():
                    event_info["generated_text"] = step + complete_generated
                    event_info["feedback"] = feedback
                    event_info["correction_index"] = len(full_text)
                    event.set()
            return full_text, feedback
        
        # Compilation succeeded — mark success
        print(f"[VerinaSpec] Verification #{current_verification}: Compilation SUCCESS after forcing spec")
        self.success_found = True
        self.verified_spec = generated_spec
        
        # Build complete generated text for success cases too
        force_prompt = self._build_force_spec_feedback()
        complete_generated = force_prompt + full_text

        success_feedback = f"""
<|im_end|>
<|im_start|>user
Your specification compiled successfully! Now give the final answer
<|im_end|>
<|im_start|>assistant
<think> Good, the specification I gave compiled successfully. Now I am confident in my answer, so I should output it in the required format.
"""
        print(f"[VerinaSpec] Injecting success feedback with verified spec in user message")
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
        
        Triggers: Every K newlines without </think> → force spec output
    
        Returns: (should_verify, text_to_verify)
        """
        last_think_start = generated_text.rfind('<think>')
        last_think_end = generated_text.rfind('</think>')
        in_think_block = last_think_start > last_think_end

        if in_think_block:
            # Count newlines only in the current think block
            current_think_content = generated_text[last_think_start + 7:]  # 7 = len('<think>')
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