"""
Step Verifier for Verina (Lean 4 code generation).

This monitor:
1. Counts reasoning steps (newlines) during generation
2. After K newlines, forces the model to output code by closing </think> and prompting [CODE]
3. Extracts code from [CODE]...[/CODE] tags
4. Verifies using Lean compilation (and optionally SLM judge)
5. If verification fails, injects feedback and lets the model retry
"""

import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional
import shutil
import httpx

from .base import VerifyMonitor


# Setup a dedicated logger for ALL verifier output - writes to separate file
logger = logging.getLogger("interwhen.monitors.verina_stepverifier")
logger.setLevel(logging.INFO)
logger.propagate = False  # Don't propagate to root logger (avoids mixing with main output)
_verina_handler = logging.FileHandler("verifier_output.log", mode="a")
_verina_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_verina_handler)

# Also add a simple stream logger for just the LLM chunks (no timestamps)
stream_logger = logging.getLogger("verina_stream")
stream_logger.setLevel(logging.INFO)
stream_logger.propagate = False
_stream_handler = logging.FileHandler("verifier_output.log", mode="a")
_stream_handler.setFormatter(logging.Formatter("%(message)s"))
stream_logger.addHandler(_stream_handler)

# ============== MODEL CONFIGURATION ==============
# Change these model names to scale experiments easily
MAIN_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# =================================================

# ============================================================================
# PATHS - Update these to point to your verina repo
# ============================================================================
# VERINA_ROOT = Path("../../verina")
VERINA_ROOT = (Path(__file__).parent.resolve() / ".." / ".." / ".." / "verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

# ============================================================================
# LEAN EVALUATION
# ============================================================================

def clean_playground():
    """Clean the lean playground directory"""
    for file in LEAN_PLAYGROUND_DIR.iterdir():
        if file.name != ".gitkeep":
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()


def create_lean_file(file_name: str, content: str) -> Path:
    """Create a lean file in the playground directory"""
    LEAN_PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)
    lean_file = LEAN_PLAYGROUND_DIR / f"{file_name}.lean"
    with open(lean_file, "w") as f:
        f.write(content)
    return lean_file


def check_lean_compile(lean_file: Path, timeout: int = 240) -> Tuple[bool, str]:
    """Check if the Lean file compiles successfully"""
    try:
        lean_file_abs = lean_file.resolve()
        verina_root_abs = VERINA_ROOT.resolve()
        result = subprocess.run(
            ["lake", "lean", str(lean_file_abs)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=verina_root_abs,
        )
        
        output = result.stdout.decode() + "\n" + result.stderr.decode()
        
        if result.returncode == 0:
            return True, output
        else:
            return False, output
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Lean compilation timed out for {lean_file}")
        return False, "TIMEOUT"
    except Exception as e:
        logger.error(f"Error during compilation: {e}")
        return False, f"ERROR: {e}"


def build_lean_content(task_data: dict, generated_code: str) -> str:
    """
    Build Lean file content from a task_data dict.
    
    Args:
        task_data: Dict with keys: signature, lean_data, data_id
        generated_code: The code to insert into the function body
    
    Returns:
        Complete Lean file content as string
    """
    signature = task_data.get("signature", {})
    lean_data = task_data.get("lean_data", {})
    
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)
    params = signature.get("parameters", [])
    param_names = " ".join([f"({p['param_name']})" for p in params])
    
    # Indent multiline code
    if '\n' in generated_code:
        lines = generated_code.split('\n')
        indented_lines = [lines[0]] + ['  ' + line if line.strip() else line for line in lines[1:]]
        generated_code = '\n'.join(indented_lines)
    
    # Build imports
    task_imports = lean_data.get("task_imports", "").strip()
    solution_imports = lean_data.get("solution_imports", "").strip()
    imports = task_imports
    if solution_imports:
        imports += "\n" + solution_imports
    if "import Mathlib" not in imports:
        imports = "import Mathlib\n" + imports
    
    # Build auxiliary definitions
    solution_aux = lean_data.get("solution_aux", "").strip()
    task_aux = lean_data.get("task_aux", "").strip()
    precond_aux = lean_data.get("precond_aux", "").strip()
    postcond_aux = lean_data.get("postcond_aux", "").strip()
    code_aux = lean_data.get("code_aux", "").strip()
    
    precond = lean_data.get("precond", "True").strip()
    postcond = lean_data.get("postcond", "").strip()
    precond_name = f"{func_name}_precond"
    postcond_name = f"{func_name}_postcond"
    
    return f"""{imports}

-- Solution auxiliary definitions
{solution_aux}

-- Task auxiliary definitions
{task_aux}

-- Precondition auxiliary definitions
{precond_aux}

@[reducible, simp]
def {precond_name} {param_list} : Prop :=
  {precond}

-- Postcondition auxiliary definitions
{postcond_aux}

-- Code auxiliary definitions
{code_aux}

def {func_name} {param_list} (h_precond : {precond_name} {param_names}) : {return_type} :=
  {generated_code}

@[reducible, simp]
def {postcond_name} {param_list} (result: {return_type}) (h_precond : {precond_name} {param_names}) : Prop :=
  {postcond}

-- Compilation check
#check {func_name}
"""

def build_code_gen_prompt(data: dict, code) -> Tuple[str, str]:
    """
    Build a simple prompt for Lean 4 code generation.
    Returns (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert Lean 4 programmer. Generate valid Lean 4 code for the function body. Wrap your final code in [CODE] [/CODE] tags strictly."""
    signature = data.get("signature", {})
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)
    params = signature.get("parameters", [])
    
    precond_name = f"{func_name}_precond"
    param_names_str = ' '.join([f"({p['param_name']})" for p in params])
    
    # Get auxiliary definitions (only if they exist)
    lean_data = data.get("lean_data", {})
    solution_aux = lean_data.get("solution_aux", "").strip()
    task_aux = lean_data.get("task_aux", "").strip()
    code_aux = lean_data.get("code_aux", "").strip()
    precond = lean_data.get("precond", "True").strip()
    postcond = lean_data.get("postcond", "").strip()
    
    # Build helper section only if there are helpers
    helper_section = ""
    all_aux = "\n".join(filter(None, [solution_aux, task_aux, code_aux]))
    if all_aux:
        helper_section = f"""
## Helper Definitions
```lean4
{all_aux}
```
"""
    
    user_prompt = f"""## Task
{data.description}

## FUNCTION TO EVALUATE
```lean4
def {func_name} {param_list} (h_precond : {precond_name} {param_names_str}) : {return_type} :=
  -- {code}
```

## Precondition
```lean4
def {precond_name} {param_list} : Prop := {precond}
```

## Postcondition  
```lean4
def {func_name}_postcond {param_list} (result: {return_type}) : Prop := {postcond}
```
{helper_section}
"""
    return system_prompt, user_prompt

# ============================================================================
# PROMPTING
# ============================================================================

def render_param_list(signature: dict) -> str:
    """Render the parameter list for a function signature"""
    params = signature.get("parameters", [])
    rendered = ""
    for param in params:
        rendered += f"({param['param_name']} : {param['param_type']}) "
    return rendered.strip()


# ============================================================================
# CODE EXTRACTION AND EVALUATION
# ============================================================================

def strip_function_definition(code: str) -> str:
    """
    Strip function definition prefix if the model accidentally included it.
    
    The prompt asks for just the function body, but sometimes the model outputs:
        def FunctionName (params) (h_precond : ...) : ReturnType :=
          actual_body
    
    We need to extract just 'actual_body' and dedent it properly.
    """
    import textwrap
    
    code = code.strip()
    
    # Pattern to match Lean function definition:
    # def <name> <params> (h_precond : <precond>) : <return_type> :=
    # The function body follows after :=
    func_def_pattern = r'^def\s+\w+\s+.*?:=[ \t]*\n?'
    
    match = re.match(func_def_pattern, code, re.DOTALL)
    if match:
        # Extract everything after the :=
        body = code[match.end():]
        # Dedent to remove common leading whitespace from all lines
        body = textwrap.dedent(body).strip()
        if body:
            return body
    
    return code


def extract_code_from_response(response: str) -> str:
    """Extract code from the LAST [CODE]...[/CODE] tags or lean code blocks.
    
    Handles cases where:
    1. Response has <think>...</think> reasoning block
    2. [CODE] tag exists but [/CODE] may be missing (truncated response)
    3. Code is in markdown lean blocks
    4. Model outputs [CORE] or other variants instead of [CODE]
    5. Model uses mismatched tags like [CORE]...[/CODE]
    6. Model includes full function definition instead of just the body
    """
    # Step 1: Remove <think>...</think> block entirely (case insensitive)
    # This prevents extracting reasoning text as code
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # If </think> exists but <think> doesn't match (partial), take everything after </think>
    if not cleaned.strip() or cleaned.strip() == response.strip():
        think_end = response.lower().rfind("</think>")
        if think_end != -1:
            cleaned = response[think_end + len("</think>"):]
    
    extracted_code = None
    
    # Step 2: Find the LAST closing tag [/CODE] or [/CORE] and work backwards to find opening tag
    # This handles mismatched tags like [CORE]...[/CODE]
    closing_pattern = r'\[/(?:CODE|CORE)\]'
    closing_matches = list(re.finditer(closing_pattern, cleaned, re.IGNORECASE))
    
    if closing_matches:
        # Get position of the last closing tag
        last_close = closing_matches[-1]
        close_pos = last_close.start()
        
        # Search backwards for the last opening tag before this closing tag
        text_before_close = cleaned[:close_pos]
        opening_pattern = r'\[(?:CODE|CORE|CORRECTED CODE)\]'
        opening_matches = list(re.finditer(opening_pattern, text_before_close, re.IGNORECASE))
        
        if opening_matches:
            last_open = opening_matches[-1]
            extracted_code = cleaned[last_open.end():close_pos].strip()
    
    # Step 3: Try [CODE] without closing tag (truncated response) - find the LAST one
    if extracted_code is None:
        code_start_matches = list(re.finditer(r'\[(?:CODE|CORE|CORRECTED CODE)\]\s*', cleaned, re.DOTALL | re.IGNORECASE))
        if code_start_matches:
            # Get the last [CODE] tag position and extract everything after it
            last_match = code_start_matches[-1]
            code = cleaned[last_match.end():].strip()
            # Remove any trailing incomplete text that looks like reasoning
            # Stop at any line that looks like it's not code (e.g., starts with "Wait", "So", etc.)
            lines = code.split('\n')
            code_lines = []
            for line in lines:
                stripped = line.strip()
                # Stop if we hit obvious non-code reasoning text
                if stripped and re.match(r'^(Wait|So |But |Now|Note|The |This |However|Therefore|Thus|In |Since)', stripped):
                    break
                code_lines.append(line)
            if code_lines:
                extracted_code = '\n'.join(code_lines).strip()
    
    # Step 4: Try markdown lean code blocks (find the LAST one)
    if extracted_code is None:
        lean_matches = list(re.finditer(r'```lean4?\s*\n(.*?)```', cleaned, re.DOTALL | re.IGNORECASE))
        if lean_matches:
            extracted_code = lean_matches[-1].group(1).strip()
    
    # Step 5: Try lean block without closing (truncated) - find the LAST one
    if extracted_code is None:
        lean_start_matches = list(re.finditer(r'```lean4?\s*\n', cleaned, re.DOTALL | re.IGNORECASE))
        if lean_start_matches:
            last_match = lean_start_matches[-1]
            code = cleaned[last_match.end():].strip()
            # Remove trailing ``` if present
            extracted_code = re.sub(r'```\s*$', '', code).strip()
    
    # Step 6: Last resort - return cleaned content if it looks like code
    if extracted_code is None:
        cleaned = cleaned.strip()
        if cleaned:
            # Filter out lines that look like reasoning
            lines = cleaned.split('\n')
            code_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not re.match(r'^(Wait|So |But |Now|Note|The |This |However|Therefore|Thus|In |Since|I |We |You )', stripped):
                    code_lines.append(line)
            if code_lines:
                extracted_code = '\n'.join(code_lines).strip()
    
    # Step 7: Strip function definition prefix if model included it
    # The prompt asks for just the body, but sometimes model outputs full "def ... :="
    if extracted_code:
        extracted_code = strip_function_definition(extracted_code)
        return extracted_code
    
    return ""

def clean_compile_output(output: str, errors_only: bool = True) -> str:
    """Strip file path and line/column numbers from Lean compiler output.
    
    Transforms:
        /long/path/to/file.lean:30:14: error: unexpected token '^'
    Into:
        error: unexpected token '^'
        
    If errors_only=True, filters out warnings and linter notes.
    """
    lines = []
    in_warning_block = False
    
    for line in output.splitlines():
        # Match pattern: /path/to/file.lean:line:col: message
        cleaned = re.sub(r'^[^\s]+\.lean:\d+:\d+:\s*', '', line)
        stripped = cleaned.strip()
        
        if not stripped:
            continue
            
        if errors_only:
            # Start of warning block
            if stripped.startswith('warning:'):
                in_warning_block = True
                continue
            # Error resets warning block
            if stripped.startswith('error:'):
                in_warning_block = False
            # Skip everything in warning block
            if in_warning_block:
                continue
            # Skip linter notes anywhere
            if stripped.startswith('Note:') or 'linter can be disabled' in stripped:
                continue
                
        lines.append(cleaned)
        
    return '\n'.join(lines).strip()


# ============================================================================
# LEAN 4 API REFERENCE (for error feedback)
# ============================================================================

LEAN4_API_REFERENCE = """## Lean 4 Quick Reference

### Functional Programming Basics
- Lean 4 is purely functional: no mutable state, no imperative loops
- Use recursion or higher-order functions (`foldl`, `map`, `filter`) instead of loops
- All values are immutable; "updating" creates new values

### Definitions and Syntax
- Use `:=` for definitions: `let x := 5`
- Match expressions don't use `end` keyword
- Helper functions inside a def: use `let rec helper := ...` not `def helper`

### Common Data Types
- `List α`: linked list, literal `[1, 2, 3]`
- `Array α`: mutable-like array, literal `#[1, 2, 3]`
- `Option α`: `some x` or `none`, use `Option.getD` for default
- `Nat`: natural numbers (≥0), `Int`: integers

### Numeric Operations  
- Arithmetic: `+`, `-`, `*`, `/`, `%` work as expected
- `^` is exponentiation, NOT bitwise XOR
- Bitwise ops are qualified: `Nat.xor`, `Int.land`, `Int.lor`, etc.
- Comparisons: `Nat.max`, `Nat.min`, `Int.max`, `Int.min`

### Collections
- List operations: `List.map`, `List.filter`, `List.foldl`, `List.foldr`
- List append: `xs ++ ys` or `xs ++ [x]`
- Array creation: `Array.mkArray n defaultVal` or `#[...]`
- Indexing: `xs[i]!` (unsafe), `xs.get? i` (returns Option)
- Ranges: `List.range n` gives `[0, 1, ..., n-1]`

### Recursion
- Lean requires termination proofs for recursive functions
- Add `termination_by` clause or use `partial def` if termination is complex
- Example: `termination_by xs => xs.length`

### Type Annotations
- Explicit types: `let x : Nat := 5`
- Lambda with types: `fun (x : Nat) => x + 1`
- Type inference works in most cases"""

# ============================================================================
# STEP VERIFIER VERINA MONITOR
# ============================================================================

class StepVerifierVerinaMonitor(VerifyMonitor):
    """
    Step-by-step verifier monitor for Verina (Lean 4 code generation).
    
    This monitor:
    1. Counts reasoning steps (newlines) during generation
    2. After K newlines, forces the model to output code by streaming with </think> + [CODE]
    3. When [CODE]...[/CODE] is detected, extracts and verifies via Lean compilation
    4. Optionally runs an SLM judge for semantic verification
    5. If verification fails, injects feedback for retry
    """
    
    def __init__(
        self,
        name: str,
        task_data: dict,  # Contains signature, lean_data, description, data_id, tests, etc.
        llm_server: dict,  # LLM server config for forcing code output
        prompt: str,  # The original prompt (needed for continuation)
        k_steps: int = 10,  # Number of newlines before forcing code output
        max_corrections: int = 3,
        use_slm_judge: bool = False,  # Whether to use SLM judge
        slm_server: Optional[dict] = None,  # LLM server config for SLM judge
        compile_timeout: int = 120,
        async_execution: bool = True,
    ):
        super().__init__(name)
        self.task_data = task_data
        self.llm_server = llm_server
        self.prompt = prompt
        self.k_steps = k_steps
        self.max_corrections = max_corrections
        self.use_slm_judge = use_slm_judge
        self.slm_server = slm_server
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
        import httpx
        import copy
        
        current_code = code
        current_prompt = prompt_with_answer
        num_retries = 0
        
        for attempt in range(max_retries):
            # Verify compilation
            compiles, compile_output = await self._verify_compilation(current_code)
            
            if compiles:
                print(f"[Verina Final] Code compiles successfully (attempt {attempt + 1})")
                return current_code, True, compile_output, num_retries
            
            # if attempt == max_retries:
            #     print(f"[Verina Final] Code failed after {max_retries} retries, giving up")
            #     break
            
            num_retries += 1
            print(f"[Verina Final] Compilation failed (attempt {attempt + 1}), retrying...")
            
            # Build retry prompt with error feedback - let the model think
            error_feedback = f"""</think>
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
            
            # Call LLM for retry - allow thinking with larger token budget
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

    # def _extract_code_from_response(self, response: str) -> Optional[str]:
    #     """Extract code from a response that may contain thinking and [CODE]...[/CODE] block."""
    #     # Try to find [CODE]...[/CODE] block (possibly incomplete due to stop sequence)
    #     code_match = re.search(r'\[CODE\](.*?)(?:\[/CODE\]|$)', response, re.DOTALL | re.IGNORECASE)
    #     if code_match:
    #         return code_match.group(1).strip()
        
    #     # If no [CODE] tag, check if response ends with code after </think>
    #     think_end = response.find('</think>')
    #     if think_end != -1:
    #         after_think = response[think_end + 8:].strip()
    #         # Remove any leading text before actual code
    #         if after_think:
    #             return after_think
        
    #     return None

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

    def _build_force_code_feedback(self) -> str:
        """Build the feedback string to force code output."""
        return """\n\nWait, let me now output the full code as per my current understanding, so that the user can give feedback.
</think> The final code is:
[CODE]"""

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
        
        # Build the payload for the LLM call - use copy to avoid race conditions
        import copy
        payload = copy.deepcopy(self.llm_server.get("payload", {}))
        payload["prompt"] = full_prompt
        # print("PROMPT: ",full_prompt)
        # logger.info(f"Force code prompt length: {len(full_prompt)} chars")
        payload["max_tokens"] = max_tokens
        
        # Copy headers too
        headers = copy.deepcopy(self.llm_server.get("headers", {}))
        url = self.llm_server["url"]
        
        # print(f"[Verina] Forcing code output via LLM stream...")
        # print(f"\n=== Forcing code (verification #{self.verification_count}) ===")
        
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
                        # stream_logger.info(chunk)  # Log each chunk to verifier_output.log
                        
                        # Stop when we see [/CODE]
                        if "[/CODE]" in generated_text:
                            # print(f"[Verina] Found [/CODE] - stopping stream")
                            break
        
        return generated_text

    async def _verify_compilation(self, code: str) -> Tuple[bool, str]:
        """Run Lean compilation check using existing helpers."""
        lean_content = build_lean_content(self.task_data, code)
        task_id = self.task_data.get("data_id", "unknown")
        lean_file = create_lean_file(f"verify_{task_id}_{self.verification_count}", lean_content)
        # print("file made")
        
        # Run compilation in thread to avoid blocking
        compiles, output = await asyncio.to_thread(
            check_lean_compile, lean_file, self.compile_timeout
        )
        return compiles, output

    def sync_verify_compilation(self, code: str) -> Tuple[bool, str]:
        """Run Lean compilation check using existing helpers."""
        lean_content = build_lean_content(self.task_data, code)
        task_id = self.task_data.get("data_id", "unknown")
        lean_file = create_lean_file(f"verify_{task_id}_{self.verification_count}", lean_content)
        # print("file made")
        
        # Run compilation in thread to avoid blocking
        compiles, output = check_lean_compile(lean_file, self.compile_timeout)
        return compiles, output


    async def _explain_compile_error(self, code: str, compile_output: str) -> str:
        """Use SLM to generate a short fix instruction for compile error."""
        if not self.slm_server:
            return clean_compile_output(compile_output)
        
        import httpx
        import copy
        
        cleaned_error = clean_compile_output(compile_output)
        # Fallback to raw output if cleaning removed everything
        if not cleaned_error.strip():
            cleaned_error = compile_output
        
        # Use completion-style prompt that starts the answer
        explain_prompt = f"""Your task is to provide the fix for a given Lean4 error. Be direct and clear. Avoid long outputs, keep it short and direct.
Lean 4 error: {cleaned_error}

Fix (In 1-2 sentences, no think tags):"""

        try:
            # Deep copy to avoid race conditions
            payload = copy.deepcopy(self.slm_server.get("payload", {}))
            payload["prompt"] = explain_prompt
            payload["max_tokens"] = 512
            payload["stream"] = False
            # payload["stop"] = ["\n\n", "Error:", "Lean"]  # Stop early
            
            headers = copy.deepcopy(self.slm_server.get("headers", {}))
            url = self.slm_server["url"]
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                result = response.json()
                fix_instruction = result["choices"][0]["text"].strip()
                # Clean up common preambles and take first useful sentence
                fix_instruction = fix_instruction.replace("We are given an error:", "").strip()
                fix_instruction = fix_instruction.replace("The error is:", "").strip()
                fix_instruction = fix_instruction.replace("To fix this:", "").strip()
                # Take only first line
                # fix_instruction = fix_instruction.split('\n')[0].strip()
                # Limit length
                # if len(fix_instruction) > 150:
                #     fix_instruction = fix_instruction[:150] + "..."
                # print(f"[Verina] SLM fix instruction: {fix_instruction}")
                return fix_instruction
        except Exception as e:
            logger.warning(f"SLM fix instruction failed: {e}")
            # Fallback: return cleaned error
            return f"Fix: {cleaned_error[:150]}"

    async def _verify_with_slm_judge(self, code: str) -> Tuple[bool, str]:
        """Use SLM as judge to verify code semantics after successful compilation.
        
        Provides the judge with full task context: description, signature, precondition,
        postcondition, and helper definitions from task_data.
        """
        if not self.slm_server:
            return True, "No SLM server configured"
        
        import httpx
        
        description = self.task_data.get("description", "")
        signature = self.task_data.get("signature", {})
        lean_data = self.task_data.get("lean_data", {})
        
        # Extract signature details
        func_name = signature.get("name", "solution")
        return_type = signature.get("return_type", "")
        param_list = render_param_list(signature)
        
        # Extract spec components
        precond = lean_data.get("precond", "True").strip()
        postcond = lean_data.get("postcond", "").strip()
        solution_aux = lean_data.get("solution_aux", "").strip()
        task_aux = lean_data.get("task_aux", "").strip()
        precond_aux = lean_data.get("precond_aux", "").strip()
        postcond_aux = lean_data.get("postcond_aux", "").strip()
        code_aux = lean_data.get("code_aux", "").strip()
        
        # Build helper defs section (only non-empty parts)
        helper_sections = []
        if solution_aux:
            helper_sections.append(f"-- Solution auxiliary:\n{solution_aux}")
        if task_aux:
            helper_sections.append(f"-- Task auxiliary:\n{task_aux}")
        if precond_aux:
            helper_sections.append(f"-- Precondition auxiliary:\n{precond_aux}")
        if postcond_aux:
            helper_sections.append(f"-- Postcondition auxiliary:\n{postcond_aux}")
        if code_aux:
            helper_sections.append(f"-- Code auxiliary:\n{code_aux}")
        helper_defs = "\n".join(helper_sections) if helper_sections else "(none)"

        _, content = build_code_gen_prompt(self.task_data, code)  # Get the user prompt part for context
        
        judge_prompt = f"""You are a Lean 4 code correctness judge. The code below has already compiled successfully. \
Your job is to judge whether the function body correctly satisfies the given precondition and postcondition.
The Function to evaluate is also marked below.

## Task Information
{content}

Answer ONLY with one of:
- "CORRECT" - if the implementation is semantically correct
- "INCORRECT: <brief specific reason>" - if the implementation is wrong

Assume there are no syntax-related issues since the code has already compiled. Focus solely on semantic correctness with respect to the specification.

Your judgment:"""

        try:
            # Deep copy to avoid race conditions
            import copy
            payload = copy.deepcopy(self.slm_server.get("payload", {}))
            payload["prompt"] = judge_prompt
            payload["max_tokens"] = 256
            payload["stream"] = False
            
            headers = copy.deepcopy(self.slm_server.get("headers", {}))
            url = self.slm_server["url"]
            
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload
                )
                result = response.json()
                judgment = result["choices"][0]["text"].strip()
                logger.info(f"[Verina] SLM judge response: {judgment}")
                print(f"[Verina] SLM judge result: {'CORRECT' if 'CORRECT' in judgment.upper() and 'INCORRECT' not in judgment.upper() else 'INCORRECT'}")
                
                is_correct = "CORRECT" in judgment.upper() and "INCORRECT" not in judgment.upper()
                return is_correct, judgment
                
        except Exception as e:
            logger.error(f"SLM judge verification failed: {e}")
            return True, f"Judge error (defaulting to pass): {e}"

#     async def verify(self, step: str, token_index: int, event: asyncio.Event, event_info: dict):
#         """
#         Verify the generated text.
        
#         Two modes:
#         1. Force code output: If K newlines reached and no code yet, inject </think> + [CODE] prompt
#         2. Verify code: If [CODE]...[/CODE] present, extract and compile
        
#         Args:
#             step: The text to verify (from step_extractor)
#             token_index: Current token index
#             event: asyncio.Event to signal when intervention needed
#             event_info: Dict to store feedback info
#         """
#         async with self.lock:
#             if self.num_corrections >= self.max_corrections:
#                 max_feedback = "\n\n</think> Let me now output the final code. Let me start\n[CODE]\n"
#                 if not event.is_set():
#                     event_info["generated_text"] = step
#                     event_info["feedback"] = max_feedback
#                     event_info["correction_index"] = token_index
#                     event_info["errors"] = ["Max corrections reached"]
#                     event_info["failed_step"] = None
#                     event.set()
#                 return step, max_feedback

#         # MODE 1: Force code output (triggered by step_extractor when K newlines reached)
#         async with self.lock:
#             self.num_corrections += 1
#             self.force_count += 1
#             print(f"[Verina] Forcing code output after {self._count_newlines(step)} newlines (force #{self.force_count})")
        
#         # Actually stream from LLM to force code output
#         full_text = await self._stream_force_code(step)
#         # print(f"[Verina] Forced code output:\n{full_text}")
#         # Now verify the generated code
#         code = extract_code_from_response(full_text).replace("[/CODE]","")
#         async with self.lock:
#             print("--------------------------------")
#             print("EXTRACTED FORCED CODE: ",code)
#             print("--------------------------------")
        
#         if not code:
#             print("[Verina] Forced code generation but could not extract code")
#             # Still set event to let interject.py handle retry
#             async with self.lock:
#                 if not event.is_set():
#                     event_info["generated_text"] = step
#                     event_info["feedback"] = None
#                     # event_info["correction_index"] = len(step)
#                     event_info["mode"] = "no_code_extracted"
#                     # event.set()
#             return step, None
        
#         # Verify the forced code
#         async with self.lock:
#             self.verification_count += 1
#             current_verification = self.verification_count
#         # print(f"[Verina] Verification #{current_verification} (after forcing code)")
#         compiles, compile_output = await self._verify_compilation(code)

#         if not compiles:
#             # Once success is found, block all future failure feedback
#             if self.success_found:
#                 print(f"[Verina] Verification #{current_verification}: Compilation FAILED but SUCCESS already found - skipping")
#                 return full_text, None
            
#             # Build the complete text: original reasoning + force prompt + generated code
#             force_prompt = self._build_force_code_feedback()
#             complete_generated = force_prompt + full_text
            
#             # Get SLM fix instruction
#             # fix_instruction = await self._explain_compile_error(code, compile_output)
            
#             # Build feedback with FULL error trace + fix suggestion
#             feedback = f"""</think>
# <|im_end|>
# <|im_start|>user
# The code you gave failed with error:
# {clean_compile_output(compile_output)}

# Please fix the error. Your code should compile.

# {LEAN4_API_REFERENCE}
# <|im_end|>
# <|im_start|>assistant
# <think>
# """
#             # print(f"[Verina] Verification #{current_verification}: Compilation FAILED after forcing code")
#             async with self.lock:
#                 print(f"[Verina] Verification #{current_verification}: Compilation FAILED after forcing code")
#                 if not event.is_set():
#                     event_info["generated_text"] = step + complete_generated
#                     event_info["feedback"] = feedback
#                     event_info["correction_index"] = len(full_text)
#                     event_info["mode"] = "compile_error"
#                     event_info["compile_output"] = compile_output
#                     event.set()
#             return full_text, feedback
        
#         # Compilation succeeded — mark success and give user message
#         async with self.lock:
#             print(f"[Verina] Verification #{current_verification}: Compilation SUCCESS after forcing code")
#         self.success_found = True
#         self.verified_code = code
        
#         # Build complete generated text for success/judge cases too
#         force_prompt = self._build_force_code_feedback()
#         complete_generated = force_prompt + full_text
        
#         if self.use_slm_judge and self.slm_server:
#             print(f"[Verina] Running SLM judge...")
#             is_correct, judgment = await self._verify_with_slm_judge(code)
#             if not is_correct:
#                 feedback = f"""[/CODE]
# </think>
# <|im_end|>
# <|im_start|>user
# Code compiled but has a logic issue: {judgment}
# <|im_end|>
# <|im_start|>assistant
# <think>
# """
#                 print(f"[Verina] Verification #{current_verification}: SLM judge FAILED - {judgment}")
#                 async with self.lock:
#                     if not event.is_set():
#                         event_info["generated_text"] = step + complete_generated
#                         event_info["feedback"] = feedback
#                         event_info["correction_index"] = len(step)
#                         event_info["mode"] = "judge_error"
#                         event_info["judgment"] = judgment
#                         event.set()
#                 return full_text, feedback
#             print(f"[Verina] SLM judge: CORRECT")
        
#         # Success: include verified code in user message
#         success_feedback = f"""
# </think>
# <|im_end|>
# <|im_start|>user
# Your code compiled successfully! Now give the final answer
# <|im_end|>
# <|im_start|>assistant
# <think> Good, the code I gave compiled successfully. Now I am confident in my answer, so I should output it in the required format.
# </think>
# The final answer is:
# [CODE]
# """
#         print(f"[Verina] Injecting success feedback with verified code in user message")
#         async with self.lock:
#             if not event.is_set():
#                 event_info["generated_text"] = step + complete_generated
#                 event_info["feedback"] = success_feedback
#                 event_info["mode"] = "success_locked"
#                 event_info["verified_code"] = code
#                 event.set()
#         return full_text, success_feedback
        
#         # MODE 2: Verify code (triggered when [CODE]...[/CODE] is present)
# #         if self._has_complete_code_block(step):
# #             code = extract_code_from_response(step)
            
# #             if not code:
# #                 logger.warning("[Verina] Could not extract code from response")
# #                 return step, None
            
# #             self.verification_count += 1
# #             logger.info(f"[Verina] Verification #{self.verification_count}")
# #             logger.info(f"[Verina] Extracted code:\n{code}...")
            
# #             # Step 1: Compilation check
# #             compiles, compile_output = await self._verify_compilation(code)
            
# #             if not compiles:
# #                 # Use SLM to explain the error if available
# #                 # error_explanation = await self._explain_compile_error(code, compile_output)
# #                 force_prompt = self._build_force_code_feedback()
# #                 complete_generated = force_prompt + full_text
                
# #                 # Get SLM fix instruction
# #                 # fix_instruction = await self._explain_compile_error(code, compile_output)
                
# #                 # Build feedback with FULL error trace + fix suggestion
# #                 feedback = f"""
# # <|im_end|>
# # <|im_start|>user
# # The code you gave failed with error:
# # {compile_output}

# # Please fix this error. Your code should compile.
# # <|im_end|>
# # <|im_start|>assistant
# # <think>
# # """
                
# #                 logger.info(f"[Verina] Compilation FAILED")
# #                 async with self.lock:
# #                     if not event.is_set():
# #                         event_info["generated_text"] = step
# #                         event_info["feedback"] = feedback
# #                         event_info["correction_index"] = token_index
# #                         event_info["mode"] = "compile_error"
# #                         event_info["compile_output"] = compile_output
# #                         event.set()
                
# #                 return step, feedback
            
# #             logger.info(f"[Verina] Compilation SUCCESS")
            
# #             # Step 2: Optional SLM judge verification
# #             if self.use_slm_judge and self.slm_server:
# #                 is_correct, judgment = await self._verify_with_slm_judge(code, compile_output)
                
# #                 if not is_correct:
# #                     feedback = f"\n[/CODE]\n\n[VERIFIER FEEDBACK - Semantic Error]\n{judgment}\n\nPlease provide corrected code:\n[CORRECTED CODE]"
                    
# #                     logger.info(f"[Verina] SLM Judge: INCORRECT - {judgment}")
# #                     async with self.lock:
# #                         if not event.is_set():
# #                             event_info["generated_text"] = step
# #                             event_info["feedback"] = feedback
# #                             event_info["correction_index"] = token_index
# #                             event_info["mode"] = "judge_error"
# #                             event_info["judgment"] = judgment
# #                             event.set()
                    
# #                     return step, feedback
                
# #                 logger.info(f"[Verina] SLM Judge: CORRECT")
            
# #             # All checks passed!
# #             logger.info(f"[Verina] All verification checks PASSED!")
# #             return step, None
        
#         # No action needed
#         print("NO VERIFICATION TRIGGERED")
#         return step, None

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
                    event_info["errors"] = ["Max corrections reached"]
                    event.set()
                return step, max_feedback

        # Force code output (triggered by step_extractor)
        async with self.lock:
            self.force_count += 1
            print(f"[Verina] Forcing code output after {self._count_newlines(step)} newlines (force #{self.force_count})")
        
        full_text = await self._stream_force_code(step)
        full_text_with_code = "[CODE]" + full_text
        # Now verify the generated code
        code = extract_code_from_response(full_text_with_code)
        # Now verify the generated code
        # code = extract_code_from_response(full_text).replace("[/CODE]","")
        
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
            # Build the complete text
            force_prompt = self._build_force_code_feedback()
            complete_generated = force_prompt + full_text
            
            # Build feedback
            feedback = f"""</think>
<|im_end|>
<|im_start|>user
The code you gave failed with error:
{clean_compile_output(compile_output)}

Please fix the error. Your code should compile.

{LEAN4_API_REFERENCE}
<|im_end|>
<|im_start|>assistant
<think>It seems my code failed to compile. I should analyze the error and try to fix it.
"""
            async with self.lock:
                print(f"[Verina] Verification #{current_verification}: Compilation FAILED after forcing code")
                self.num_corrections += 1
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
</think>
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
                event_info["mode"] = "success_locked"
                event_info["verified_code"] = code
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
        # current_newlines = self._count_newlines(generated_text)
        # last_think_start = generated_text.rfind('<think>')
        # last_think_end = generated_text.rfind('</think>')
        # in_think_block = last_think_start > last_think_end

        # if in_think_block:
        #     if(current_newlines//self.k_steps > self.last_newline_count//self.k_steps) and current_newlines > 0:
        #         self.last_newline_count = current_newlines
        #         # print("FOUND STEP:")
        #         # logger.info(f"[Verina] step_extractor: {generated_text}")
        #         return True, generated_text
        
        # return False, None
        # last_think_start = generated_text.rfind('<think>')
        # last_think_end = generated_text.rfind('</think>')
        # in_think_block = last_think_start > last_think_end

        # if in_think_block:
        #     # Count newlines only in the CURRENT think block
        #     current_think_content = generated_text[last_think_start + 7:]  # 7 = len('<think>')
        #     think_newlines = current_think_content.count('\n')
            
        #     # If this is a new think block, reset the tracking
        #     if last_think_start != self.last_triggered_think_start:
        #         self.last_triggered_think_start = last_think_start
        #         self.last_triggered_think_newlines = 0
            
        #     # Trigger if we've accumulated k_steps MORE newlines since last trigger
        #     if think_newlines >= self.last_triggered_think_newlines + self.k_steps:
        #         self.last_triggered_think_newlines = think_newlines  # Update to current count
        #         return True, generated_text

        # return False, None
        current_newlines = self._count_newlines(generated_text)
        last_think_start = generated_text.rfind('<think>')
        last_think_end = generated_text.rfind('</think>')
        in_think_block = last_think_start > last_think_end

        if in_think_block:
            # Skip if we already triggered for this think block
            if last_think_start == self.last_triggered_think_start:
                return False, None
            
            # Count newlines only in the CURRENT think block
            current_think_content = generated_text[last_think_start + 7:]  # 7 = len('<think>')
            think_newlines = current_think_content.count('\n')
            
            if think_newlines >= self.k_steps:
                self.last_triggered_think_start = last_think_start  # Mark this think block as triggered
                return True, generated_text

        return False, None
