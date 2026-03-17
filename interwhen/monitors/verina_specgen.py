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
from .base import VerifyMonitor


# Setup a dedicated logger for all verifier output
logger = logging.getLogger("interwhen.monitors.verina_specgen")
logger.setLevel(logging.INFO)
logger.propagate = False
_verina_handler = logging.FileHandler("specgen_verifier_output.log", mode="a")
_verina_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_verina_handler)

# Also add a simple stream logger for just the LLM chunks (no timestamps)
stream_logger = logging.getLogger("verina_spec_stream")
stream_logger.setLevel(logging.INFO)
stream_logger.propagate = False
_stream_handler = logging.FileHandler("specgen_verifier_output.log", mode="a")
_stream_handler.setFormatter(logging.Formatter("%(message)s"))
stream_logger.addHandler(_stream_handler)

# Modify model names as required
MAIN_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

LEAN4_SPEC_API_REFERENCE = """## Lean 4 Quick Reference

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

#Paths
VERINA_ROOT = (Path(__file__).parent.resolve() / ".." / ".." / ".." / "verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

# Lean Evaluation Helpers
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

def check_lean_compile(lean_file: Path, timeout: int = 120) -> Tuple[bool, str]:
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


def render_param_list(signature: dict) -> str:
    """Render the parameter list for a function signature"""
    params = signature.get("parameters", [])
    rendered = ""
    for param in params:
        rendered += f"({param['param_name']} : {param['param_type']}) "
    return rendered.strip()


def build_spec_lean_content(task_data: dict, generated_spec: Dict[str, str]) -> str:
    """
    Build Lean file content for spec verification.
    
    Args:
        task_data: Dict with keys: signature, lean_data, data_id
        generated_spec: Dict with keys: precond, postcond, precond_aux, postcond_aux
    
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
    
    # Build imports
    task_imports = lean_data.get("task_imports", "").strip()
    solution_imports = lean_data.get("solution_imports", "").strip()
    imports = task_imports
    if solution_imports:
        imports += "\n" + solution_imports
    if "import Mathlib" not in imports:
        imports = "import Mathlib\n" + imports
    if "import Plausible" not in imports:
        imports = "import Plausible\n" + imports
    
    # Build auxiliary definitions from task
    task_aux = lean_data.get("task_aux", "").strip()
    solution_aux = lean_data.get("solution_aux", "").strip()
    
    # Get generated spec components
    precond = generated_spec.get("precond", "True").strip()
    postcond = generated_spec.get("postcond", "").strip()
    precond_aux = generated_spec.get("precond_aux", "").strip()
    postcond_aux = generated_spec.get("postcond_aux", "").strip()
    
    # Make aux reducible if provided
    if precond_aux:
        precond_aux = make_aux_reducible(precond_aux)
    if postcond_aux:
        postcond_aux = make_aux_reducible(postcond_aux)
    
    precond_name = f"{func_name}_precond"
    postcond_name = f"{func_name}_postcond"
    
    return f"""{imports}

-- Task auxiliary definitions
{task_aux}

-- Solution auxiliary definitions
{solution_aux}

-- Generated precondition auxiliary
{precond_aux}

@[reducible, simp]
def {precond_name} {param_list} : Prop :=
  {precond}

-- Generated postcondition auxiliary
{postcond_aux}

@[reducible, simp]
def {postcond_name} {param_list} (result: {return_type}) (h_precond : {precond_name} {param_names}) : Prop :=
  {postcond}

-- Compilation check
#check {precond_name}
#check {postcond_name}
"""


def make_aux_reducible(aux: str) -> str:
    """Add @[reducible, simp] to definitions if not present"""
    lines = aux.split("\n")
    result = []
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            if i == 0 or "@[reducible, simp]" not in lines[i-1]:
                result.append("@[reducible, simp]")
        result.append(line)
    return "\n".join(result)


def build_spec_gen_prompt(data: dict) -> Tuple[str, str]:
    """
    Build a prompt for Lean 4 specification generation.
    Returns (system_prompt, user_prompt)
    """
    system_prompt = """You are an expert Lean 4 programmer specializing in formal specifications.
Generate valid Lean 4 preconditions and postconditions for the function described.

The precondition should:
- Be as permissive as possible while ensuring the function can execute correctly
- Capture constraints on input values that are necessary for correct execution

The postcondition should:
- Be sound: Only accept correct outputs (reject any incorrect output)
- Be complete: Accept all correct outputs (don't reject valid solutions)
- Fully specify the relationship between inputs and the expected output

Wrap your precondition in [PRECOND]...[/PRECOND] tags.
Wrap your postcondition in [POSTCOND]...[/POSTCOND] tags.
If you need auxiliary definitions for precondition, wrap them in [PRECOND_AUX]...[/PRECOND_AUX] tags.
If you need auxiliary definitions for postcondition, wrap them in [POSTCOND_AUX]...[/POSTCOND_AUX] tags.
"""

    signature = data.get("signature", {})
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)
    params = signature.get("parameters", [])
    param_names_str = ' '.join([f"({p['param_name']})" for p in params])
    
    lean_data = data.get("lean_data", {})
    description = data.get("description", "")
    
    # Get ground truth code to show (spec is generated, not code)
    code = lean_data.get("code", "").strip()
    code_aux = lean_data.get("code_aux", "").strip()
    task_aux = lean_data.get("task_aux", "").strip()
    
    # Natural language spec descriptions if available
    spec_desc = data.get("spec_desc", {})
    precond_desc = spec_desc.get("precond_desc", "")
    postcond_desc = spec_desc.get("postcond_desc", "")
    
    spec_desc_section = ""
    if precond_desc or postcond_desc:
        spec_desc_section = f"""
## Specification Hints
Precondition: {precond_desc if precond_desc else "Derive from task description"}
Postcondition: {postcond_desc if postcond_desc else "Derive from task description"}
"""
    
    helper_section = ""
    if task_aux or code_aux:
        all_aux = "\n".join(filter(None, [task_aux, code_aux]))
        helper_section = f"""
## Helper Definitions
```lean4
{all_aux}
```
"""

    code_section = ""
    if code:
        code_section = f"""
## Reference Implementation
```lean4
def {func_name} {param_list} (h_precond : {func_name}_precond {param_names_str}) : {return_type} :=
  {code}
```
"""
    
    user_prompt = f"""## Task
{description}

## Function Signature
- Function name: {func_name}
- Parameters: {param_list}
- Return type: {return_type}

## Expected Output Format
```lean4
-- Precondition auxiliary (optional)
[PRECOND_AUX]
-- helper definitions for precondition
[/PRECOND_AUX]

-- Precondition: when should the function be allowed to run?
def {func_name}_precond {param_list} : Prop :=
  [PRECOND]
  -- your precondition here (e.g., True, or constraints on inputs)
  [/PRECOND]

-- Postcondition auxiliary (optional)  
[POSTCOND_AUX]
-- helper definitions for postcondition
[/POSTCOND_AUX]

-- Postcondition: what must be true about the result?
def {func_name}_postcond {param_list} (result: {return_type}) (h_precond : {func_name}_precond {param_names_str}) : Prop :=
  [POSTCOND]
  -- your postcondition here
  [/POSTCOND]
```
{spec_desc_section}{helper_section}{code_section}
Generate the precondition and postcondition. Use [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] tags."""

    return system_prompt, user_prompt

# def extract_spec_from_response(response: str) -> Dict[str, str]:
#     """Extract precondition and postcondition from response.
    
#     Returns dict with keys: precond, postcond, precond_aux, postcond_aux
#     """
#     result = {
#         "precond": "True",  # Default
#         "postcond": "",
#         "precond_aux": "",
#         "postcond_aux": "",
#     }
    
#     # Remove <think>...</think> block
#     cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    
#     # Handle partial </think>
#     if not cleaned.strip() or cleaned.strip() == response.strip():
#         think_end = response.lower().rfind("</think>")
#         if think_end != -1:
#             cleaned = response[think_end + len("</think>"):]
    
#     # Extract PRECOND_AUX
#     precond_aux_match = re.search(r'\[PRECOND_AUX\](.*?)\[/PRECOND_AUX\]', cleaned, re.DOTALL | re.IGNORECASE)
#     if precond_aux_match:
#         result["precond_aux"] = precond_aux_match.group(1).strip()
    
#     # Extract POSTCOND_AUX
#     postcond_aux_match = re.search(r'\[POSTCOND_AUX\](.*?)\[/POSTCOND_AUX\]', cleaned, re.DOTALL | re.IGNORECASE)
#     if postcond_aux_match:
#         result["postcond_aux"] = postcond_aux_match.group(1).strip()
    
#     # Extract PRECOND
#     precond_match = re.search(r'\[PRECOND\](.*?)\[/PRECOND\]', cleaned, re.DOTALL | re.IGNORECASE)
#     if precond_match:
#         result["precond"] = precond_match.group(1).strip()
#     else:
#         # Try without closing tag (truncated)
#         precond_start_match = re.search(r'\[PRECOND\]\s*(.*)', cleaned, re.DOTALL | re.IGNORECASE)
#         if precond_start_match:
#             precond = precond_start_match.group(1).strip()
#             # Stop at POSTCOND tag if present
#             postcond_idx = precond.lower().find("[postcond")
#             if postcond_idx != -1:
#                 precond = precond[:postcond_idx].strip()
#             result["precond"] = precond if precond else "True"
    
#     # Extract POSTCOND
#     postcond_match = re.search(r'\[POSTCOND\](.*?)\[/POSTCOND\]', cleaned, re.DOTALL | re.IGNORECASE)
#     if postcond_match:
#         result["postcond"] = postcond_match.group(1).strip()
#     else:
#         # Try without closing tag
#         postcond_start_match = re.search(r'\[POSTCOND\]\s*(.*)', cleaned, re.DOTALL | re.IGNORECASE)
#         if postcond_start_match:
#             postcond = postcond_start_match.group(1).strip()
#             result["postcond"] = postcond
    
#     # Clean up any remaining tags in the extracted content
#     for key in result:
#         result[key] = re.sub(r'```(lean4?)?\s*', '', result[key])
#         result[key] = re.sub(r'```\s*$', '', result[key])
    
#     return result

def extract_spec_from_response(response: str) -> Dict[str, str]:
    """Extract precondition and postcondition from response.
    
    Returns dict with keys: precond, postcond, precond_aux, postcond_aux
    """
    result = {
        "precond": "",
        "postcond": "",
        "precond_aux": "",
        "postcond_aux": "",
    }
    
    # Remove <think>...</think> block
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # Handle partial </think>
    if not cleaned.strip() or cleaned.strip() == response.strip():
        think_end = response.lower().rfind("</think>")
        if think_end != -1:
            cleaned = response[think_end + len("</think>"):]
    
    # Extract PRECOND_AUX (take LAST match to get the most recent/corrected version)
    precond_aux_matches = re.findall(r'\[PRECOND_AUX\](.*?)\[/PRECOND_AUX\]', cleaned, re.DOTALL | re.IGNORECASE)
    if precond_aux_matches:
        result["precond_aux"] = precond_aux_matches[-1].strip()
    
    # Extract POSTCOND_AUX (take LAST match)
    postcond_aux_matches = re.findall(r'\[POSTCOND_AUX\](.*?)\[/POSTCOND_AUX\]', cleaned, re.DOTALL | re.IGNORECASE)
    if postcond_aux_matches:
        result["postcond_aux"] = postcond_aux_matches[-1].strip()
    
    # Extract PRECOND (take LAST match to get the most recent/corrected version)
    precond_matches = re.findall(r'\[PRECOND\](.*?)\[/PRECOND\]', cleaned, re.DOTALL | re.IGNORECASE)
    if precond_matches:
        result["precond"] = precond_matches[-1].strip()
    else:
        precond_start_match = re.search(r'\[PRECOND\]\s*(.*)', cleaned, re.DOTALL | re.IGNORECASE)
        if precond_start_match:
            precond = precond_start_match.group(1).strip()
            postcond_idx = precond.lower().find("[postcond")
            if postcond_idx != -1:
                precond = precond[:postcond_idx].strip()
            result["precond"] = precond
    
    # Extract POSTCOND (take LAST match to get the most recent/corrected version)
    postcond_matches = re.findall(r'\[POSTCOND\](.*?)\[/POSTCOND\]', cleaned, re.DOTALL | re.IGNORECASE)
    if postcond_matches:
        result["postcond"] = postcond_matches[-1].strip()
    else:
        postcond_start_match = re.search(r'\[POSTCOND\]\s*(.*)', cleaned, re.DOTALL | re.IGNORECASE)
        if postcond_start_match:
            postcond = postcond_start_match.group(1).strip()
            result["postcond"] = postcond
    
    # Clean up any remaining markdown tags
    for key in result:
        result[key] = re.sub(r'```(lean4?)?\s*', '', result[key])
        result[key] = re.sub(r'```\s*$', '', result[key])
    
    return result

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

# Monitor
class StepVerifierVerinaSpecMonitor(VerifyMonitor):
    """
    Step-by-step verifier monitor for Verina Specification Generation.
    
    This monitor:
    1. Counts reasoning steps (newlines) during generation
    2. After K newlines, forces the model to output spec by streaming with </think> + [PRECOND]/[POSTCOND]
    3. When [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] are detected, extracts and verifies via Lean compilation
    4. Optionally runs an SLM judge for semantic verification
    5. If verification fails, injects feedback for retry
    """
    
    def __init__(
        self,
        name: str,
        task_data: dict,  # Contains signature, lean_data, description, data_id, tests, etc.
        llm_server: dict,  # LLM server config for forcing spec output
        prompt: str,  # The original prompt (needed for continuation)
        k_steps: int = 40,  # Number of newlines before forcing spec output
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
        self.force_count = 0  # Track how many times spec was forced
        self.last_newline_count = 0
        self.last_verified_spec_end = 0  # Position of last [/POSTCOND] we verified
        self.success_found = False  # Once True, block all future failure feedback
        self.verified_spec = None  # Store the spec that compiled successfully
        self.last_triggered_think_start = -1
        self.last_triggered_think_newlines = 0

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
            
            # Build retry prompt with error feedback - let the model think
            error_feedback = f"""</think>
<|im_end|>
<|im_start|>user
The specification you gave failed with error:
{clean_compile_output(compile_output)}

Please fix the error and provide the corrected specification. Think through the problem carefully, then output your solution with [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] tags.

{LEAN4_SPEC_API_REFERENCE}
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
                    new_spec = self._extract_spec_from_response(full_response)
                    
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

    def _extract_spec_from_response(self, response: str) -> Optional[Dict[str, str]]:
        """Extract spec from a response that may contain thinking and spec blocks."""
        # Try to extract using the standard function
        spec = extract_spec_from_response(response)
        if spec.get("postcond"):
            return spec
        
        # If no spec found, check if there's content after </think>
        think_end = response.find('</think>')
        if think_end != -1:
            after_think = response[think_end + 8:]
            spec = extract_spec_from_response(after_think)
            if spec.get("postcond"):
                return spec
        
        return None

    def _count_newlines(self, text: str) -> int:
        """Count the number of newlines in text."""
        return text.count('\n')
    

    def _build_force_spec_feedback(self) -> str:
        """Build the feedback string to force spec output."""
        return """\n\nWait, let me now output the specification as per my current understanding, so that the user can give feedback.
</think> The final specification is:
[PRECOND]"""

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


    async def _verify_with_slm_judge(self, generated_spec: Dict[str, str]) -> Tuple[bool, str]:
        """Use SLM as judge to verify spec semantics after successful compilation.
        
        Provides the judge with full task context: description, signature, and 
        the generated precondition/postcondition.
        """
        if not self.slm_server:
            return True, "No SLM server configured"
        
        
        description = self.task_data.get("description", "")
        signature = self.task_data.get("signature", {})
        lean_data = self.task_data.get("lean_data", {})
        
        # Extract signature details
        func_name = signature.get("name", "solution")
        return_type = signature.get("return_type", "")
        param_list = render_param_list(signature)
        
        # Extract task auxiliary definitions
        task_aux = lean_data.get("task_aux", "").strip()
        solution_aux = lean_data.get("solution_aux", "").strip()
        
        # Build helper defs section (only non-empty parts)
        helper_sections = []
        if solution_aux:
            helper_sections.append(f"-- Solution auxiliary:\n{solution_aux}")
        if task_aux:
            helper_sections.append(f"-- Task auxiliary:\n{task_aux}")
        helper_defs = "\n".join(helper_sections) if helper_sections else "(none)"
        
        # Get generated spec components
        precond = generated_spec.get("precond", "True")
        postcond = generated_spec.get("postcond", "")
        precond_aux = generated_spec.get("precond_aux", "")
        postcond_aux = generated_spec.get("postcond_aux", "")
        
        judge_prompt = f"""You are a Lean 4 specification correctness judge. The specification below has already compiled successfully.
Your job is to judge whether the precondition and postcondition correctly capture the requirements.

## Task Description
{description}

## Function Signature
- Function name: {func_name}
- Parameters: {param_list}
- Return type: {return_type}

## Helper Definitions
{helper_defs}

## Generated Precondition
{precond_aux if precond_aux else "(no auxiliary)"}
{precond}

## Generated Postcondition
{postcond_aux if postcond_aux else "(no auxiliary)"}
{postcond}

Answer ONLY with one of:
- "CORRECT" - if the specification is semantically correct
- "INCORRECT: <brief specific reason>" - if the specification is wrong

Assume there are no syntax-related issues since the spec has already compiled. Focus solely on semantic correctness with respect to the task description.

Your judgment:"""

        try:
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
                logger.info(f"[VerinaSpec] SLM judge response: {judgment}")
                print(f"[VerinaSpec] SLM judge result: {'CORRECT' if 'CORRECT' in judgment.upper() and 'INCORRECT' not in judgment.upper() else 'INCORRECT'}")
                
                is_correct = "CORRECT" in judgment.upper() and "INCORRECT" not in judgment.upper()
                return is_correct, judgment
                
        except Exception as e:
            logger.error(f"SLM judge verification failed: {e}")
            return True, f"Judge error (defaulting to pass): {e}"

#     async def verify(self, step: str, token_index: int, event: asyncio.Event, event_info: dict):
#         """
#         Verify the generated text.
        
#         Two modes:
#         1. Force spec output: If K newlines reached and no spec yet, inject </think> + [PRECOND] prompt
#         2. Verify spec: If [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] present, extract and compile
        
#         Args:
#             step: The text to verify (from step_extractor)
#             token_index: Current token index
#             event: asyncio.Event to signal when intervention needed
#             event_info: Dict to store feedback info
#         """
#         async with self.lock:
#             if self.num_corrections >= self.max_corrections:
#                 max_feedback = "\nthe answer is \\boxed{no solution}"
#                 if not event.is_set():
#                     event_info["generated_text"] = step
#                     event_info["feedback"] = max_feedback
#                     event_info["correction_index"] = token_index
#                     event.set()
#                 return step, max_feedback

#         async with self.lock:
#             self.force_count += 1
#         print(f"[VerinaSpec] Forcing spec output after {self._count_newlines(step)} newlines (force #{self.force_count})")
        
#         # Stream from LLM to force spec output
#         full_text = await self._stream_force_spec(step)
        
#         generated_spec = extract_spec_from_response(full_text)
#         print("--------------------------------")
#         print("EXTRACTED FORCED PRECOND: ", generated_spec.get("precond", "")[:200])
#         print("EXTRACTED FORCED POSTCOND: ", generated_spec.get("postcond", "")[:200])
#         print("--------------------------------")
        
#         if not generated_spec.get("precond") and not generated_spec.get("postcond"):
#             print("[VerinaSpec] Forced spec generation but could not extract spec")
#             async with self.lock:
#                 if not event.is_set():
#                     event_info["generated_text"] = step
#                     event_info["feedback"] = None
#             return step, None
        
#         async with self.lock:
#             self.verification_count += 1
#             current_verification = self.verification_count
        
#         # Verify the generated spec
#         compiles, compile_output = await self._verify_compilation(generated_spec)
        
#         if not compiles:
#             # Once success has been found, block all future failure feedback
#             if self.success_found:
#                 print(f"[VerinaSpec] Verification #{current_verification}: Compilation FAILED but SUCCESS already found - skipping")
#                 return full_text, None
            
#             # Build the complete text: original reasoning + force prompt + generated spec
#             force_prompt = self._build_force_spec_feedback()
#             complete_generated = force_prompt + full_text
            
#             # Build feedback with error
#             feedback = f"""</think>
# <|im_end|>
# <|im_start|>user
# The specification you gave failed with error:
# {clean_compile_output(compile_output)}

# Please fix this error. Your specification should compile.

# {LEAN4_SPEC_API_REFERENCE}
# <|im_end|>
# <|im_start|>assistant
# <think>
# """
#             print(f"[VerinaSpec] Verification #{current_verification}: Compilation FAILED after forcing spec")
#             async with self.lock:
#                 if not event.is_set():
#                     event_info["generated_text"] = step + complete_generated
#                     event_info["feedback"] = feedback
#                     event_info["correction_index"] = len(full_text)
#                     event_info["mode"] = "compile_error"
#                     event_info["compile_output"] = compile_output
#                     event.set()
#             return full_text, feedback
        
#         # Compilation succeeded — mark success
#         print(f"[VerinaSpec] Verification #{current_verification}: Compilation SUCCESS after forcing spec")
#         self.success_found = True
#         self.verified_spec = generated_spec
        
#         # Build complete generated text for success cases too
#         force_prompt = self._build_force_spec_feedback()
#         complete_generated = force_prompt + full_text
        
#         if self.use_slm_judge and self.slm_server:
#             print(f"[VerinaSpec] Running SLM judge...")
#             is_correct, judgment = await self._verify_with_slm_judge(generated_spec)
#             if not is_correct:
#                 feedback = f"""[/POSTCOND]
# </think>
# <|im_end|>
# <|im_start|>user
# Specification compiled but has a logic issue: {judgment}
# <|im_end|>
# <|im_start|>assistant
# <think>
# """
#                 print(f"[VerinaSpec] Verification #{current_verification}: SLM judge FAILED - {judgment}")
#                 async with self.lock:
#                     if not event.is_set():
#                         event_info["generated_text"] = step + complete_generated
#                         event_info["feedback"] = feedback
#                         event_info["correction_index"] = len(step)
#                         event_info["mode"] = "judge_error"
#                         event_info["judgment"] = judgment
#                         event.set()
#                 return full_text, feedback
#             print(f"[VerinaSpec] SLM judge: CORRECT")
        
#         # Success: include verified spec in user message
#         success_feedback = f"""
# </think>
# <|im_end|>
# <|im_start|>user
# Your specification compiled successfully! Now give the final answer
# <|im_end|>
# <|im_start|>assistant
# <think> Good, the specification I gave compiled successfully. Now I am confident in my answer, so I should output it in the required format.
# </think>
# The final answer is:
# [PRECOND]
# """
#         print(f"[VerinaSpec] Injecting success feedback with verified spec in user message")
#         async with self.lock:
#             if not event.is_set():
#                 event_info["generated_text"] = step + complete_generated
#                 event_info["feedback"] = success_feedback
#                 event.set()
#         return full_text, success_feedback

#         # No verification triggered
#         return step, None

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
        
        # generated_spec = extract_spec_from_response(full_text)
        print("--------------------------------")
        print("EXTRACTED FORCED PRECOND: ", generated_spec.get("precond", "")[:200])
        print("EXTRACTED FORCED POSTCOND: ", generated_spec.get("postcond", "")[:200])
        print("--------------------------------")
        
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
            # Build the complete text: original reasoning + force prompt + generated spec
            force_prompt = self._build_force_spec_feedback()
            complete_generated = force_prompt + full_text
            
            # Build feedback with error
            feedback = f"""</think>
<|im_end|>
<|im_start|>user
The specification you gave failed with error:
{clean_compile_output(compile_output)}

Please fix this error. Your specification should compile.

{LEAN4_SPEC_API_REFERENCE}
<|im_end|>
<|im_start|>assistant
<think>It seems my code failed to compile. I should analyze the error and try to fix it.
"""
            print(f"[VerinaSpec] Verification #{current_verification}: Compilation FAILED after forcing spec")
            async with self.lock:
                self.num_corrections += 1
                if not event.is_set():
                    event_info["generated_text"] = step + complete_generated
                    event_info["feedback"] = feedback
                    event_info["correction_index"] = len(full_text)
                    event_info["mode"] = "compile_error"
                    event_info["compile_output"] = compile_output
                    event.set()
            return full_text, feedback
        
        # Compilation succeeded — mark success
        print(f"[VerinaSpec] Verification #{current_verification}: Compilation SUCCESS after forcing spec")
        self.success_found = True
        self.verified_spec = generated_spec
        
        # Build complete generated text for success cases too
        force_prompt = self._build_force_spec_feedback()
        complete_generated = force_prompt + full_text
        
        if self.use_slm_judge and self.slm_server:
            print(f"[VerinaSpec] Running SLM judge...")
            is_correct, judgment = await self._verify_with_slm_judge(generated_spec)
            if not is_correct:
                feedback = f"""[/POSTCOND]
</think>
<|im_end|>
<|im_start|>user
Specification compiled but has a logic issue: {judgment}
<|im_end|>
<|im_start|>assistant
<think>
"""
                print(f"[VerinaSpec] Verification #{current_verification}: SLM judge FAILED - {judgment}")
                async with self.lock:
                    if not event.is_set():
                        event_info["generated_text"] = step + complete_generated
                        event_info["feedback"] = feedback
                        event_info["correction_index"] = len(step)
                        event_info["mode"] = "judge_error"
                        event_info["judgment"] = judgment
                        event.set()
                return full_text, feedback
            print(f"[VerinaSpec] SLM judge: CORRECT")

        success_feedback = f"""
</think>
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
            
            # Count newlines only in the cur think block
            current_think_content = generated_text[last_think_start + 7:]  # since len('<think>') = 7
            think_newlines = current_think_content.count('\n')
            
            if think_newlines >= self.k_steps:
                self.last_triggered_think_start = last_think_start  # Mark this think block as triggered
                return True, generated_text

        return False, None
