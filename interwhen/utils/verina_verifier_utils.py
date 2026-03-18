import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import shutil
import copy

# Verina Paths
VERINA_ROOT = (Path(__file__).parent.resolve() / ".." / ".." / ".." / "verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

# Lean Eval Utils
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
        return False, "TIMEOUT"
    except Exception as e:
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

def render_param_list(signature: dict) -> str:
    """Render the parameter list for a function signature"""
    params = signature.get("parameters", [])
    rendered = ""
    for param in params:
        rendered += f"({param['param_name']} : {param['param_type']}) "
    return rendered.strip()


# Code Extraction Utils

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
    
    # Extract PRECOND_AUX (take last match to get the most recent/corrected version)
    precond_aux_matches = re.findall(r'\[PRECOND_AUX\](.*?)\[/PRECOND_AUX\]', cleaned, re.DOTALL | re.IGNORECASE)
    if precond_aux_matches:
        result["precond_aux"] = precond_aux_matches[-1].strip()
    
    # Extract POSTCOND_AUX (take last match)
    postcond_aux_matches = re.findall(r'\[POSTCOND_AUX\](.*?)\[/POSTCOND_AUX\]', cleaned, re.DOTALL | re.IGNORECASE)
    if postcond_aux_matches:
        result["postcond_aux"] = postcond_aux_matches[-1].strip()
    
    # Extract PRECOND (take last match to get the most recent/corrected version)
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
    
    # Extract POSTCOND (take last match to get the most recent/corrected version)
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

# LEAN 4 API REFERENCE (for error feedback)

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