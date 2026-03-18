import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import shutil
import copy

# Verina Paths
VERINA_ROOT = (Path(__file__).parent.resolve() / ".." / ".." / ".." / "verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

# Dataaset Loading (adapted from verina/src/verina/dataset/dataset.py)
class BenchmarkData:
    def __init__(self, data_id: str, description: str, signature: dict, 
                 lean_data: dict, spec_desc: dict, tests: list, metadata: dict):
        self.data_id = data_id
        self.description = description
        self.signature = signature  # {"name": str, "parameters": list, "return_type": str}
        self.lean_data = lean_data  # contains task_imports, task_aux, code, precond, postcond, proof, etc.
        self.spec_desc = spec_desc  # {"precond_desc": str, "postcond_desc": str}
        self.tests = tests
        self.metadata = metadata

def parse_benchmark_lean_data(raw_lean_data: str) -> dict:
    """Parse a .lean file with !benchmark markers into sections"""
    lines = raw_lean_data.strip().splitlines()
    
    lean_data = {
        "task_imports": "",
        "solution_imports": "",
        "task_aux": "",
        "solution_aux": "",
        "code_aux": "",
        "precond_aux": "",
        "postcond_aux": "",
        "proof_aux": "",
        "code": "",
        "precond": "True",
        "postcond": "",
        "proof": "sorry",
    }
    
    current_section = None
    current_content = []
    current_args = {}
    
    for line in lines:
        if "-- !benchmark" in line:
            marker_part = line.split("-- !benchmark", 1)[1].strip()
            
            if marker_part.startswith("@start"):
                # Save previous section if any
                if current_section is not None:
                    content = "\n".join(current_content).strip()
                    if current_section == "import":
                        import_type = current_args.get("type", "task")
                        if import_type == "task":
                            lean_data["task_imports"] += content + "\n"
                        elif import_type == "solution":
                            lean_data["solution_imports"] += content + "\n"
                    elif current_section in lean_data:
                        lean_data[current_section] = content
                
                # Start new section
                parts = marker_part.split("@start", 1)[1].strip().split(None, 1)
                current_section = parts[0].strip()
                current_args = {}
                current_content = []
                
                if len(parts) > 1:
                    for arg in parts[1].strip().split():
                        if "=" in arg:
                            key, value = arg.split("=", 1)
                            current_args[key] = value
                            
            elif marker_part.startswith("@end"):
                if current_section is not None:
                    content = "\n".join(current_content).strip()
                    if current_section == "import":
                        import_type = current_args.get("type", "task")
                        if import_type == "task":
                            lean_data["task_imports"] += content + "\n"
                        elif import_type == "solution":
                            lean_data["solution_imports"] += content + "\n"
                    elif current_section in lean_data:
                        lean_data[current_section] = content
                current_section = None
                current_content = []
                current_args = {}
        else:
            if current_section is not None:
                current_content.append(line)
    
    return lean_data


def load_benchmark_data_from_task_dir(task_dir: Path) -> Optional[BenchmarkData]:
    """Load a single benchmark task from its directory"""
    task_path = task_dir / "task.json"
    if not task_path.exists():
        return None
    
    try:
        with open(task_path, "r") as f:
            task_data = json.load(f)
        
        task_id = task_data.get("id")
        if not task_id:
            return None
        
        # Read description
        desc_path = task_dir / task_data.get("description_file", "description.txt")
        description = desc_path.read_text().strip() if desc_path.exists() else ""
        
        # Read signature
        signature = task_data.get("signature", {})
        
        # Read lean file
        lean_path = task_dir / task_data.get("lean_file", "task.lean")
        if lean_path.exists():
            lean_data = parse_benchmark_lean_data(lean_path.read_text())
        else:
            lean_data = {}
        
        # Read spec description
        spec_desc = {
            "precond_desc": task_data.get("specification", {}).get("preconditions", ""),
            "postcond_desc": task_data.get("specification", {}).get("postconditions", ""),
        }
        
        # Read tests
        test_path = task_dir / task_data.get("test_file", "test.json")
        tests = []
        if test_path.exists():
            with open(test_path, "r") as f:
                tests = json.load(f)
        
        metadata = task_data.get("metadata", {})
        
        return BenchmarkData(
            data_id=task_id,
            description=description,
            signature=signature,
            lean_data=lean_data,
            spec_desc=spec_desc,
            tests=tests,
            metadata=metadata,
        )
    except Exception as e:
        return None


def load_verina_dataset() -> List[BenchmarkData]:
    """Load all verina benchmark tasks from the datasets directory"""
    results = []
    
    # Get all task directories sorted by ID
    task_dirs = sorted(
        [d for d in VERINA_DATASETS_PATH.glob("verina_*") if d.is_dir()],
        key=lambda x: (x.name.split("_")[1], int(x.name.split("_")[-1])),
    )
    
    for task_dir in task_dirs:
        data = load_benchmark_data_from_task_dir(task_dir)
        if data:
            results.append(data)
    
    return results


# Lean Helpers
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
        result = subprocess.run(
            ["lake", "lean", str(lean_file)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            cwd=VERINA_ROOT,
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

# Helpers for prompt building
def render_param_list(signature: dict) -> str:
    """Render the parameter list for a function signature"""
    params = signature.get("parameters", [])
    rendered = ""
    for param in params:
        rendered += f"({param['param_name']} : {param['param_type']}) "
    return rendered.strip()


def build_code_gen_prompt(data: BenchmarkData) -> Tuple[str, str]:
    """
    Build a simple prompt for Lean 4 code generation.
    Returns (system_prompt, user_prompt)
    """
    system_prompt = f"""You are an expert Lean 4 programmer. Generate valid Lean 4 code for the function body. Wrap your final code in [CODE] [/CODE] tags strictly.""" 
    signature = data.signature
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)
    params = signature.get("parameters", [])
    
    precond_name = f"{func_name}_precond"
    param_names_str = ' '.join([f"({p['param_name']})" for p in params])
    
    # Get auxiliary definitions (only if they exist)
    solution_aux = data.lean_data.get("solution_aux", "").strip()
    task_aux = data.lean_data.get("task_aux", "").strip()
    code_aux = data.lean_data.get("code_aux", "").strip()
    precond = data.lean_data.get("precond", "True").strip()
    postcond = data.lean_data.get("postcond", "").strip()
    
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

## Function Signature
```lean4
def {func_name} {param_list} (h_precond : {precond_name} {param_names_str}) : {return_type} :=
  -- YOUR CODE HERE (just output this part inside [CODE] [/CODE] tags)
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
Provide ONLY the function body expression wrapped in [CODE]...[/CODE] tags."""

    return system_prompt, user_prompt


def build_full_prompt(data: BenchmarkData) -> str:
    """Build the full prompt string for the LLM"""
    system_prompt, user_prompt = build_code_gen_prompt(data)
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

# Code Extraction and Eval
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

# Unit Test Rendering
CODE_TEST_MSG_MARKER = "code_test"
DECIDABLE_ERR_MSG = "did not evaluate to `true`"


def render_unit_test_value(lean_type: str, value) -> str:
    """Convert a Python value to Lean syntax based on type"""
    if lean_type == "Bool":
        return str(value).lower()
    elif lean_type == "String":
        return f'"{value}"'
    elif lean_type == "Char":
        return f"'{value}'"
    else:
        # For Int, List, Array, etc. - use value as-is (already in Lean format from JSON)
        return str(value)


def render_code_unit_test(signature: dict, test_case: dict, test_idx: int) -> str:
    """Render a single unit test using #guard"""
    func_name = signature.get("name", "solution")
    params = signature.get("parameters", [])
    return_type = signature.get("return_type", "Bool")
    
    rendered = f'#print "<{CODE_TEST_MSG_MARKER}>{test_idx}</{CODE_TEST_MSG_MARKER}>"\n\n'
    rendered += f"#guard {func_name}"
    
    for param in params:
        param_name = param["param_name"]
        param_type = param["param_type"]
        input_value = test_case["input"].get(param_name, "")
        rendered += f" ({render_unit_test_value(param_type, input_value)})"
    
    # Add (by sorry) to satisfy precondition hypothesis
    rendered += " (by sorry)"
    
    # Add expected value comparison
    expected = test_case.get("expected", "")
    rendered += f" == ({render_unit_test_value(return_type, expected)})"
    
    return rendered


def build_test_lean_file(data: BenchmarkData, generated_code: str, include_unit_tests: bool = True) -> str:
    """Build a complete Lean file to test the generated code"""
    signature = data.signature
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)
    params = signature.get("parameters", [])
    param_names = " ".join([f"({p['param_name']})" for p in params])
    
    # Indent multiline generated_code so all lines have proper indentation
    # First line gets 2 spaces from template, subsequent lines need explicit indentation
    if '\n' in generated_code:
        lines = generated_code.split('\n')
        # First line has no extra indent (template adds 2 spaces)
        # Subsequent lines need 2 spaces prepended
        indented_lines = [lines[0]] + ['  ' + line if line.strip() else line for line in lines[1:]]
        generated_code = '\n'.join(indented_lines)
    
    # Build imports - include both task and solution imports
    task_imports = data.lean_data.get("task_imports", "").strip()
    solution_imports = data.lean_data.get("solution_imports", "").strip()
    imports = task_imports
    if solution_imports:
        imports += "\n" + solution_imports
    if "import Mathlib" not in imports:
        imports = "import Mathlib\n" + imports
    
    # Build auxiliary definitions - include solution_aux which has helper functions
    solution_aux = data.lean_data.get("solution_aux", "").strip()
    task_aux = data.lean_data.get("task_aux", "").strip()
    precond_aux = data.lean_data.get("precond_aux", "").strip()
    postcond_aux = data.lean_data.get("postcond_aux", "").strip()
    code_aux = data.lean_data.get("code_aux", "").strip()
    
    # Build precondition
    precond = data.lean_data.get("precond", "True").strip()
    precond_name = f"{func_name}_precond"
    
    # Build postcondition
    postcond = data.lean_data.get("postcond", "").strip()
    postcond_name = f"{func_name}_postcond"
    
    lean_content = f"""{imports}

-- Solution auxiliary definitions (helper functions)
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

-- Verification theorem (compilation test)
-- If this compiles, the code at least type-checks
#check {func_name}
"""
    
    # Add unit tests if requested
    if include_unit_tests and data.tests:
        lean_content += "\n-- Unit Tests\n"
        for idx, test_case in enumerate(data.tests):
            lean_content += "\n" + render_code_unit_test(signature, test_case, idx) + "\n"
    
    return lean_content


def parse_unit_test_results(compile_output: str, num_tests: int) -> Tuple[int, int, dict]:
    """
    Parse the compilation output to determine which unit tests passed/failed.
    
    Returns: (num_passed, num_failed, test_results_dict)
    """
    test_results = {}
    
    # If compilation succeeded with no errors, all tests passed
    if "error" not in compile_output.lower():
        for idx in range(num_tests):
            test_results[idx] = "pass"
        return num_tests, 0, test_results
    
    # Parse the output to find which tests failed
    # Look for markers like <code_test>0</code_test> followed by error messages
    code_test_start = f"<{CODE_TEST_MSG_MARKER}>"
    code_test_end = f"</{CODE_TEST_MSG_MARKER}>"
    
    # Split by start marker to get test sections
    parts = compile_output.split(code_test_start)
    
    # Build a map of test index to message
    test_messages = {}
    for part in parts[1:]:  # Skip first part (before any marker)
        if code_test_end in part:
            idx_str, rest = part.split(code_test_end, 1)
            try:
                test_idx = int(idx_str.strip())
                test_messages[test_idx] = rest
            except ValueError:
                continue
    
    num_passed = 0
    num_failed = 0
    
    for idx in range(num_tests):
        msg = test_messages.get(idx, "")
        if DECIDABLE_ERR_MSG in msg:
            test_results[idx] = "fail"
            num_failed += 1
        elif "error" in msg.lower():
            # Some other error (e.g., type mismatch) - count as fail
            test_results[idx] = "error"
            num_failed += 1
        else:
            test_results[idx] = "pass"
            num_passed += 1
    
    return num_passed, num_failed, test_results


def evaluate_generated_code(data: BenchmarkData, generated_code: str, task_idx: int) -> Tuple[bool, bool, str, dict]:
    """
    Evaluate the generated code by compiling it with Lean and running unit tests.
    
    Returns: (compiles, all_tests_pass, output, test_results)
    """
    lean_content = build_test_lean_file(data, generated_code, include_unit_tests=True)
    
    # Create lean file
    lean_file = create_lean_file(f"test_{data.data_id}_{task_idx}", lean_content)
    
    # Check compilation (which also runs unit tests via #guard)
    compiles, output = check_lean_compile(lean_file)
    
    # Parse unit test results
    num_tests = len(data.tests) if data.tests else 0
    if num_tests > 0:
        num_passed, num_failed, test_results = parse_unit_test_results(output, num_tests)
        all_tests_pass = (num_failed == 0) and compiles
    else:
        # No tests, just check compilation
        test_results = {}
        all_tests_pass = compiles
    
    return compiles, all_tests_pass, output, test_results

def test_lean_compile():
    """Test if Lean compile check works with valid and invalid code."""
    print("Testing Lean compile check...")
    clean_playground()
    
    # Test 1: Valid Lean code
    valid_code = """
def hello : Nat := 42
#check hello
theorem one_eq_one : 1 = 1 := rfl
"""
    lean_file = create_lean_file("test_valid", valid_code)
    success, output = check_lean_compile(lean_file)
    print(f"\n[Test 1] Valid code:")
    print(f"  Compiled successfully: {success}")
    if not success:
        print(f"  Error: {output[:300]}")
    
    # Test 2: Invalid Lean code (should fail)
    invalid_code = """
def broken : Nat := "not a nat"
"""
    lean_file2 = create_lean_file("test_invalid", invalid_code)
    success2, output2 = check_lean_compile(lean_file2)
    print(f"\n[Test 2] Invalid code:")
    print(f"  Compiled successfully: {success2} (expected: False)")
    
    # Summary
    print(f"\n" + "="*50)
    if success and not success2:
        print("Lean compile check is working correctly!")
    else:
        print("Lean compile check may have issues.")
        if not success:
            print("  - Valid code failed to compile")
        if success2:
            print("  - Invalid code unexpectedly compiled")
    print("="*50)
    
    return success and not success2
