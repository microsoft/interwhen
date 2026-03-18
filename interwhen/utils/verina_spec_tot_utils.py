import argparse
import asyncio
import json
import logging
import os
import re
import sys
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import aiohttp
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

__all__ = [
    # Path constants
    "VERINA_ROOT",
    "VERINA_DATASETS_PATH",
    "LEAN_PLAYGROUND_DIR",
    # Classes
    "BenchmarkData",
    # Constants
    "CODE_TEST_MSG_MARKER",
    "DECIDABLE_ERR_MSG",
    "PRECOND_TEST_MSG_MARKER",
    "POSTCOND_TEST_MSG_MARKER",
    "PLAUSIBLE_SUCCESS_MSG",
    "PLAUSIBLE_FAILED_MSG",
    "PLAUSIBLE_TEST_COMMAND",
    "parse_benchmark_lean_data",
    "load_benchmark_data_from_task_dir",
    "load_verina_dataset",
    "render_param_list",
    "strip_function_definition",
    "create_lean_file",
    "check_lean_compile",
    "render_unit_test_value",
    "render_code_unit_test",
    "build_test_lean_file",
    "parse_unit_test_results",
    "evaluate_generated_code",
    "evaluate_verina_answer",
    # Spec generation functions
    "build_spec_gen_prompt",
    "build_verina_spec_prompt",
    "extract_spec_from_response",
    "make_aux_reducible",
    "build_spec_test_lean_file",
    "render_precond_sound_test",
    "render_precond_complete_test",
    "render_postcond_complete_test",
    "render_postcond_sound_test",
    "parse_spec_test_results",
    "evaluate_generated_spec",
]

_SCRIPT_DIR = Path(__file__).parent.resolve()
VERINA_ROOT = (_SCRIPT_DIR / "../../../verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

class BenchmarkData:
    """Verina benchmark data structure"""
    def __init__(self, data_id: str, description: str, signature: dict, 
                 lean_data: dict, spec_desc: dict, tests: list, reject_inputs: list, metadata: dict):
        self.data_id = data_id
        self.description = description
        self.signature = signature  # {"name": str, "parameters": list, "return_type": str}
        self.lean_data = lean_data  # contains task_imports, task_aux, code, precond, postcond, proof, etc.
        self.spec_desc = spec_desc  # {"precond_desc": str, "postcond_desc": str}
        self.tests = tests
        self.reject_inputs = reject_inputs  # For precondition completeness testing
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
        
        # Load reject_inputs for precondition completeness testing
        reject_inputs_path = task_dir / "reject_inputs.json"
        reject_inputs = []
        if reject_inputs_path.exists():
            with open(reject_inputs_path, "r") as f:
                reject_inputs = json.load(f)
        
        metadata = task_data.get("metadata", {})
        
        return BenchmarkData(
            data_id=task_id,
            description=description,
            signature=signature,
            lean_data=lean_data,
            spec_desc=spec_desc,
            tests=tests,
            reject_inputs=reject_inputs,
            metadata=metadata,
        )
    except Exception as e:
        #logger.error(f"Error loading {task_dir}: {e}")
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
    
    #logger.info(f"Loaded {len(results)} verina tasks")
    return results


def render_param_list(signature: dict) -> str:
    """Render the parameter list for a function signature"""
    params = signature.get("parameters", [])
    rendered = ""
    for param in params:
        rendered += f"({param['param_name']} : {param['param_type']}) "
    return rendered.strip()


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
        #logger.warning(f"Lean compilation timed out for {lean_file}")
        return False, "TIMEOUT"
    except Exception as e:
        #logger.error(f"Error during compilation: {e}")
        return False, f"ERROR: {e}"


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


def evaluate_verina_answer(output: str, data: BenchmarkData, task_idx: int) -> Tuple[bool, str, str]:
    """Evaluate Verina code generation output - wrapper for best-of-k interface"""
    generated_code = extract_code_from_response(output)
    
    if not generated_code.strip():
        return False, "", "No code extracted from response"
    
    compiles, all_tests_pass, compile_output, test_results = evaluate_generated_code(data, generated_code, task_idx)
    
    num_tests = len(data.tests) if data.tests else 0
    num_passed = sum(1 for v in test_results.values() if v == "pass")
    
    if compiles and all_tests_pass:
        return True, generated_code, f"Code compiles and all {num_tests} tests pass"
    elif compiles:
        return False, generated_code, f"Compilation succeeded but {num_tests - num_passed}/{num_tests} tests failed"
    else:
        error_preview = compile_output[:300] if compile_output else "Unknown error"
        return False, generated_code, f"Compilation failed: {error_preview}"


# --------------------- Verina Spec Gen helpers ---------------------

PRECOND_TEST_MSG_MARKER = "precond_test"
POSTCOND_TEST_MSG_MARKER = "postcond_test"
PLAUSIBLE_SUCCESS_MSG = "Unable to find a counter-example"
PLAUSIBLE_FAILED_MSG = "Found a counter-example!"
PLAUSIBLE_TEST_COMMAND = "plausible ( config := { numInst := 1000, maxSize := 100, numRetries := 20, randomSeed := some 42})"


def build_spec_gen_prompt(data: BenchmarkData) -> Tuple[str, str]:
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

    signature = data.signature
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)
    params = signature.get("parameters", [])
    param_names_str = ' '.join([f"({p['param_name']})" for p in params])
    
    # Get ground truth code to show
    code = data.lean_data.get("code", "").strip()
    code_aux = data.lean_data.get("code_aux", "").strip()
    task_aux = data.lean_data.get("task_aux", "").strip()
    
    # Natural language spec descriptions if available
    precond_desc = data.spec_desc.get("precond_desc", "")
    postcond_desc = data.spec_desc.get("postcond_desc", "")
    
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
{data.description}

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


def build_verina_spec_prompt(data: BenchmarkData) -> str:
    """Build the full prompt string for the LLM for spec generation"""
    system_prompt, user_prompt = build_spec_gen_prompt(data)
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


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


def build_spec_test_lean_file(
    data: BenchmarkData, 
    generated_spec: Dict[str, str],
    test_type: str = "compile"
) -> str:
    """Build a Lean file to test the generated specification."""
    signature = data.signature
    func_name = signature.get("name", "solution")
    return_type = signature.get("return_type", "Bool")
    param_list = render_param_list(signature)
    params = signature.get("parameters", [])
    param_names = " ".join([f"({p['param_name']})" for p in params])
    
    # Build imports
    task_imports = data.lean_data.get("task_imports", "").strip()
    solution_imports = data.lean_data.get("solution_imports", "").strip()
    imports = task_imports
    if solution_imports:
        imports += "\n" + solution_imports
    if "import Mathlib" not in imports:
        imports = "import Mathlib\n" + imports
    if "import Plausible" not in imports:
        imports = "import Plausible\n" + imports
    
    # Build auxiliary definitions
    task_aux = data.lean_data.get("task_aux", "").strip()
    solution_aux = data.lean_data.get("solution_aux", "").strip()
    
    precond_name = f"{func_name}_precond"
    postcond_name = f"{func_name}_postcond"
    
    # Use generated spec
    precond = generated_spec.get("precond", "True").strip()
    postcond = generated_spec.get("postcond", "").strip()
    precond_aux = generated_spec.get("precond_aux", "").strip()
    postcond_aux = generated_spec.get("postcond_aux", "").strip()
    
    if precond_aux:
        precond_aux = make_aux_reducible(precond_aux)
    if postcond_aux:
        postcond_aux = make_aux_reducible(postcond_aux)
    
    lean_content = f"""{imports}

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

    # Add tests based on test_type
    if test_type == "precond_sound":
        lean_content += "\n-- Precondition Soundness Tests (valid inputs should satisfy precond)\n"
        for idx, test_case in enumerate(data.tests):
            lean_content += render_precond_sound_test(signature, precond_name, test_case, idx)
    
    elif test_type == "precond_complete":
        lean_content += "\n-- Precondition Completeness Tests (invalid inputs should NOT satisfy precond)\n"
        for idx, reject_input in enumerate(data.reject_inputs):
            lean_content += render_precond_complete_test(signature, precond_name, reject_input, idx)
    
    elif test_type == "postcond_sound":
        lean_content += "\n-- Postcondition Soundness Tests (wrong outputs should NOT satisfy postcond)\n"
        global_idx = 0
        for idx, test_case in enumerate(data.tests):
            unexpected_list = test_case.get("unexpected", [])
            for unexpected_idx, unexpected in enumerate(unexpected_list):
                lean_content += render_postcond_sound_test(
                    signature, precond_name, postcond_name, test_case, global_idx, unexpected, unexpected_idx
                )
                global_idx += 1
    
    elif test_type == "postcond_complete":
        lean_content += "\n-- Postcondition Completeness Tests (correct outputs should satisfy postcond)\n"
        for idx, test_case in enumerate(data.tests):
            lean_content += render_postcond_complete_test(
                signature, precond_name, postcond_name, test_case, idx
            )
    
    return lean_content


def render_precond_sound_test(signature: dict, precond_name: str, test_case: dict, test_idx: int) -> str:
    """Render test: valid input should satisfy precondition"""
    params = signature.get("parameters", [])
    
    rendered = f'\n#print "<{PRECOND_TEST_MSG_MARKER}_sound>{test_idx}</{PRECOND_TEST_MSG_MARKER}_sound>"\n'
    rendered += f"#guard decide ({precond_name}"
    
    for param in params:
        param_name = param["param_name"]
        param_type = param["param_type"]
        input_value = test_case["input"].get(param_name, "")
        rendered += f" ({render_unit_test_value(param_type, input_value)})"
    
    rendered += ")\n"
    return rendered


def render_precond_complete_test(signature: dict, precond_name: str, reject_input: dict, test_idx: int) -> str:
    """Render test: reject_input should NOT satisfy precondition"""
    params = signature.get("parameters", [])
    
    rendered = f'\n#print "<{PRECOND_TEST_MSG_MARKER}_complete>{test_idx}</{PRECOND_TEST_MSG_MARKER}_complete>"\n'
    rendered += f"#guard decide (¬ ({precond_name}"
    
    for param in params:
        param_name = param["param_name"]
        param_type = param["param_type"]
        input_value = reject_input.get("input", {}).get(param_name, "")
        rendered += f" ({render_unit_test_value(param_type, input_value)})"
    
    rendered += "))\n"
    return rendered


def render_postcond_complete_test(signature: dict, precond_name: str, postcond_name: str, test_case: dict, test_idx: int) -> str:
    """Render test: expected output should satisfy postcondition"""
    params = signature.get("parameters", [])
    return_type = signature.get("return_type", "Bool")
    
    rendered = f'\n#print "<{POSTCOND_TEST_MSG_MARKER}_complete>{test_idx}</{POSTCOND_TEST_MSG_MARKER}_complete>"\n'
    rendered += f"#guard decide ({postcond_name}"
    
    for param in params:
        param_name = param["param_name"]
        param_type = param["param_type"]
        input_value = test_case["input"].get(param_name, "")
        rendered += f" ({render_unit_test_value(param_type, input_value)})"
    
    expected = test_case.get("expected", "")
    rendered += f" ({render_unit_test_value(return_type, expected)}) (by sorry))\n"
    
    return rendered


def render_postcond_sound_test(
    signature: dict, precond_name: str, postcond_name: str, 
    test_case: dict, global_idx: int, unexpected: Any, unexpected_idx: int
) -> str:
    """Render test: unexpected output should NOT satisfy postcondition"""
    params = signature.get("parameters", [])
    return_type = signature.get("return_type", "Bool")
    test_idx = global_idx
    
    rendered = f'\n#print "<{POSTCOND_TEST_MSG_MARKER}_sound>{test_idx}</{POSTCOND_TEST_MSG_MARKER}_sound>"\n'
    rendered += f"#guard decide (¬ ({postcond_name}"
    
    for param in params:
        param_name = param["param_name"]
        param_type = param["param_type"]
        input_value = test_case["input"].get(param_name, "")
        rendered += f" ({render_unit_test_value(param_type, input_value)})"
    
    rendered += f" ({render_unit_test_value(return_type, unexpected)}) (by sorry)))\n"
    
    return rendered


def parse_spec_test_results(compile_output: str, marker: str, num_tests: int) -> Tuple[int, int, dict]:
    """Parse compilation output for spec test results."""
    test_results = {}
    
    if "error" not in compile_output.lower():
        for idx in range(num_tests):
            test_results[idx] = "pass"
        return num_tests, 0, test_results
    
    start_marker = f"<{marker}>"
    end_marker = f"</{marker}>"
    
    parts = compile_output.split(start_marker)
    test_messages = {}
    
    for part in parts[1:]:
        if end_marker in part:
            idx_str, rest = part.split(end_marker, 1)
            try:
                test_idx = int(idx_str.strip().split(",")[0])
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
            test_results[idx] = "error"
            num_failed += 1
        else:
            test_results[idx] = "pass"
            num_passed += 1
    
    return num_passed, num_failed, test_results


def evaluate_generated_spec(
    data: BenchmarkData, 
    generated_spec: Dict[str, str], 
    task_idx: int
) -> Dict[str, Any]:
    """Evaluate the generated specification using soundness and completeness tests."""
    result = {
        "compiles": False,
        "precond_sound_pass": 0,
        "precond_sound_total": 0,
        "precond_complete_pass": 0,
        "precond_complete_total": 0,
        "postcond_sound_pass": 0,
        "postcond_sound_total": 0,
        "postcond_complete_pass": 0,
        "postcond_complete_total": 0,
        "precond_correct": False,
        "postcond_correct": False,
        "spec_sound": False,
        "spec_complete": False,
        "full_spec_correct": False,
        "compile_error": "",
    }
    
    # First check if spec compiles
    compile_content = build_spec_test_lean_file(data, generated_spec, "compile")
    lean_file = create_lean_file(f"spec_compile_{data.data_id}_{task_idx}", compile_content)
    compiles, output = check_lean_compile(lean_file)
    
    result["compiles"] = compiles
    if not compiles:
        result["compile_error"] = output[:500]
        return result
    
    # Test precondition soundness
    if data.tests:
        precond_sound_content = build_spec_test_lean_file(data, generated_spec, "precond_sound")
        lean_file = create_lean_file(f"spec_precond_sound_{data.data_id}_{task_idx}", precond_sound_content)
        _, output = check_lean_compile(lean_file)
        
        result["precond_sound_total"] = len(data.tests)
        passed, failed, _ = parse_spec_test_results(output, f"{PRECOND_TEST_MSG_MARKER}_sound", len(data.tests))
        result["precond_sound_pass"] = passed
    
    # Test precondition completeness
    if data.reject_inputs:
        precond_complete_content = build_spec_test_lean_file(data, generated_spec, "precond_complete")
        lean_file = create_lean_file(f"spec_precond_complete_{data.data_id}_{task_idx}", precond_complete_content)
        _, output = check_lean_compile(lean_file)
        
        result["precond_complete_total"] = len(data.reject_inputs)
        passed, failed, _ = parse_spec_test_results(output, f"{PRECOND_TEST_MSG_MARKER}_complete", len(data.reject_inputs))
        result["precond_complete_pass"] = passed
    
    # Test postcondition completeness
    if data.tests:
        postcond_complete_content = build_spec_test_lean_file(data, generated_spec, "postcond_complete")
        lean_file = create_lean_file(f"spec_postcond_complete_{data.data_id}_{task_idx}", postcond_complete_content)
        _, output = check_lean_compile(lean_file)
        
        result["postcond_complete_total"] = len(data.tests)
        passed, failed, _ = parse_spec_test_results(output, f"{POSTCOND_TEST_MSG_MARKER}_complete", len(data.tests))
        result["postcond_complete_pass"] = passed
    
    # Test postcondition soundness
    total_unexpected = sum(len(t.get("unexpected", [])) for t in data.tests) if data.tests else 0
    if total_unexpected > 0:
        postcond_sound_content = build_spec_test_lean_file(data, generated_spec, "postcond_sound")
        lean_file = create_lean_file(f"spec_postcond_sound_{data.data_id}_{task_idx}", postcond_sound_content)
        _, output = check_lean_compile(lean_file)
        
        result["postcond_sound_total"] = total_unexpected
        passed, failed, _ = parse_spec_test_results(output, f"{POSTCOND_TEST_MSG_MARKER}_sound", total_unexpected)
        result["postcond_sound_pass"] = passed
    
    # Compute combined correctness metrics
    precond_sound_all_pass = (result["precond_sound_pass"] == result["precond_sound_total"] and result["precond_sound_total"] > 0) or result["precond_sound_total"] == 0
    precond_complete_all_pass = (result["precond_complete_pass"] == result["precond_complete_total"] and result["precond_complete_total"] > 0) or result["precond_complete_total"] == 0
    result["precond_correct"] = precond_sound_all_pass and precond_complete_all_pass
    
    postcond_sound_all_pass = (result["postcond_sound_pass"] == result["postcond_sound_total"] and result["postcond_sound_total"] > 0) or result["postcond_sound_total"] == 0
    postcond_complete_all_pass = (result["postcond_complete_pass"] == result["postcond_complete_total"] and result["postcond_complete_total"] > 0) or result["postcond_complete_total"] == 0
    result["postcond_correct"] = postcond_sound_all_pass and postcond_complete_all_pass
    
    result["spec_sound"] = precond_sound_all_pass and postcond_sound_all_pass
    result["spec_complete"] = precond_complete_all_pass and postcond_complete_all_pass
    result["full_spec_correct"] = result["precond_correct"] and result["postcond_correct"]
    
    return result
