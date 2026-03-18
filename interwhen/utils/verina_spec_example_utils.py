import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import shutil
import numpy as np
import csv
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

_SCRIPT_DIR = Path(__file__).parent.resolve()
VERINA_ROOT = (_SCRIPT_DIR / "../../../../verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"

# Data Utils
class BenchmarkData:
    def __init__(self, data_id: str, description: str, signature: dict, 
                 lean_data: dict, spec_desc: dict, tests: list, reject_inputs: list, metadata: dict):
        self.data_id = data_id
        self.description = description
        self.signature = signature
        self.lean_data = lean_data
        self.spec_desc = spec_desc
        self.tests = tests
        self.reject_inputs = reject_inputs
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
        
        desc_path = task_dir / task_data.get("description_file", "description.txt")
        description = desc_path.read_text().strip() if desc_path.exists() else ""
        
        signature = task_data.get("signature", {})
        
        lean_path = task_dir / task_data.get("lean_file", "task.lean")
        if lean_path.exists():
            lean_data = parse_benchmark_lean_data(lean_path.read_text())
        else:
            lean_data = {}
        
        spec_desc = {
            "precond_desc": task_data.get("specification", {}).get("preconditions", ""),
            "postcond_desc": task_data.get("specification", {}).get("postconditions", ""),
        }
        
        # Load test cases
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
        return None


def load_verina_dataset() -> List[BenchmarkData]:
    """Load all verina benchmark tasks from the datasets directory"""
    results = []
    
    task_dirs = sorted(
        [d for d in VERINA_DATASETS_PATH.glob("verina_*") if d.is_dir()],
        key=lambda x: (x.name.split("_")[1], int(x.name.split("_")[-1])),
    )
    
    for task_dir in task_dirs:
        data = load_benchmark_data_from_task_dir(task_dir)
        if data:
            results.append(data)
    
    return results


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


# Prompt Building Utils

def render_param_list(signature: dict) -> str:
    """Render the parameter list for a function signature"""
    params = signature.get("parameters", [])
    rendered = ""
    for param in params:
        rendered += f"({param['param_name']} : {param['param_type']}) "
    return rendered.strip()


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


def build_full_prompt(data: BenchmarkData) -> str:
    """Build the full prompt string for the LLM"""
    system_prompt, user_prompt = build_spec_gen_prompt(data)
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"


# Extraction Logic
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

# Spec Eval

PRECOND_TEST_MSG_MARKER = "precond_test"
POSTCOND_TEST_MSG_MARKER = "postcond_test"
DECIDABLE_ERR_MSG = "did not evaluate to `true`"
PLAUSIBLE_SUCCESS_MSG = "Unable to find a counter-example"
PLAUSIBLE_FAILED_MSG = "Found a counter-example!"
PLAUSIBLE_TEST_COMMAND = "plausible ( config := { numInst := 1000, maxSize := 100, numRetries := 20, randomSeed := some 42})"


def render_unit_test_value(lean_type: str, value: Any) -> str:
    """Convert a Python value to Lean syntax based on type"""
    if lean_type == "Bool":
        return str(value).lower()
    elif lean_type == "String":
        return f'"{value}"'
    elif lean_type == "Char":
        return f"'{value}'"
    else:
        return str(value)


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
