"""
VERINA Specification Generation Benchmark with Step Verification

This script evaluates LLM-generated specifications (preconditions and postconditions)
on the VERINA benchmark using soundness and completeness metrics, integrated with
the StepVerifierVerinaSpecMonitor for streaming verification.

Soundness: Tests that the spec correctly rejects invalid inputs/outputs
Completeness: Tests that the spec correctly accepts valid inputs/outputs

Usage:
    python verina_specgen.py --num_examples 50
    python verina_specgen.py --debug
"""

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

from interwhen.interject import stream_completion
from interwhen.monitors import EATMonitor, StepVerifierVerinaSpecMonitor

logger = logging.getLogger(__name__)

# ============== MODEL CONFIGURATION ==============
MAIN_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"
EARLYSTOP_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# =================================================


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from the model path."""
    short_name = model_name.split("/")[-1]
    short_name = short_name.replace(" ", "_").replace(":", "-")
    return short_name


def get_output_dirs(main_model: str, base_dir: str = "../verina_spec_results"):
    """Create and return output directory paths based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    
    dirs = {
        "base": output_base,
        "reasoning": os.path.join(output_base, "Reasoning_output_verina_spec"),
        "csv_saved": os.path.join(output_base, "csv_saved"),
        "plots": os.path.join(output_base, "plots"),
    }
    
    # Create all directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def get_log_filename(main_model: str, num_examples: int, base_dir: str = "../verina_spec_results") -> str:
    """Generate log filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"TTS_spec_{num_examples}examples.log")


def get_token_filename(main_model: str, num_examples: int, base_dir: str = "../verina_spec_results") -> str:
    """Generate token CSV filename based on model name."""
    model_short_name = get_model_short_name(main_model)
    output_base = os.path.join(base_dir, model_short_name)
    os.makedirs(output_base, exist_ok=True)
    return os.path.join(output_base, f"TTS_spec_{num_examples}examples.csv")


# ============================================================================
# PATHS - Update these to point to your verina repo
# ============================================================================
_SCRIPT_DIR = Path(__file__).parent.resolve()
VERINA_ROOT = (_SCRIPT_DIR / "../../../verina").resolve()
VERINA_DATASETS_PATH = VERINA_ROOT / "datasets" / "verina"
LEAN_PLAYGROUND_DIR = VERINA_ROOT / "lean-playground"


# ============================================================================
# DATASET LOADING
# ============================================================================

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
        logger.error(f"Error loading {task_dir}: {e}")
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
    
    logger.info(f"Loaded {len(results)} verina tasks")
    return results


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
        logger.warning(f"Lean compilation timed out for {lean_file}")
        return False, "TIMEOUT"
    except Exception as e:
        logger.error(f"Error during compilation: {e}")
        return False, f"ERROR: {e}"


# ============================================================================
# PROMPTING FOR SPEC GENERATION
# ============================================================================

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


# ============================================================================
# SPEC EXTRACTION
# ============================================================================

# def extract_spec_from_response(response: str) -> Dict[str, str]:
#     """Extract precondition and postcondition from response.
    
#     Returns dict with keys: precond, postcond, precond_aux, postcond_aux
#     """
#     result = {
#         "precond": "",
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
#         precond_start_match = re.search(r'\[PRECOND\]\s*(.*)', cleaned, re.DOTALL | re.IGNORECASE)
#         if precond_start_match:
#             precond = precond_start_match.group(1).strip()
#             postcond_idx = precond.lower().find("[postcond")
#             if postcond_idx != -1:
#                 precond = precond[:postcond_idx].strip()
#             result["precond"] = precond
    
#     # Extract POSTCOND
#     postcond_match = re.search(r'\[POSTCOND\](.*?)\[/POSTCOND\]', cleaned, re.DOTALL | re.IGNORECASE)
#     if postcond_match:
#         result["postcond"] = postcond_match.group(1).strip()
#     else:
#         postcond_start_match = re.search(r'\[POSTCOND\]\s*(.*)', cleaned, re.DOTALL | re.IGNORECASE)
#         if postcond_start_match:
#             postcond = postcond_start_match.group(1).strip()
#             result["postcond"] = postcond
    
#     # Clean up any remaining markdown tags
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

# ============================================================================
# SPEC EVALUATION - SOUNDNESS AND COMPLETENESS
# ============================================================================

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


# ============================================================================
# LLM SERVER SETUP
# ============================================================================

def init_llm_server(modelname: str, max_tokens: int = 20480, port: int = 8000) -> dict:
    """Initialize LLM server configuration"""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": modelname,
        "max_tokens": max_tokens,
        "top_k": 20,
        "top_p": 0.95,
        "min_p": 0.0,
        "temperature": 0.6,
        "stream": True,
        "logprobs": 20,
        "use_beam_search": False,
        "prompt_cache": True,
        "seed": 42,
    }
    headers = {"Content-Type": "application/json"}
    return {"url": url, "payload": payload, "headers": headers}


# ============================================================================
# SAVING UTILITIES
# ============================================================================

def save_reasoning_trace(idx: int, data_id: str, prompt_with_answer: str, reason_dir: str):
    """Save the full reasoning trace to a file"""
    filename = os.path.join(reason_dir, f"reason_{idx}_{data_id}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prompt_with_answer)


def save_results_csv(results: list, output_path: str):
    """Save results to CSV file"""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "idx", "data_id", "compiles",
            "precond_sound_pass", "precond_sound_total",
            "precond_complete_pass", "precond_complete_total",
            "postcond_sound_pass", "postcond_sound_total",
            "postcond_complete_pass", "postcond_complete_total",
            "precond_correct", "postcond_correct",
            "spec_sound", "spec_complete", "full_spec_correct",
            "reasoning_tokens", "precond", "postcond", "num_times_forced","finally_wrong"
        ])
        for r in results:
            precond_escaped = r.get("precond", "").replace("\n", "\\n")
            postcond_escaped = r.get("postcond", "").replace("\n", "\\n")
            writer.writerow([
                r["idx"], 
                r["data_id"], 
                r["compiles"],
                r["precond_sound_pass"],
                r["precond_sound_total"],
                r["precond_complete_pass"],
                r["precond_complete_total"],
                r["postcond_sound_pass"],
                r["postcond_sound_total"],
                r["postcond_complete_pass"],
                r["postcond_complete_total"],
                r.get("precond_correct", False),
                r.get("postcond_correct", False),
                r.get("spec_sound", False),
                r.get("spec_complete", False),
                r.get("full_spec_correct", False),
                r.get("reasoning_tokens", 0),
                precond_escaped,
                postcond_escaped,
                r.get("num_times_forced", 0),
                r.get("finally_wrong", False)
            ])


def compute_average_tokens(token_file: str) -> float:
    """Compute average reasoning tokens from the token file"""
    if not os.path.exists(token_file):
        return 0.0
    
    tokens = []
    with open(token_file, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                tokens.append(int(row[0]))
    
    return np.mean(tokens) if tokens else 0.0


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
        print("✓ Lean compile check is working correctly!")
    else:
        print("✗ Lean compile check may have issues.")
        if not success:
            print("  - Valid code failed to compile")
        if success2:
            print("  - Invalid code unexpectedly compiled")
    print("="*50)
    
    return success and not success2


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verina spec generation benchmark with step verification")
    parser.add_argument("--monitor", "-m", action="store_true", default=True, help="Enable monitors")
    parser.add_argument("--num_examples", "-n", type=int, default=50, help="Number of examples to run")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logs")
    parser.add_argument("--port", "-p", type=int, default=8000, help="LLM server port")
    parser.add_argument("--main_model", type=str, default=MAIN_MODEL, help="Main model to use for generation")
    parser.add_argument("--earlystop_model", type=str, default=EARLYSTOP_MODEL, help="Model to use for early stopping")
    parser.add_argument("--k_steps", "-k", type=int, default=40, help="Newlines threshold for forcing spec output")
    args = parser.parse_args()
    
    main_model = args.main_model
    earlystop_model = args.earlystop_model
    
    output_dirs = get_output_dirs(main_model)
    logfile = get_log_filename(main_model, args.num_examples)
    token_file = get_token_filename(main_model, args.num_examples)
    reason_dir = output_dirs["reasoning"]
    csv_dir = output_dirs["csv_saved"]
    
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler()
        ],
        force=True,
    )
    
    logger.info(f"Main model: {main_model}")
    logger.info(f"Early stop model: {earlystop_model}")
    logger.info(f"Output directory: {output_dirs['base']}")
    
    logger.info("Loading verina dataset...")
    dataset = load_verina_dataset()
    logger.info(f"Loaded {len(dataset)} tasks")

    print("=============testing lean compile=================")
    test_lean_compile()
    
    llm_server = init_llm_server(main_model, max_tokens=20480, port=args.port)
    
    N = args.num_examples if args.num_examples > 0 else len(dataset)
    total = len(dataset)
    indices = [i for i in range(N)]
    
    results = []
    num_correct = 0
    
    logger.info(f"Running on {N} examples...")
    
    for i, idx in enumerate(indices):
        print("SAMPLE ", i+1)
        data = dataset[idx]
        logger.info(f"\n{'='*50}")
        logger.info(f"[{i+1}/{N}] Task: {data.data_id}")
        logger.info(f"{'='*50}")
        
        prompt = build_full_prompt(data)
        
        # Convert BenchmarkData to dict for the monitor
        task_data = {
            "data_id": data.data_id,
            "description": data.description,
            "signature": data.signature,
            "lean_data": data.lean_data,
            "spec_desc": data.spec_desc,
            "tests": data.tests,
            "reject_inputs": data.reject_inputs,
            "metadata": data.metadata,
        }
        
        # Setup monitors
        if args.monitor:
            monitors = [
                StepVerifierVerinaSpecMonitor(
                    name="VerinaSpecStepVerifier",
                    task_data=task_data,
                    llm_server=llm_server,
                    prompt=prompt,
                    k_steps=args.k_steps,
                    compile_timeout=120,
                    slm_server=llm_server,
                    use_slm_judge=False,
                ),
            ]
        else:
            monitors = []
        
        try:
            answer = asyncio.run(
                stream_completion(
                    prompt,
                    prev_text="",
                    llm_server=llm_server,
                    monitors=monitors,
                    add_delay=False,
                    num_calls_index=0,
                    async_execution=True,
                )
            )
            prompt_with_answer = prompt + answer
            print("ANSWER: ", answer)
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            results.append({
                "idx": idx,
                "data_id": data.data_id,
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
                "reasoning_tokens": 0,
                "precond": "",
                "postcond": "",
                "num_times_forced": 0,
                "any_precond_generated": False,
                "any_postcond_generated": False,
            })
            continue
        
        save_reasoning_trace(idx, data.data_id, prompt_with_answer, reason_dir)
        
        # Extract generated spec
        generated_spec = extract_spec_from_response(answer)
        logger.info(f"Extracted precond: {generated_spec['precond'][:100]}...")
        logger.info(f"Extracted postcond: {generated_spec['postcond'][:100]}...")
        
        # Final spec verification loop - retry if compilation fails
        if args.monitor and monitors and generated_spec.get("postcond"):
            final_spec, final_compiles, final_output, num_final_retries = asyncio.run(
                monitors[0].verify_final_spec(
                    spec=generated_spec,
                    prompt_with_answer=prompt_with_answer,
                    max_retries=1
                )
            )
            if final_spec != generated_spec:
                logger.info(f"[Final verification] Spec fixed after {num_final_retries} retries")
                generated_spec = final_spec

        # check for soundness
        if args.monitor and monitors and generated_spec:
            compiled, _ = monitors[0].sync_verify_compilation(generated_spec)
        else:
            compiled = True
        
        # Evaluate
        eval_result = evaluate_generated_spec(data, generated_spec, idx)
        
        if eval_result["compiles"]:
            precond_sound_rate = eval_result["precond_sound_pass"] / max(1, eval_result["precond_sound_total"])
            precond_complete_rate = eval_result["precond_complete_pass"] / max(1, eval_result["precond_complete_total"])
            postcond_sound_rate = eval_result["postcond_sound_pass"] / max(1, eval_result["postcond_sound_total"])
            postcond_complete_rate = eval_result["postcond_complete_pass"] / max(1, eval_result["postcond_complete_total"])
            
            logger.info(f"✓ Compiles")
            logger.info(f"  Precond soundness: {eval_result['precond_sound_pass']}/{eval_result['precond_sound_total']} ({precond_sound_rate:.1%})")
            logger.info(f"  Precond completeness: {eval_result['precond_complete_pass']}/{eval_result['precond_complete_total']} ({precond_complete_rate:.1%})")
            logger.info(f"  Postcond soundness: {eval_result['postcond_sound_pass']}/{eval_result['postcond_sound_total']} ({postcond_sound_rate:.1%})")
            logger.info(f"  Postcond completeness: {eval_result['postcond_complete_pass']}/{eval_result['postcond_complete_total']} ({postcond_complete_rate:.1%})")
            logger.info(f"  Precond correct: {eval_result['precond_correct']} | Postcond correct: {eval_result['postcond_correct']}")
            logger.info(f"  Spec sound: {eval_result['spec_sound']} | Spec complete: {eval_result['spec_complete']}")
            logger.info(f"  ★ Full spec correct: {eval_result['full_spec_correct']}")
            
            if eval_result['full_spec_correct']:
                num_correct += 1
            logger.info(f"Running Accuracy so far: {(num_correct/(i+1))*100:.2f}%")
        else:
            logger.info(f"✗ FAIL - Compilation error")
            logger.debug(f"Error: {eval_result.get('compile_error', '')[:300]}")
            logger.info(f"Running Accuracy so far: {(num_correct/(i+1))*100:.2f}%")
        
        results.append({
            "idx": idx,
            "data_id": data.data_id,
            "compiles": eval_result["compiles"],
            "precond_sound_pass": eval_result["precond_sound_pass"],
            "precond_sound_total": eval_result["precond_sound_total"],
            "precond_complete_pass": eval_result["precond_complete_pass"],
            "precond_complete_total": eval_result["precond_complete_total"],
            "postcond_sound_pass": eval_result["postcond_sound_pass"],
            "postcond_sound_total": eval_result["postcond_sound_total"],
            "postcond_complete_pass": eval_result["postcond_complete_pass"],
            "postcond_complete_total": eval_result["postcond_complete_total"],
            "precond_correct": eval_result.get("precond_correct", False),
            "postcond_correct": eval_result.get("postcond_correct", False),
            "spec_sound": eval_result.get("spec_sound", False),
            "spec_complete": eval_result.get("spec_complete", False),
            "full_spec_correct": eval_result.get("full_spec_correct", False),
            "reasoning_tokens": 0,
            "precond": generated_spec["precond"],
            "postcond": generated_spec["postcond"],
            "num_times_forced": monitors[0].get_force_count() if monitors else 0,
            "finally_wrong": not compiled
        })
    
    # Save final results
    results_csv = os.path.join(output_dirs["base"], "verina_spec_results.csv")
    save_results_csv(results, results_csv)
    
    avg_tokens = compute_average_tokens(token_file)
    
    # Compute statistics
    num_compile = sum(1 for r in results if r["compiles"])
    
    total_precond_sound_pass = sum(r["precond_sound_pass"] for r in results)
    total_precond_sound_total = sum(r["precond_sound_total"] for r in results)
    total_precond_complete_pass = sum(r["precond_complete_pass"] for r in results)
    total_precond_complete_total = sum(r["precond_complete_total"] for r in results)
    total_postcond_sound_pass = sum(r["postcond_sound_pass"] for r in results)
    total_postcond_sound_total = sum(r["postcond_sound_total"] for r in results)
    total_postcond_complete_pass = sum(r["postcond_complete_pass"] for r in results)
    total_postcond_complete_total = sum(r["postcond_complete_total"] for r in results)
    
    compile_rate = num_compile / N if N > 0 else 0
    precond_sound_rate = total_precond_sound_pass / max(1, total_precond_sound_total)
    precond_complete_rate = total_precond_complete_pass / max(1, total_precond_complete_total)
    postcond_sound_rate = total_postcond_sound_pass / max(1, total_postcond_sound_total)
    postcond_complete_rate = total_postcond_complete_pass / max(1, total_postcond_complete_total)
    
    num_precond_correct = sum(1 for r in results if r.get("precond_correct", False))
    num_postcond_correct = sum(1 for r in results if r.get("postcond_correct", False))
    num_spec_sound = sum(1 for r in results if r.get("spec_sound", False))
    num_spec_complete = sum(1 for r in results if r.get("spec_complete", False))
    num_full_spec_correct = sum(1 for r in results if r.get("full_spec_correct", False))
    
    precond_correct_rate = num_precond_correct / N if N > 0 else 0
    postcond_correct_rate = num_postcond_correct / N if N > 0 else 0
    spec_sound_rate = num_spec_sound / N if N > 0 else 0
    spec_complete_rate = num_spec_complete / N if N > 0 else 0
    full_spec_correct_rate = num_full_spec_correct / N if N > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - SPECIFICATION GENERATION WITH STEP VERIFICATION")
    print(f"{'='*60}")
    print(f"Model: {main_model}")
    print(f"Total examples: {N}")
    print(f"Successful compilations: {num_compile} ({compile_rate:.2%})")
    print(f"\n--- Individual Metrics (test-level) ---")
    print(f"Precondition Soundness: {total_precond_sound_pass}/{total_precond_sound_total} ({precond_sound_rate:.2%})")
    print(f"Precondition Completeness: {total_precond_complete_pass}/{total_precond_complete_total} ({precond_complete_rate:.2%})")
    print(f"Postcondition Soundness: {total_postcond_sound_pass}/{total_postcond_sound_total} ({postcond_sound_rate:.2%})")
    print(f"Postcondition Completeness: {total_postcond_complete_pass}/{total_postcond_complete_total} ({postcond_complete_rate:.2%})")
    print(f"\n--- Combined Metrics (task-level) ---")
    print(f"Precond Fully Correct (sound+complete): {num_precond_correct}/{N} ({precond_correct_rate:.2%})")
    print(f"Postcond Fully Correct (sound+complete): {num_postcond_correct}/{N} ({postcond_correct_rate:.2%})")
    print(f"Spec Sound (precond+postcond sound): {num_spec_sound}/{N} ({spec_sound_rate:.2%})")
    print(f"Spec Complete (precond+postcond complete): {num_spec_complete}/{N} ({spec_complete_rate:.2%})")
    print(f"\n★ FULL SPEC CORRECT (all sound+complete): {num_full_spec_correct}/{N} ({full_spec_correct_rate:.2%})")
    print(f"\nAverage reasoning tokens: {avg_tokens:.2f}")
    print(f"Results saved to: {results_csv}")
    print(f"Reasoning traces saved to: {reason_dir}")
    
    # Save summary
    summary_file = os.path.join(output_dirs["base"], "summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "model": main_model,
            "earlystop_model": earlystop_model,
            "total_examples": N,
            "num_compile": num_compile,
            "compile_rate": compile_rate,
            "precond_sound_pass": total_precond_sound_pass,
            "precond_sound_total": total_precond_sound_total,
            "precond_sound_rate": precond_sound_rate,
            "precond_complete_pass": total_precond_complete_pass,
            "precond_complete_total": total_precond_complete_total,
            "precond_complete_rate": precond_complete_rate,
            "postcond_sound_pass": total_postcond_sound_pass,
            "postcond_sound_total": total_postcond_sound_total,
            "postcond_sound_rate": postcond_sound_rate,
            "postcond_complete_pass": total_postcond_complete_pass,
            "postcond_complete_total": total_postcond_complete_total,
            "postcond_complete_rate": postcond_complete_rate,
            "num_precond_correct": num_precond_correct,
            "precond_correct_rate": precond_correct_rate,
            "num_postcond_correct": num_postcond_correct,
            "postcond_correct_rate": postcond_correct_rate,
            "num_spec_sound": num_spec_sound,
            "spec_sound_rate": spec_sound_rate,
            "num_spec_complete": num_spec_complete,
            "spec_complete_rate": spec_complete_rate,
            "num_full_spec_correct": num_full_spec_correct,
            "full_spec_correct_rate": full_spec_correct_rate,
            "avg_reasoning_tokens": avg_tokens,
        }, f, indent=2)
    
    logger.info(f"Experiment completed. Summary saved to {summary_file}")
