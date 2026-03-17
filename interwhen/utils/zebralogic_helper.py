"""
Prompt and dataset utilities for WildEval/ZebraLogic.
"""

import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from importlib import resources
import datasets


# ============== Prompt Templates ==============

SYSTEM_PROMPT_VANILLA = """\
# Problem Description

You are solving a house grid problem. You are given:
1. Features and Domains
    - A fixed number of houses, indexed sequentially (e.g., House 1, House 2, …) from left to right.
    - A set of features (e.g., color, name, pet, book genre).
    - Each feature has a finite domain of possible values.
2. Constraint:
    - Each house has exactly one value per feature.
    - No two houses share the same value for the same feature.
3. Clues / Constraints descrbing:
    - Houses and their positions
    - Feature values
    - Relative ordering (e.g., "next to", "to the left of", "2 houses away from")

Solve to your best ability the arrangement of features across the houses.

# Final Answer Format

```json
{
    "House 1": { "feature1": "value1", "feature2": "value2", ... },
    "House 2": { "feature1": "value1", "feature2": "value2", ... },
    ...
}
```

Make sure to use the exact feature/value names as given in the problem description.
Make sure the JSON is valid and parsable."""

SYSTEM_PROMPT_STATEEXTRACT = """\
# Problem Description

You are solving a house grid problem. You are given:
1. Features and Domains
    - A fixed number of houses, indexed sequentially (e.g., House 1, House 2, …) from left to right.
    - A set of features (e.g., color, name, pet, book genre).
    - Each feature has a finite domain of possible values.
2. Constraint:
    - Each house has exactly one value per feature.
    - No two houses share the same value for the same feature.
3. Clues / Constraints describing:
    - Houses and their positions
    - Feature values
    - Relative ordering (e.g., "next to", "to the left of", "2 houses away from")

# Rules for Solving

1. Reason about the problem in text.
2. After every inference, no matter how minor or tentative, immediately report the updated partial assignments.
    - Always output partial assignments frequently as the reasoning progresses, not only at major steps or when confident.
    - If an inference adds, removes, or narrows even a single possibility, report it.

# House/Feature Partial Assignment Reporting Format

```json
{
    "House N": { "feature1": "value1", "feature2": "value2", ... },
    ...
}
```

Omit any unassigned features.
Make sure to use the exact feature/value names as given in the problem description.
Make sure the JSON is valid and parsable."""

SYSTEM_PROMPT_VANILLA = """\
# Problem Description

You are solving a house grid problem. You are given:
1. Features and Domains
    - A fixed number of houses, indexed sequentially (e.g., House 1, House 2, …) from left to right.
    - A set of features (e.g., color, name, pet, book genre).
    - Each feature has a finite domain of possible values.
2. Constraint:
    - Each house has exactly one value per feature.
    - No two houses share the same value for the same feature.
3. Clues / Constraints describing:
    - Houses and their positions
    - Feature values
    - Relative ordering (e.g., "next to", "to the left of", "2 houses away from")

# Rules for Solving

1. Reason about the problem in text.
2. You may receive feedback from the user if anything is wrong. Use any feedback to guide your reasoning until a complete solution is reached.
3. Do not stop responding until you've assigned each and every variable.

# Final Answer Reporting Format

```json
{
    "House 1": { "feature1": "value1", "feature2": "value2", ... },
    "House 2": { "feature1": "value1", "feature2": "value2", ... },
    ...
}
```

Make sure to use the exact feature/value names as given in the problem description.
Make sure the JSON is valid and parsable."""

USER_PROMPT_TEMPLATE = "{problem_text}"

# ============== Dataset Loading ==============

def clean_problem_text(problem_text: str, features: dict) -> str:
    """Clean up problem text giving explicit feature domains."""
    desc, clues = problem_text.split('## clues:')
    line0 = desc.splitlines()[0]

    feature_text = ''
    for feature, values in features.items():
        values_str = ', '.join(f"'{v}'" for v in values)
        feature_text += f"- '{feature}': {values_str}\n"

    return f"{line0}\n{feature_text}\n## clues:{clues}".strip()


def process_zebralogic_problem(problem: dict, ir_map: dict) -> dict:
    """Process a raw ZebraLogic problem into the format needed by ZebraLogicProblem.

    Args:
        problem: Raw problem dict from the HuggingFace dataset.
        ir_map: Dict mapping problem IDs to their IR representations.

    Returns:
        Processed problem dict with keys: n_houses, n_features, features, clues,
        clue_irs, solution_irs, solution, puzzle_clean, etc.
    """
    def apply_text_replacements(problem):
        replacements = [
            ['january', 'jan'], ['february', 'feb'], ['march', 'mar'],
            ['august', 'aug'], ['september', 'sept'], ['f-150', 'f150'],
            ['animal', 'pet'], ['loves the spaghetti eater', 'loves spaghetti'],
            ['very short', 'veryshort'], ['super short', 'supershort'],
            ['very tall', 'verytall'], ['super tall', 'supertall'],
        ]
        problem_str = json.dumps(problem)
        for old, new in replacements:
            problem_str = problem_str.replace(old, new)
        return json.loads(problem_str)

    pid = problem['id']
    size = problem["size"]
    n_houses, n_features = map(int, size.split("*"))

    problem_text = problem["puzzle"].lower()
    clues_raw = re.split(r"##\s*clues\s*:", problem_text, flags=re.IGNORECASE)[1].strip().split("\n")
    clues = []
    for clue in clues_raw:
        clue_text_index, clue_text = clue.strip().split(". ", 1)
        clues.append({
            "text_index": int(clue_text_index.strip()),
            "text": clue_text.strip()
        })

    solution = problem["solution"]
    solution['header'] = [h.lower() for h in solution['header']]
    solution['rows'] = [[v.lower() for v in row] for row in solution['rows']]

    features = defaultdict(list)
    for row in solution['rows']:
        for fname, value in zip(solution['header'], row):
            if fname.lower() == 'house':
                continue
            features[fname].append(value)
    features = dict(features)
    for fname in features:
        features[fname] = sorted(list(set(features[fname])))
        assert len(features[fname]) == n_houses
    assert len(features) == n_features

    processed_solution = {f'House {i+1}': {} for i in range(n_houses)}
    for house_i, row in enumerate(solution['rows']):
        for fname, value in zip(solution['header'][1:], row[1:]):
            processed_solution[f'House {house_i+1}'][fname.lower()] = value.lower()
    problem['solution'] = processed_solution

    problem['puzzle'] = problem_text
    problem["clues"] = clues
    problem["features"] = features
    problem["n_houses"] = n_houses
    problem["n_features"] = n_features
    problem["clue_irs"] = ir_map[pid]["clue_irs"]
    problem["solution_irs"] = ir_map[pid]["solution_irs"]
    problem['puzzle_clean'] = clean_problem_text(problem['puzzle'], problem['features'])

    return apply_text_replacements(problem)

def get_zebralogic_dataset() -> list:
    """Load and process the ZebraLogic dataset from HuggingFace.

    Loads WildEval/ZebraLogic grid_mode test split and processes each problem
    with the IR map.

    The IR map file must be at interwhen/data/zebralogic_ir_map.json

    Returns:
        List of processed problem dicts.
    """

    dataset = datasets.load_dataset("WildEval/ZebraLogic", "grid_mode", split="test").to_list()

    pkg = "interwhen.data"
    with resources.files(pkg).joinpath("zebralogic_ir_map.json").open("r") as f:
        ir_map = json.load(f)

    # Known problematic problem IDs (unsolvable or malformed)
    bad_ids = {
        'lgp-test-6x5-2', 'lgp-test-6x6-5', 'lgp-test-2x5-1', 'lgp-test-4x5-5',
        'lgp-test-2x4-6', 'lgp-test-2x6-11', 'lgp-test-4x6-35', 'lgp-test-3x5-15',
        'lgp-test-5x5-37', 'lgp-test-5x5-17', 'lgp-test-4x5-15', 'lgp-test-6x6-2',
        'lgp-test-5x6-4', 'lgp-test-5x6-2', 'lgp-test-5x5-1'
    }
    dataset = [p for p in dataset if p['id'] not in bad_ids]
    dataset = [process_zebralogic_problem(p, ir_map) for p in dataset]
    return dataset

def extract_last_json(text):
    """Extract the last JSON object from the model's output text."""
    json_text = text.split('</think>')[-1].strip()
    
    # try with md tags
    matches = re.findall(r'```json(.*?)```', json_text, re.DOTALL)
    if matches and len(matches) > 0:
        json_match = matches[-1]
        return json.loads(json_match.strip())
    
    # try without assuming md tags
    matches = re.findall(r'\{.*\}\s*?}', json_text, re.DOTALL)
    if matches and len(matches) > 0:
        json_match = matches[0]
        return json.loads(json_match.strip())
    
    return None

def zebra_correctness(problem: dict, candidate_solution: dict) -> Tuple[int, int, int, int]:
    """Check candidate solution against ground truth.

    Args:
        problem: Processed problem dict with 'solution', 'n_houses', 'n_features', 'features'.
        candidate_solution: Dict mapping "House N" -> {feature: value}.

    Returns:
        (correct, skipped, missing, total) counts where:
            correct: Number of matching assignments
            skipped: Assignments for invalid houses/features
            missing: Features not in candidate solution
            total: Total expected assignments (n_houses * n_features)
    """
    c, s = 0, 0
    t_soln = 0
    t = problem['n_houses'] * problem['n_features']
    solution = problem['solution']

    for house in candidate_solution:
        if house not in solution:
            s += len(problem['features'])
            continue
        for fname in candidate_solution[house]:
            if fname not in solution[house]:
                s += 1
                continue
            t_soln += 1
            if candidate_solution[house][fname] == solution[house][fname]:
                c += 1

    m = t - t_soln
    return c, s, m, t
