"""
SpatialMap Step Verifier

Z3-based constraint solver for verifying spatial/directional relationships.
Used to check if model's directional claims are consistent with the given constraints.

Directions use a coordinate system where:
- North/up = y increases
- South/down = y decreases  
- East/right = x increases
- West/left = x decreases
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from z3 import Solver, Real, And, sat


class SpatialMapZ3Solver:
    """
    Z3-based constraint solver for spatial/directional relationships.
    """
    
    def __init__(self, problem_text: str = None):
        self.solver = Solver()
        self.solver.set(unsat_core=True)
        self.entities: Dict[str, Real] = {}
        self.parsed_relations: List[Dict] = []
        
        if problem_text:
            named_entities, relations = self.parse_problem(problem_text)
            self.entities = self.setup_vars(named_entities)
            for relation in relations:
                self.apply_ir(relation)
                self.parsed_relations.append(relation)
    
    def parse_problem(self, question: str) -> Tuple[Set[str], List[Dict]]:
        """
        Parse the problem text to extract entities and relations.
        
        Returns:
            (named_entities, relations)
        """
        # Parse out the constraints before "Please answer"
        question = question.split("Please answer")[0]
        question = question.replace('Consider a map with multiple objects:', '').strip()
        constraints = question.split('. ')
        constraints = [c.strip() for c in constraints if c.strip()]
        
        named_entities = set()
        relations = []
        
        # First constraint: "X is in the map"
        if constraints:
            first_match = re.match(r'^(.*?) is in the map', constraints[0])
            if first_match:
                entity = first_match.group(1).strip()
                named_entities.add(entity)
        
        # Remaining constraints: "X is to the DIRECTION of Y"
        for constraint in constraints[1:]:
            match = re.match(r'^(.*?) is to the (.*?) of (.*?)$', constraint)
            if match:
                entity1 = match.group(1).strip()
                direction = match.group(2).strip()
                entity2 = match.group(3).strip()
                
                # Clean up entity names (remove trailing punctuation)
                entity1 = re.sub(r'[,\.\!\?]+$', '', entity1).strip()
                entity2 = re.sub(r'[,\.\!\?]+$', '', entity2).strip()
                
                named_entities.add(entity1)
                named_entities.add(entity2)
                
                relations.append({
                    'A': entity1,
                    'direction': direction.lower(),
                    'B': entity2
                })
        
        return named_entities, relations
    
    def setup_vars(self, entity_names: Set[str]) -> Dict[str, Real]:
        """Setup Z3 x, y Real variables for each entity."""
        entities = {}
        for ent in entity_names:
            entities[f'{ent}_x'] = Real(f'{ent}_x')
            entities[f'{ent}_y'] = Real(f'{ent}_y')
        return entities
    
    def ensure_entity_vars(self, entity_name: str):
        """Ensure entity variables exist, create if not."""
        if f'{entity_name}_x' not in self.entities:
            self.entities[f'{entity_name}_x'] = Real(f'{entity_name}_x')
            self.entities[f'{entity_name}_y'] = Real(f'{entity_name}_y')
    
    def compile_constraint(self, ir: Dict):
        """
        Compile an IR constraint into a Z3 BoolRef.
        
        IR format: {
            "A": "entity1",
            "direction": "north" | "south" | "east" | "west" | 
                        "northeast" | "northwest" | "southeast" | "southwest",
            "B": "entity2",
        }
        
        Direction mappings (A is to the X of B):
        - northwest: A_x < B_x AND A_y > B_y (left and up)
        - northeast: A_x > B_x AND A_y > B_y (right and up)
        - southwest: A_x < B_x AND A_y < B_y (left and down)
        - southeast: A_x > B_x AND A_y < B_y (right and down)
        """
        A, B = ir['A'], ir['B']
        direction = ir['direction'].lower()
        
        # Ensure both entities have variables
        self.ensure_entity_vars(A)
        self.ensure_entity_vars(B)
        
        A_x = self.entities[f'{A}_x']
        A_y = self.entities[f'{A}_y']
        B_x = self.entities[f'{B}_x']
        B_y = self.entities[f'{B}_y']
        
        z3_constraints = []
        
        # For SpatialMap, directions are diagonal (NW/NE/SW/SE)
        # "A is to the northwest of B" means A is up-left of B
        # Up = higher y (north), Left = lower x (west)
        if 'north' in direction:
            z3_constraints.append(A_y > B_y)  # A is above B (higher y)
        if 'south' in direction:
            z3_constraints.append(A_y < B_y)  # A is below B (lower y)
        if 'east' in direction:
            z3_constraints.append(A_x > B_x)  # A is right of B (higher x)
        if 'west' in direction:
            z3_constraints.append(A_x < B_x)  # A is left of B (lower x)
        
        if z3_constraints:
            return And(*z3_constraints)
        return None
    
    def apply_ir(self, ir: Dict):
        """
        Apply an IR constraint to the solver.
        
        Args:
            ir: an IR constraint dict, or a list of IR constraint dicts.
        """
        if isinstance(ir, dict):
            ir = [ir]
        
        for entry in ir:
            compiled = self.compile_constraint(entry)
            if compiled is not None:
                self.solver.add(compiled)
    
    def check_with_new_constraint(self, ir: Dict) -> bool:
        """
        Check if adding a new constraint would keep the system satisfiable.
        Does NOT permanently add the constraint.
        
        Returns:
            True if satisfiable with the new constraint, False otherwise.
        """
        self.solver.push()  # Save state
        compiled = self.compile_constraint(ir)
        if compiled is not None:
            self.solver.add(compiled)
        result = self.solver.check() == sat
        self.solver.pop()  # Restore state
        return result
    
    @property
    def is_satisfiable(self) -> bool:
        return self.solver.check() == sat

    def count_objects_in_direction(
        self, reference: str, direction: str
    ) -> Optional[int]:
        """
        Count how many entities are in a **strict** direction from *reference*.

        For cardinal directions the semantics are strict:
        - "north"  → same x, higher y   (but see note below)
        - "south"  → same x, lower y
        - "east"   → higher x, same y
        - "west"   → lower x, same y

        However, since every constraint in the SpatialMap dataset is diagonal
        (NE/NW/SE/SW), no two objects can share an x- or y-coordinate.
        Therefore the strict-cardinal count is always **0** whenever the
        problem only has diagonal constraints — which is exactly the
        ground-truth expectation.

        For diagonal directions:
        - "northeast" → higher x AND higher y
        - "northwest" → lower x  AND higher y
        - "southeast" → higher x AND lower y
        - "southwest" → lower x  AND lower y

        Returns the count, or ``None`` if the solver cannot determine it
        (e.g. reference entity not found).
        """
        direction = direction.lower().strip()

        # Resolve the reference entity's variable names
        ref_x_key = f"{reference}_x"
        ref_y_key = f"{reference}_y"
        if ref_x_key not in self.entities:
            # Try fuzzy match — dataset names may differ in whitespace
            for key in self.entities:
                if key.endswith("_x") and reference.lower() in key.lower():
                    ref_x_key = key
                    ref_y_key = key.replace("_x", "_y")
                    reference = key[:-2]
                    break
            else:
                return None

        ref_x = self.entities[ref_x_key]
        ref_y = self.entities[ref_y_key]

        # Collect all other entity names (unique base names)
        all_entities = set()
        for key in self.entities:
            if key.endswith("_x"):
                ename = key[:-2]
                if ename != reference:
                    all_entities.add(ename)

        # Determine x/y constraints for the direction
        is_cardinal = direction in ("north", "south", "east", "west")

        # Since all given constraints are strictly diagonal, any pair of
        # objects cannot share the same x- or y-coordinate.  Cardinal
        # directions require an exact match on one axis, which is impossible.
        if is_cardinal:
            return 0

        # For diagonal directions, check each entity with Z3
        count = 0
        for ename in all_entities:
            e_x = self.entities[f"{ename}_x"]
            e_y = self.entities[f"{ename}_y"]

            if direction == "northeast":
                constraint = And(e_x > ref_x, e_y > ref_y)
            elif direction == "northwest":
                constraint = And(e_x < ref_x, e_y > ref_y)
            elif direction == "southeast":
                constraint = And(e_x > ref_x, e_y < ref_y)
            elif direction == "southwest":
                constraint = And(e_x < ref_x, e_y < ref_y)
            else:
                continue

            # Check if this entity MUST be in that direction
            # (i.e. the negation is unsatisfiable)
            self.solver.push()
            from z3 import Not
            self.solver.add(Not(constraint))
            must_be = self.solver.check() != sat
            self.solver.pop()

            if must_be:
                count += 1

        return count


def parse_counting_question(problem_text: str) -> Optional[Dict]:
    """
    If the problem asks a *counting* question ("How many objects are in
    the X of Y?"), return a dict with the direction and reference entity.

    Returns ``None`` for non-counting questions.
    """
    m = re.search(
        r'How many objects are in the (\w+) of ([^?]+?)\?',
        problem_text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "direction": m.group(1).strip().lower(),
        "reference": m.group(2).strip().rstrip("."),
    }


def parse_model_count_from_answer(text_after_think: str, options: dict = None) -> Optional[int]:
    """
    Extract the numeric count the model chose from its ``\\boxed{}`` answer.

    Looks for ``\\boxed{LETTER}`` then maps through *options* to get the
    numeric value.  Falls back to extracting a number directly.
    """
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text_after_think)
    if not boxed:
        return None
    answer = boxed[-1].strip()

    # If options mapping is provided, resolve letter → value
    if options and answer in options:
        try:
            return int(options[answer])
        except (ValueError, TypeError):
            return None

    # Try direct numeric
    try:
        return int(answer)
    except (ValueError, TypeError):
        return None


def parse_directional_claims_from_text(text: str) -> List[Dict]:
    """
    Parse directional claims from model output text.
    
    Looks for patterns like:
    - "X is to the northwest of Y"
    - "X is NORTHWEST of Y"
    - "X is northwest of Y" (affirmative claims)
    - "X is NW of Y" (abbreviated directions)
    - "[X] is to the northwest of [Y]" (bracket-wrapped names)
    
    Returns list of IR dicts: [{"A": ..., "direction": ..., "B": ...}, ...]
    """
    # Expand abbreviated directions before parsing
    abbrev_map = {
        'NW': 'northwest', 'NE': 'northeast',
        'SW': 'southwest', 'SE': 'southeast',
    }
    expanded_text = text
    for abbr, full in abbrev_map.items():
        # Replace standalone abbreviations like "is NE of" → "is northeast of"
        expanded_text = re.sub(
            rf'\b{abbr}\b(?=\s+of\b)', full, expanded_text
        )

    # Strip square brackets around entity names: [Foo Bar] → Foo Bar
    expanded_text = re.sub(r'\[([A-Z][A-Za-z\'\s]*?)\]', r'\1', expanded_text)

    claims = []
    
    # Pattern: "X is (to the) DIRECTION of Y"
    pattern = r"([A-Z][A-Za-z'][A-Za-z'\s]*?)\s+is\s+(?:to\s+the\s+)?(northwest|northeast|southwest|southeast|north|south|east|west)\s+of\s+([A-Z][A-Za-z'][A-Za-z'\s]*?)(?:\.|,|;|:|\s*[→✓✗]|\s*\n|\s*$|\s+(?:and|so|which|therefore|thus|but|since|because|while|whereas|however|hence|then|for|as|meaning|indicating|implying|suggesting|confirming|\())"
    
    matches = re.finditer(pattern, expanded_text, re.IGNORECASE)
    
    for match in matches:
        entity_a = match.group(1).strip()
        direction = match.group(2).strip().lower()
        entity_b = match.group(3).strip()
        
        # Clean up entity names
        entity_a = re.sub(r'[,\.\!\?]+$', '', entity_a).strip()
        entity_b = re.sub(r'[,\.\!\?]+$', '', entity_b).strip()
        
        # Skip if entities look like fragments, pronouns, or are too short
        skip_words = {'then', 'if', 'so', 'thus', 'therefore', 'it', 'this', 'that', 
                      'which', 'what', 'where', 'when', 'also', 'not', 'the', 'a', 'an'}
        if entity_a.lower() in skip_words or entity_b.lower() in skip_words:
            continue
        if len(entity_a) < 2 or len(entity_b) < 2:
            continue
        if not entity_a[0].isupper():
            continue
            
        claims.append({
            'A': entity_a,
            'direction': direction,
            'B': entity_b
        })
    
    return claims


def extract_step2_claims(answer_text: str) -> List[Dict]:
    """
    Extract directional claims specifically from STEP 2 of the answer.
    
    STEP 2 is where the model makes inferences/claims about relationships
    that should be verified against Z3.
    
    STEP 1 is just parsing the given relationships (no need to verify).
    """
    # Find STEP 2 section
    step2_pattern = re.compile(
        r'>>>\s*STEP\s*2[:\s].*?(?=>>>\s*STEP\s*3|>>>\s*FINAL|\\boxed|$)',
        re.DOTALL | re.IGNORECASE
    )
    
    match = step2_pattern.search(answer_text)
    if not match:
        return []
    
    step2_text = match.group(0)
    return parse_directional_claims_from_text(step2_text)


def verify_spatialmap_step(
    claim: Dict,
    z3_solver: SpatialMapZ3Solver,
    add_if_valid: bool = True
) -> Tuple[bool, List[str]]:
    """
    Verify a single directional claim against the Z3 solver.
    
    Args:
        claim: {"A": entity1, "direction": direction, "B": entity2}
        z3_solver: The Z3 solver with known constraints
        add_if_valid: If True, add the claim to the solver **only if it
            is entailed** (i.e. its negation is UNSAT).  Merely
            satisfiable claims are accepted but NOT committed to the
            solver so they cannot over-constrain future checks.
    
    Returns:
        (is_valid, errors)
    """
    from z3 import Not as Z3Not, sat as z3sat

    errors = []
    
    is_consistent = z3_solver.check_with_new_constraint(claim)
    
    if not is_consistent:
        errors.append(
            f"Contradiction: '{claim['A']} is to the {claim['direction']} of {claim['B']}' "
            f"contradicts the given spatial relationships."
        )
        return False, errors
    
    if add_if_valid:
        # Only commit the claim if it is *entailed* (negation is UNSAT).
        # This prevents merely-satisfiable-but-unproven claims from
        # over-constraining the solver and blocking valid solutions later.
        compiled = z3_solver.compile_constraint(claim)
        if compiled is not None:
            z3_solver.solver.push()
            z3_solver.solver.add(Z3Not(compiled))
            is_entailed = z3_solver.solver.check() != z3sat
            z3_solver.solver.pop()
            if is_entailed:
                z3_solver.apply_ir(claim)
    
    return True, []


def format_spatialmap_feedback(errors: List[str], claim: Optional[Dict] = None) -> str:
    """Format verification errors as feedback to append."""
    if not errors:
        return ""
    
    feedback = "\n\n[VERIFIER FEEDBACK: Contradiction detected!\n"
    for err in errors:
        feedback += f"  ✗ {err}\n"
    feedback += "Please reconsider this relationship and correct your reasoning.\n"
    feedback += "IMPORTANT: Always use the full location names (e.g., 'Old Town Artisanal Bakery', "
    feedback += "'Walrus Watches') instead of abbreviations or shorthand (e.g., 'O', 'W', 'P'). "
    feedback += "The verifier can only check claims with full names.]\n\n"
    return feedback


# ---------------------------------------------------------------------------
#  Direction-question helpers
# ---------------------------------------------------------------------------

def parse_direction_question(problem_text: str) -> Optional[Dict]:
    """
    If the problem asks a *direction* question
    ("In which direction is X relative to Y?"),
    return ``{"entity_a": X, "entity_b": Y}``.

    Returns ``None`` for non-direction questions.
    """
    m = re.search(
        r'In which direction is (.+?) relative to (.+?)\?',
        problem_text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "entity_a": m.group(1).strip(),
        "entity_b": m.group(2).strip(),
    }


def parse_object_question(problem_text: str) -> Optional[Dict]:
    """
    If the problem asks an *object* question
    ("Which object is in the [direction] of [entity]?"),
    return ``{"direction": ..., "reference": ...}``.

    Returns ``None`` for non-object questions.
    """
    m = re.search(
        r'Which object is (?:located )?(?:to the |in the )'
        r'(northeast|northwest|southeast|southwest|north|south|east|west)'
        r' of (.+?)\?',
        problem_text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return {
        "direction": m.group(1).strip().lower(),
        "reference": m.group(2).strip().rstrip("."),
    }


def parse_model_boxed_answer(
    text_after_think: str, options: Dict[str, str]
) -> Optional[str]:
    """
    Extract the text value the model chose from its ``\\boxed{}`` answer.
    Maps letter → option text using *options* dict.
    Returns the raw option text (lowercase stripped) or None.
    """
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text_after_think)
    if not boxed:
        return None
    answer = boxed[-1].strip().upper()
    if answer in options:
        return options[answer].strip().lower().rstrip(".")
    # Try the raw value
    return answer.lower()


def get_possible_directions(
    solver: SpatialMapZ3Solver,
    entity_a: str,
    entity_b: str,
) -> List[str]:
    """
    Return the list of diagonal directions (NE/NW/SE/SW) that are
    *satisfiable* for entity_a relative to entity_b under the current
    constraints.

    ``entity_a`` and ``entity_b`` are matched fuzzily against solver
    entity names.
    """
    from z3 import And as Z3And, sat as z3sat

    def _find(name):
        nl = name.lower()
        for k in solver.entities:
            if k.endswith('_x') and k[:-2].lower() == nl:
                return k[:-2]
        for k in solver.entities:
            if k.endswith('_x') and (nl in k[:-2].lower() or k[:-2].lower() in nl):
                return k[:-2]
        return None

    ba = _find(entity_a)
    bb = _find(entity_b)
    if not ba or not bb:
        return ['northeast', 'northwest', 'southeast', 'southwest']

    ax = solver.entities[f'{ba}_x']
    ay = solver.entities[f'{ba}_y']
    bx = solver.entities[f'{bb}_x']
    by = solver.entities[f'{bb}_y']

    dir_constraints = {
        'northeast': Z3And(ax > bx, ay > by),
        'northwest': Z3And(ax < bx, ay > by),
        'southeast': Z3And(ax > bx, ay < by),
        'southwest': Z3And(ax < bx, ay < by),
    }

    possible = []
    for dname, dc in dir_constraints.items():
        solver.solver.push()
        solver.solver.add(dc)
        if solver.solver.check() == z3sat:
            possible.append(dname)
        solver.solver.pop()

    return possible if possible else ['northeast', 'northwest', 'southeast', 'southwest']


def get_consistent_object_options(
    solver: SpatialMapZ3Solver,
    direction: str,
    reference: str,
    options: Dict[str, str],
) -> List[str]:
    """
    For an *object* question, return the list of MCQ letters whose entity
    *could* be in ``direction`` of ``reference`` (Z3-satisfiable).

    Letters whose entities cannot be found in the solver are kept as
    "possible" (benefit of the doubt).
    """
    from z3 import And as Z3And, sat as z3sat

    def _find(name):
        nl = name.lower()
        for k in solver.entities:
            if k.endswith('_x') and k[:-2].lower() == nl:
                return k[:-2]
        for k in solver.entities:
            if k.endswith('_x') and (nl in k[:-2].lower() or k[:-2].lower() in nl):
                return k[:-2]
        return None

    ref_base = _find(reference)
    if not ref_base:
        return list(options.keys())  # can't check, keep all

    rx = solver.entities[f'{ref_base}_x']
    ry = solver.entities[f'{ref_base}_y']

    dfunc = {
        'northeast': lambda ox, oy: Z3And(ox > rx, oy > ry),
        'northwest': lambda ox, oy: Z3And(ox < rx, oy > ry),
        'southeast': lambda ox, oy: Z3And(ox > rx, oy < ry),
        'southwest': lambda ox, oy: Z3And(ox < rx, oy < ry),
    }.get(direction.lower())
    if not dfunc:
        return list(options.keys())

    consistent = []
    for letter, opt_name in options.items():
        opt_base = _find(opt_name.strip().rstrip('.'))
        if not opt_base:
            consistent.append(letter)  # can't verify, assume possible
            continue
        ox = solver.entities[f'{opt_base}_x']
        oy = solver.entities[f'{opt_base}_y']
        solver.solver.push()
        solver.solver.add(dfunc(ox, oy))
        if solver.solver.check() == z3sat:
            consistent.append(letter)
        solver.solver.pop()

    return consistent


def get_possible_count_range(
    solver: SpatialMapZ3Solver,
    reference: str,
    direction: str,
) -> Optional[Tuple[int, int]]:
    """
    Compute the *[min, max]* range of how many entities could be in
    ``direction`` of ``reference`` across all satisfying assignments.

    Uses Z3 must-be / can-be checks per entity:
    - *must_be*:  negation is UNSAT → entity is ALWAYS in that direction
    - *can_be*:   adding constraint is SAT → entity COULD be there

    min = count(must_be), max = count(must_be) + count(maybe)

    Returns ``None`` if the reference entity cannot be found.
    """
    from z3 import And as Z3And, Not as Z3Not, sat as z3sat

    direction = direction.lower().strip()
    if direction in ('north', 'south', 'east', 'west'):
        return (0, 0)  # cardinal → always 0 with diagonal-only constraints

    def _find(name):
        nl = name.lower()
        for k in solver.entities:
            if k.endswith('_x') and k[:-2].lower() == nl:
                return k[:-2]
        for k in solver.entities:
            if k.endswith('_x') and (nl in k[:-2].lower() or k[:-2].lower() in nl):
                return k[:-2]
        return None

    ref_base = _find(reference)
    if not ref_base:
        return None

    rx = solver.entities[f'{ref_base}_x']
    ry = solver.entities[f'{ref_base}_y']

    others = [
        k[:-2] for k in solver.entities
        if k.endswith('_x') and k[:-2] != ref_base
    ]

    dfunc = {
        'northeast': lambda ox, oy: Z3And(ox > rx, oy > ry),
        'northwest': lambda ox, oy: Z3And(ox < rx, oy > ry),
        'southeast': lambda ox, oy: Z3And(ox > rx, oy < ry),
        'southwest': lambda ox, oy: Z3And(ox < rx, oy < ry),
    }.get(direction)
    if not dfunc:
        return None

    must_count = 0
    maybe_count = 0

    for ename in others:
        ex = solver.entities[f'{ename}_x']
        ey = solver.entities[f'{ename}_y']
        c = dfunc(ex, ey)

        # Can it be in that direction?
        solver.solver.push()
        solver.solver.add(c)
        can_be = solver.solver.check() == z3sat
        solver.solver.pop()

        if not can_be:
            continue

        # Must it be?
        solver.solver.push()
        solver.solver.add(Z3Not(c))
        must_be = solver.solver.check() != z3sat
        solver.solver.pop()

        if must_be:
            must_count += 1
        else:
            maybe_count += 1

    return (must_count, must_count + maybe_count)


# Export
__all__ = [
    'SpatialMapZ3Solver',
    'parse_directional_claims_from_text',
    'parse_counting_question',
    'parse_model_count_from_answer',
    'parse_direction_question',
    'parse_object_question',
    'parse_model_boxed_answer',
    'get_possible_directions',
    'get_consistent_object_options',
    'get_possible_count_range',
    'extract_step2_claims',
    'verify_spatialmap_step',
    'format_spatialmap_feedback',
]
