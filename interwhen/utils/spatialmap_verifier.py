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


def parse_directional_claims_from_text(text: str) -> List[Dict]:
    """
    Parse directional claims from model output text.
    
    Looks for patterns like:
    - "X is to the northwest of Y"
    - "X is NORTHWEST of Y"
    - "X is northwest of Y" (affirmative claims)
    
    Returns list of IR dicts: [{"A": ..., "direction": ..., "B": ...}, ...]
    """
    claims = []
    
    # Pattern: "X is (to the) DIRECTION of Y"
    pattern = r"([A-Z][A-Za-z'][A-Za-z'\s]*?)\s+is\s+(?:to\s+the\s+)?(northwest|northeast|southwest|southeast|north|south|east|west)\s+of\s+([A-Z][A-Za-z'][A-Za-z'\s]*?)(?:\.|,|\s*[→✓✗]|\s*$|\s+(?:and|so|which|therefore|thus|but|\())"
    
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
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
        if len(entity_a) < 3 or len(entity_b) < 3:
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
        add_if_valid: If True, add the claim to the solver if it's valid
    
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    is_consistent = z3_solver.check_with_new_constraint(claim)
    
    if not is_consistent:
        errors.append(
            f"Contradiction: '{claim['A']} is to the {claim['direction']} of {claim['B']}' "
            f"contradicts the given spatial relationships."
        )
        return False, errors
    
    if add_if_valid:
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


# Export
__all__ = [
    'SpatialMapZ3Solver',
    'parse_directional_claims_from_text',
    'extract_step2_claims',
    'verify_spatialmap_step',
    'format_spatialmap_feedback',
]
