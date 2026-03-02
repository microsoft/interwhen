"""
Tree of Thought implementation for interwhen-style streaming completion.

Implements proper ToT search using:
1. Propose function to generate candidate next steps
2. Value function to evaluate intermediate states
3. Search algorithm (BFS/DFS/beam) to explore the tree
4. Integrated with interwhen's async streaming architecture
"""

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from .value_prompts import (
    build_game24_value_prompt,
    build_mcq_value_prompt,
    build_tot_value_prompt as build_tot_value_prompt_impl,
)

logger = logging.getLogger(__name__)


# --------------------- Dataset prompt helpers ---------------------

def build_game24_prompt(nums: List[int]) -> str:
    """Return the canonical Game24 instruction block used across baselines."""
    if len(nums) != 4:
        raise ValueError("Game24 requires exactly four numbers.")
    a, b, c, d = nums
    boxed = r"\\boxed{}"
    return (
        "You are solving the Game of 24.\n\n"
        f"You are given four numbers: {a}, {b}, {c}, {d}\n\n"
        "Your job is to produce a valid arithmetic expression using:\n"
        "- ALL four numbers exactly once\n- ONLY +, -, *, /\n"
        "- The expression must evaluate to exactly 24.\n\n"
        "Please reason step by step, and put your final answer containing"
        f" only the expression within {boxed}."
    )


def build_maze_prompt(example: Dict[str, Any]) -> str:
    """Construct the maze reasoning instructions used in other pipelines."""
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following "
        "multiple-choice question and think through the solution step-by-step "
        "before providing your final answer. Provide the final answer option by "
        "enclosing it within \\boxed{A/B/C/D}."
    )
    description = str(example.get("prompt", ""))
    return f"{pre_prompt}\n\n{description.strip()}"


def build_spatialmap_prompt(example: Dict[str, Any]) -> str:
    """Construct the spatial reasoning instructions for TOT experiments."""
    pre_prompt = (
        "You are an expert problem solver. Carefully read the following "
        "multiple-choice question and think through the solution step-by-step "
        "before providing your final answer. Provide the final answer option by "
        "enclosing it within \\boxed{A/B/C/D}."
    )
    description = str(example.get("prompt", ""))
    return f"{pre_prompt}\n\n{description.strip()}"


def build_zebralogic_prompt(example: Dict[str, Any]) -> str:
    """Construct the Zebra Logic puzzle solving instructions for TOT experiments."""
    puzzle_text = str(example.get("puzzle", ""))
    prompt = (
        "# Problem Description\n\n"
        "You are solving a house grid logic puzzle. You are given:\n"
        "1. Features and Domains\n"
        "    - A fixed number of houses, indexed sequentially (e.g., House 1, House 2, …) from left to right.\n"
        "    - A set of features (e.g., color, name, pet, book genre).\n"
        "    - Each feature has a finite domain of possible values.\n"
        "2. Constraints:\n"
        "    - Each house has exactly one value per feature.\n"
        "    - No two houses share the same value for the same feature.\n"
        "3. Clues / Constraints describing:\n"
        "    - Houses and their positions\n"
        "    - Feature values\n"
        "    - Relative ordering (e.g., 'next to', 'to the left of', '2 houses away from')\n\n"
        "Solve this puzzle to your best ability by determining the arrangement of features across the houses.\n\n"
        "# Puzzle\n\n"
        f"{puzzle_text}\n\n"
        "# Solution Format\n\n"
        "Provide your final answer in this exact JSON format:\n"
        "```json\n"
        '{\n'
        '    "House 1": { "feature1": "value1", "feature2": "value2", ... },\n'
        '    "House 2": { "feature1": "value1", "feature2": "value2", ... },\n'
        '    ...\n'
        '}\n'
        "```\n\n"
        "Make sure to use the exact feature/value names as given in the puzzle.\n"
        "Ensure the JSON is valid and parsable."
    )
    return prompt


def build_tot_problem(task: str, example: Dict[str, Any], nums: Optional[List[int]] = None) -> str:
    """Helper that mirrors the best-of-k prompt builders for ToT runs."""
    task_lower = task.lower()
    if task_lower == "game24":
        numbers = nums or example.get("numbers")
        if numbers is None:
            raise ValueError("Game24 prompt requires 'numbers' in the example")
        return build_game24_prompt(list(numbers))
    if task_lower == "maze":
        return build_maze_prompt(example)
    if task_lower == "spatialmap":
        return build_spatialmap_prompt(example)
    if task_lower == "zebralogic":
        return build_zebralogic_prompt(example)
    raise ValueError(f"Unsupported task for ToT prompt building: {task}")


def build_tot_value_prompt(
    task: str,
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """
    Build value prompt for Tree of Thought evaluation.
    
    Args:
        task: The task type (e.g., "game24", "maze", "spatialmap")
        problem: The original problem statement
        trajectory: Current partial solution (or 'No progress yet' if empty)
        use_fewshot: Whether to use few-shot examples (default True for better evaluation)
        
    Returns:
        Formatted value prompt with or without few-shot examples
    """
    if not trajectory.strip():
        trajectory = "No progress yet"
    return build_tot_value_prompt_impl(task, problem, trajectory, use_fewshot=use_fewshot)


class SearchMethod(Enum):
    """Search algorithm types"""
    BFS = "bfs"
    DFS = "dfs"
    BEAM = "beam"


@dataclass
class TreeNode:
    """Represents a node in the Tree of Thought"""
    trajectory: str
    depth: int
    value: float = 0.5
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    is_terminal: bool = False
    proposals: List[str] = field(default_factory=list)
    evaluation_log: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.trajectory)
    
    def __eq__(self, other):
        return isinstance(other, TreeNode) and self.trajectory == other.trajectory


@dataclass
class ToTSearchConfig:
    """Configuration for Tree of Thought search"""
    branching_factor: int = 4
    max_depth: int = 6
    search_method: SearchMethod = SearchMethod.BFS
    beam_width: int = 2
    
    # Value thresholds
    sure_threshold: float = 0.7
    likely_threshold: float = 0.5
    impossible_threshold: float = 0.2
    
    # Optimization settings
    early_termination: bool = True
    cache_evaluations: bool = True
    max_candidates_per_level: int = 3


class TreeOfThoughtSearch:
    """
    Tree of Thought search controller compatible with interwhen's streaming.
    
    Provides propose/evaluate/search methods that work with vLLM API calls
    via the llm_server interface used in interwhen.
    """
    
    def __init__(self, config: ToTSearchConfig = None):
        self.config = config or ToTSearchConfig()
        self.evaluation_cache = {}
        self.proposal_cache = {}
        self.search_stats = {
            "nodes_explored": 0,
            "evaluations_performed": 0,
            "branches_pruned": 0,
            "cache_hits": 0,
            "solutions_found": 0,
            "total_nodes_in_tree": 0,
        }
        self.decision_tree = []
        self.root = None
    
    # ===================== PROPOSE FUNCTION =====================
    
    async def propose_next_steps(
        self,
        task: str,
        problem: str,
        current_trajectory: str,
        llm_server: Dict,
        num_proposals: Optional[int] = None,
    ) -> List[str]:
        """
        Generate candidate next steps using the model's propose capability.
        
        Args:
            task: The task type (e.g., "game24", "maze", "spatialmap")
            problem: The original problem statement
            current_trajectory: Current partial solution
            llm_server: vLLM server config (url, headers, payload template)
            num_proposals: Number of proposals to generate (defaults to branching_factor)
            
        Returns:
            List of proposed next steps
        """
        if num_proposals is None:
            num_proposals = self.config.branching_factor
        
        # Check cache
        cache_key = f"propose_{hash(problem)}_{hash(current_trajectory)}"
        if self.config.cache_evaluations and cache_key in self.proposal_cache:
            self.search_stats["cache_hits"] += 1
            return self.proposal_cache[cache_key]
        
        self.search_stats["nodes_explored"] += 1
        
        # Build propose prompt
        propose_prompt = self._build_propose_prompt(
            task,
            problem, 
            current_trajectory, 
            num_proposals
        )
        
        # Call model with streaming
        proposal_text = await self._call_llm_streaming(
            llm_server,
            propose_prompt
        )
        
        # Parse proposals from response
        proposals = self._parse_proposals(proposal_text, num_proposals)

        logger.info(
            "Generated %d proposals at depth hint=%s",
            len(proposals),
            "root" if not current_trajectory.strip() else "non-root",
        )
        
        # Log decision point
        decision_log = {
            "type": "proposal_generation",
            "timestamp": time.time(),
            "problem_hash": hash(problem),
            "trajectory": current_trajectory,
            "prompt": propose_prompt,
            "prompt_preview": propose_prompt[:200] + "..." if len(propose_prompt) > 200 else propose_prompt,
            "raw_response": proposal_text,
            "raw_response_preview": proposal_text[:300] + "..." if len(proposal_text) > 300 else proposal_text,
            "parsed_proposals": proposals,
        }
        self.decision_tree.append(decision_log)
        
        # Cache
        if self.config.cache_evaluations:
            self.proposal_cache[cache_key] = proposals
        
        return proposals
    
    def _build_propose_prompt(
        self,
        task: str,
        problem: str,
        trajectory: str,
        num_proposals: int
    ) -> str:
        """Build a prompt requesting proposals for next steps."""
        if task == "maze":
            return self._build_maze_propose_prompt(problem, trajectory, num_proposals)
        if task == "spatialmap":
            return self._build_spatialmap_propose_prompt(problem, trajectory, num_proposals)

        return f"""Given the following problem and current progress, propose {num_proposals} possible next steps.

PROBLEM:
{problem}

CURRENT PROGRESS/TRAJECTORY:
{trajectory if trajectory.strip() else "Starting fresh - no progress yet"}

Generate {num_proposals} distinct next steps that could advance the solution. Be specific and actionable.

Format each proposal clearly, one per line:
1. [Proposal 1]
2. [Proposal 2]
...

Think step by step about what makes each proposal viable.
"""

    def _detect_maze_question_type(self, problem: str) -> str:
        """Detect maze subtype for proposal steering (Q0/Q2/Q4)."""
        lower = problem.lower()
        if "how many right turns" in lower:
            logger.debug("Detected Q0: right turn counting")
            return "q0"
        if "how many turns" in lower and "right turns" not in lower:
            logger.debug("Detected Q2: total turn counting")
            return "q2"
        if "starting from s" in lower and "where is e" in lower:
            logger.debug("Detected Q4: spatial relation")
            return "q4"
        if "relative" in lower and "s" in lower and "e" in lower:
            logger.debug("Detected Q4: spatial relation (relative)")
            return "q4"
        logger.warning(f"Maze question type not recognized, using generic. Problem preview: {lower[:200]}")
        return "generic"

    def _extract_last_move_info(self, trajectory: str) -> dict:
        """Extract previous direction and counter from last step in trajectory."""
        if not trajectory.strip():
            return {"prev_direction": None, "right_count": 0, "left_count": 0, "total_count": 0}
        
        lines = trajectory.strip().split('\n')
        last_line = lines[-1] if lines else ""
        
        # Try to extract direction from "Next move: [DIRECTION]"
        import re
        direction_match = re.search(r'Next move:\s*(UP|DOWN|LEFT|RIGHT)', last_line, re.IGNORECASE)
        prev_direction = direction_match.group(1).upper() if direction_match else None
        
        # Extract counters
        right_match = re.search(r'Right-turn count:\s*(\d+)', last_line)
        left_match = re.search(r'Left-turn count:\s*(\d+)', last_line)
        total_match = re.search(r'Total-turn count:\s*(\d+)', last_line)
        
        right_count = int(right_match.group(1)) if right_match else 0
        left_count = int(left_match.group(1)) if left_match else 0
        total_count = int(total_match.group(1)) if total_match else 0
        
        return {
            "prev_direction": prev_direction,
            "right_count": right_count,
            "left_count": left_count,
            "total_count": total_count
        }

    def _build_maze_propose_prompt(
        self,
        problem: str,
        trajectory: str,
        num_proposals: int,
    ) -> str:
        """Build maze-specific atomic next-step proposal prompts by question type."""
        question_type = self._detect_maze_question_type(problem)
        logger.debug(f"Detected maze question type: {question_type}")
        
        # Extract previous move info for bookkeeping
        prev_info = self._extract_last_move_info(trajectory)
        
        if not trajectory.strip():
            current = "Starting fresh - no progress yet"
            last_step_hint = ""
        else:
            current = trajectory.strip()
            lines = current.split('\n')
            last_line = lines[-1] if lines else ""
            last_step_hint = f"\nLAST COMPLETED STEP: {last_line}\nNow generate the NEXT move after this (do NOT repeat this move).\n"

        if question_type == "q0":
            prev_dir = prev_info["prev_direction"]
            prev_count = prev_info["right_count"]
            
            if prev_dir is None:
                # First move - all directions result in STRAIGHT with count 0
                examples = f"""PARENT: first move, count=0

Valid answers (pick {num_proposals}):
Next move: UP | Turn: STRAIGHT | Right-turn count: 0
Next move: DOWN | Turn: STRAIGHT | Right-turn count: 0
Next move: LEFT | Turn: STRAIGHT | Right-turn count: 0
Next move: RIGHT | Turn: STRAIGHT | Right-turn count: 0"""
            else:
                # Define turn mappings
                turn_map = {
                    "UP": {"RIGHT": "RIGHT", "LEFT": "LEFT", "UP": "STRAIGHT", "DOWN": "STRAIGHT"},
                    "DOWN": {"LEFT": "RIGHT", "RIGHT": "LEFT", "DOWN": "STRAIGHT", "UP": "STRAIGHT"},
                    "LEFT": {"UP": "RIGHT", "DOWN": "LEFT", "LEFT": "STRAIGHT", "RIGHT": "STRAIGHT"},
                    "RIGHT": {"DOWN": "RIGHT", "UP": "LEFT", "RIGHT": "STRAIGHT", "LEFT": "STRAIGHT"}
                }
                
                moves = turn_map.get(prev_dir, {})
                examples_list = []
                for next_dir, turn_type in moves.items():
                    new_count = prev_count + 1 if turn_type == "RIGHT" else prev_count
                    examples_list.append(f"Next move: {next_dir} | Turn: {turn_type} | Right-turn count: {new_count}")
                
                examples = f"""PARENT: direction={prev_dir}, count={prev_count}

Valid answers (pick {num_proposals}):
{chr(10).join(examples_list)}"""
            
            return f"""{examples}

DO NOT explain. DO NOT reason. Just output {num_proposals} lines from above."""

        if question_type == "q2":
            prev_dir = prev_info["prev_direction"]
            prev_count = prev_info["total_count"]
            
            if prev_dir is None:
                examples = f"""PARENT: first move, count=0

Valid answers (pick {num_proposals}):
Next move: UP | Turn: STRAIGHT | Total-turn count: 0
Next move: DOWN | Turn: STRAIGHT | Total-turn count: 0
Next move: LEFT | Turn: STRAIGHT | Total-turn count: 0
Next move: RIGHT | Turn: STRAIGHT | Total-turn count: 0"""
            else:
                # Define turn mappings (same as Q0)
                turn_map = {
                    "UP": {"RIGHT": "RIGHT", "LEFT": "LEFT", "UP": "STRAIGHT", "DOWN": "STRAIGHT"},
                    "DOWN": {"LEFT": "RIGHT", "RIGHT": "LEFT", "DOWN": "STRAIGHT", "UP": "STRAIGHT"},
                    "LEFT": {"UP": "RIGHT", "DOWN": "LEFT", "LEFT": "STRAIGHT", "RIGHT": "STRAIGHT"},
                    "RIGHT": {"DOWN": "RIGHT", "UP": "LEFT", "RIGHT": "STRAIGHT", "LEFT": "STRAIGHT"}
                }
                
                moves = turn_map.get(prev_dir, {})
                examples_list = []
                for next_dir, turn_type in moves.items():
                    new_count = prev_count + 1 if turn_type in ["RIGHT", "LEFT"] else prev_count
                    examples_list.append(f"Next move: {next_dir} | Turn: {turn_type} | Total-turn count: {new_count}")
                
                examples = f"""PARENT: direction={prev_dir}, count={prev_count}

Valid answers (pick {num_proposals}):
{chr(10).join(examples_list)}"""
            
            return f"""{examples}

DO NOT explain. DO NOT reason. Just output {num_proposals} lines from above."""

        if question_type == "q4":
            # Q4 should also be structured - no long reasoning
            return f"""Maze spatial question. Generate {num_proposals} brief factual statements.

{current if current else "Starting."}

Output {num_proposals} lines. Each line: one short fact. NO long explanations."""

        # Generic fallback - also keep it structured
        return f"""Maze question. Generate {num_proposals} next steps.

{current if current else "Starting."}

Output {num_proposals} lines. Each line: one short step. NO explanations."""

    def _detect_spatialmap_question_type(self, problem: str) -> str:
        """Detect spatialmap subtype for proposal steering (direction/object/counting)."""
        lower = problem.lower()
        if "how many" in lower and ("objects" in lower or "places" in lower or "locations" in lower):
            return "counting"
        if "which object" in lower or "what object" in lower or "which place" in lower or "which location" in lower:
            return "object"
        if "in which direction" in lower or "what direction" in lower or "relative to" in lower:
            return "direction"
        return "generic"

    def _build_spatialmap_propose_prompt(
        self,
        problem: str,
        trajectory: str,
        num_proposals: int,
    ) -> str:
        """Build spatialmap-specific atomic next-step proposal prompts by question type."""
        question_type = self._detect_spatialmap_question_type(problem)
        logger.debug(f"Detected spatialmap question type: {question_type}")

        if not trajectory.strip():
            current = "Starting fresh - no progress yet"
            last_step_hint = ""
        else:
            current = trajectory.strip()
            lines = [line.strip() for line in current.split("\n") if line.strip()]
            last_line = lines[-1] if lines else ""
            last_step_hint = (
                f"\nLAST COMPLETED STEP: {last_line}\n"
                "Now generate the NEXT atomic step after this (do NOT repeat this step).\n"
            )

        if question_type == "direction":
            return f"""You are solving a spatial-map DIRECTION question.

PROBLEM:
{problem}

TRAJECTORY SO FAR:
{current}
{last_step_hint}
Your task: Propose {num_proposals} ATOMIC next steps only.
- Each step must advance exactly ONE concrete spatial inference
- Prefer one of: parse one relation, apply reversibility once, apply transitivity once, or map target-vs-reference direction
- Do NOT restate the whole map

Output format (one line per proposal, no preamble):
1. [Atomic spatial inference]
2. [Atomic spatial inference]
..."""

        if question_type == "object":
            return f"""You are solving a spatial-map OBJECT-IDENTIFICATION question.

PROBLEM:
{problem}

TRAJECTORY SO FAR:
{current}
{last_step_hint}
Your task: Propose {num_proposals} ATOMIC next steps only.
- Each step should do one action: identify candidate set, eliminate one candidate, or validate one relation against query direction
- Keep steps local and specific to the asked direction/object
- Do NOT rewrite all relationships

Output format (one line per proposal, no preamble):
1. [Atomic candidate/evidence step]
2. [Atomic candidate/evidence step]
..."""

        if question_type == "counting":
            return f"""You are solving a spatial-map COUNTING question.

PROBLEM:
{problem}

TRAJECTORY SO FAR:
{current}
{last_step_hint}
Your task: Propose {num_proposals} ATOMIC next steps only.
- Each step should do one action: identify one qualifying object, rule out one object, or update running count by exactly one justified change
- Keep a clear running count state
- Do NOT provide final answer yet unless count is fully justified

Output format (one line per proposal, no preamble):
1. [Atomic counting step, e.g., "Qualifies: <object>; running count = n"]
2. [Atomic counting step, e.g., "Ruled out: <object>; running count = n"]
..."""

        return f"""You are solving a spatial-map reasoning question.

PROBLEM:
{problem}

TRAJECTORY SO FAR:
{current}
{last_step_hint}
Propose {num_proposals} atomic, actionable next steps.
Each step must add ONE new spatial fact/inference and be different from prior steps.

Output format (one line per proposal, no preamble):
1. ...
2. ...
..."""
    
    def _parse_proposals(self, response: str, num_proposals: int) -> List[str]:
        """
        Parse proposals from model response.
        Handles various formats (numbered lists, bullets, etc.)
        """
        proposals: List[str] = []

        # First, try to extract exact format lines: "Next move: X | Turn: Y | ...count: Z"
        # Match line by line to avoid cross-line pollution
        for line in response.split('\n'):
            line = line.strip()
            # Check if it matches our exact format
            if 'Next move:' in line and 'Turn:' in line and 'count:' in line:
                proposals.append(line)
                if len(proposals) >= num_proposals:
                    return proposals[:num_proposals]

        # If we got enough exact format proposals, return them
        if len(proposals) >= num_proposals:
            return proposals[:num_proposals]

        # Fallback: numbered multiline blocks (preserve continuation lines)
        numbered_blocks = re.findall(
            r"(?:^|\n)\s*\d+[\.)]\s*(.+?)(?=(?:\n\s*\d+[\.)]\s*)|\Z)",
            response,
            flags=re.DOTALL,
        )
        for block in numbered_blocks:
            cleaned = " ".join(block.strip().split())
            if cleaned and len(cleaned) > 3:
                proposals.append(cleaned)
            if len(proposals) >= num_proposals:
                return proposals[:num_proposals]

        # Second pass: fallback line parser for bullets/single-line proposals
        for line in response.split("\n"):
            cleaned = line.strip()
            if not cleaned or cleaned in ["Next steps:", "Proposals:", "possible next steps"]:
                continue

            cleaned = re.sub(r"^\s*(?:\d+[\.)]|[-•*])\s*", "", cleaned)
            cleaned = re.sub(r"^\s*\[\d+\]\s*", "", cleaned)
            cleaned = re.sub(r"^\s*proposal\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = " ".join(cleaned.split())

            if cleaned and len(cleaned) > 3:
                proposals.append(cleaned)
            if len(proposals) >= num_proposals:
                return proposals[:num_proposals]

        return proposals[:num_proposals]
    
    # ===================== EVALUATE FUNCTION =====================
    
    async def evaluate_state(
        self,
        task: str,
        problem: str,
        trajectory: str,
        llm_server: Dict,
    ) -> float:
        """
        Evaluate the quality/progress of current state.
        
        Args:
            task: The task type (e.g., "game24", "maze", "spatialmap")
            problem: Original problem
            trajectory: Current solution trajectory
            llm_server: vLLM server config
            
        Returns:
            Value score between 0.0 and 1.0
        """
        # Check cache
        cache_key = f"evaluate_{hash(problem)}_{hash(trajectory)}"
        if self.config.cache_evaluations and cache_key in self.evaluation_cache:
            self.search_stats["cache_hits"] += 1
            return self.evaluation_cache[cache_key]
        
        self.search_stats["evaluations_performed"] += 1
        
        # Build evaluation prompt
        eval_prompt = self._build_evaluation_prompt(task, problem, trajectory)
        
        # Call model
        eval_response = await self._call_llm_streaming(llm_server, eval_prompt)
        
        # Parse evaluation into score
        score = self._parse_evaluation(eval_response)
        confidence_label = self._extract_confidence_label(eval_response, score)
        
        # Log evaluation
        eval_log = {
            "type": "state_evaluation",
            "timestamp": time.time(),
            "trajectory": trajectory,
            "prompt_preview": eval_prompt[:200] + "...",
            "response_preview": eval_response[:200] + "...",
            "score": score,
            "confidence": confidence_label,
        }
        if self.decision_tree:
            if "evaluations" not in self.decision_tree[-1]:
                self.decision_tree[-1]["evaluations"] = []
            self.decision_tree[-1]["evaluations"].append(eval_log)
        
        # Cache
        if self.config.cache_evaluations:
            self.evaluation_cache[cache_key] = score
        
        return score
    
    def _build_evaluation_prompt(self, task: str, problem: str, trajectory: str) -> str:
        """Build dataset-aware evaluation prompts reused by ToT scoring."""
        return build_tot_value_prompt(task, problem, trajectory)
    
    def _parse_evaluation(self, response: str) -> float:
        """
        Parse evaluation response into a scalar score [0, 1]
        """
        response_lower = response.lower()
        
        confidence_keywords = {
            "sure": 0.9, "certain": 0.9, "confident": 0.9,
            "likely": 0.7, "probably": 0.7,
            "possible": 0.5, "maybe": 0.5,
            "unlikely": 0.3, "doubtful": 0.3,
            "impossible": 0.1, "blocked": 0.1,
        }
        
        for keyword, score in confidence_keywords.items():
            if keyword in response_lower:
                return score
        
        # Try to extract numeric score if present (1-9 scale)
        for i, char in enumerate(response):
            if char.isdigit():
                digit = int(char)
                if 1 <= digit <= 9:
                    return digit / 9.0  # Normalize to [0, 1]
        
        return 0.5  # Default neutral score

    def _score_to_confidence(self, score: float) -> str:
        """Map scalar score [0,1] to confidence bucket."""
        if score >= 0.8:
            return "sure"
        if score >= 0.6:
            return "likely"
        if score >= 0.4:
            return "possible"
        if score >= 0.2:
            return "unlikely"
        return "impossible"

    def _extract_confidence_label(self, response: str, score: float) -> str:
        """Extract confidence label from value response; fallback to score mapping."""
        lower = response.lower()
        for label in ["sure", "likely", "possible", "unlikely", "impossible"]:
            if label in lower:
                return label
        return self._score_to_confidence(score)

    def _log_proposal_transition(
        self,
        depth: int,
        parent_trajectory: str,
        proposal: str,
        next_state: str,
        value: float,
        is_terminal: bool,
        pruned: bool,
    ) -> None:
        """Log proposal -> next-state -> value transition for debugging/analysis."""
        self.decision_tree.append(
            {
                "type": "proposal_transition",
                "timestamp": time.time(),
                "depth": depth,
                "parent_trajectory": parent_trajectory,
                "proposal": proposal,
                "next_state": next_state,
                "value": value,
                "value_confidence": self._score_to_confidence(value),
                "is_terminal": is_terminal,
                "pruned": pruned,
            }
        )

        logger.info(
            "ToT transition | depth=%d | proposal=%s | value=%.3f | confidence=%s | pruned=%s | terminal=%s",
            depth,
            proposal,
            value,
            self._score_to_confidence(value),
            pruned,
            is_terminal,
        )
    
    # ===================== SEARCH IMPLEMENTATION =====================
    
    async def search(
        self,
        task: str,
        problem: str,
        llm_server: Dict,
    ) -> Dict[str, Any]:
        """
        Perform Tree of Thought search on the problem.
        
        Args:
            task: The task type (e.g., "game24", "maze", "spatialmap")
            problem: Problem statement
            llm_server: vLLM server config
            
        Returns:
            Dictionary with best_trajectory, best_value, search_log
        """
        logger.info(f"Starting ToT search with method={self.config.search_method.value}")
        
        # Initialize root node
        self.root = TreeNode(trajectory="", depth=0, value=0.5)
        
        if self.config.search_method == SearchMethod.BFS:
            return await self._bfs_search(task, problem, llm_server)
        elif self.config.search_method == SearchMethod.BEAM:
            return await self._beam_search(task, problem, llm_server)
        else:
            return await self._dfs_search(task, problem, llm_server)
    
    async def _bfs_search(self, task: str, problem: str, llm_server: Dict) -> Dict[str, Any]:
        """Breadth-First Search implementation"""
        queue = [self.root]
        best_terminal = None
        best_value = 0.0
        best_candidate = None
        best_candidate_value = float('-inf')
        
        for depth in range(self.config.max_depth):
            if not queue:
                break
            
            next_queue = []
            
            for node in queue:
                # Generate proposals
                proposals = await self.propose_next_steps(
                    task,
                    problem,
                    node.trajectory,
                    llm_server,
                    self.config.branching_factor
                )
                node.proposals = proposals
                
                # Create child nodes
                for prop in proposals:
                    new_trajectory = f"{node.trajectory}\n{prop}" if node.trajectory else prop
                    child = TreeNode(
                        trajectory=new_trajectory,
                        depth=depth + 1,
                        parent=node,
                    )
                    
                    # Evaluate
                    value = await self.evaluate_state(task, problem, new_trajectory, llm_server)
                    child.value = value
                    
                    # Track best candidate regardless of terminal status
                    if value > best_candidate_value:
                        best_candidate = child
                        best_candidate_value = value

                    # Check if terminal and meets threshold
                    if self._is_terminal(new_trajectory):
                        child.is_terminal = True
                        self.search_stats["solutions_found"] += 1
                        if value > best_value:
                            best_value = value
                            best_terminal = child
                        is_terminal = True
                        
                        # Early termination if high confidence
                        if self.config.early_termination and value >= self.config.sure_threshold:
                            self._log_proposal_transition(
                                depth=depth + 1,
                                parent_trajectory=node.trajectory,
                                proposal=prop,
                                next_state=new_trajectory,
                                value=value,
                                is_terminal=is_terminal,
                                pruned=False,
                            )
                            return self._format_search_result(best_terminal, problem)
                    else:
                        is_terminal = False
                    
                    # Prune low-value nodes
                    if value < self.config.impossible_threshold:
                        self.search_stats["branches_pruned"] += 1
                        self._log_proposal_transition(
                            depth=depth + 1,
                            parent_trajectory=node.trajectory,
                            proposal=prop,
                            next_state=new_trajectory,
                            value=value,
                            is_terminal=is_terminal,
                            pruned=True,
                        )
                        continue

                    self._log_proposal_transition(
                        depth=depth + 1,
                        parent_trajectory=node.trajectory,
                        proposal=prop,
                        next_state=new_trajectory,
                        value=value,
                        is_terminal=is_terminal,
                        pruned=False,
                    )
                    
                    node.children.append(child)
                    next_queue.append(child)
                    self.search_stats["total_nodes_in_tree"] += 1
            
            queue = next_queue[:self.config.max_candidates_per_level]
        
        return self._format_search_result(best_terminal or best_candidate, problem)
    
    async def _beam_search(self, task: str, problem: str, llm_server: Dict) -> Dict[str, Any]:
        """Beam Search implementation"""
        beam = [self.root]
        best_terminal = None
        best_value = 0.0
        best_candidate = None
        best_candidate_value = float('-inf')
        
        for depth in range(self.config.max_depth):
            candidates = []
            
            for node in beam:
                # Generate and evaluate proposals
                proposals = await self.propose_next_steps(
                    task,
                    problem,
                    node.trajectory,
                    llm_server,
                    self.config.branching_factor
                )
                node.proposals = proposals

                for prop in proposals:
                    new_trajectory = f"{node.trajectory}\n{prop}" if node.trajectory else prop
                    value = await self.evaluate_state(task, problem, new_trajectory, llm_server)
                    
                    child = TreeNode(
                        trajectory=new_trajectory,
                        depth=depth + 1,
                        value=value,
                        parent=node,
                    )
                    
                    candidates.append((child, value))

                    if value > best_candidate_value:
                        best_candidate = child
                        best_candidate_value = value
                    self.search_stats["total_nodes_in_tree"] += 1
                    
                    if self._is_terminal(new_trajectory):
                        child.is_terminal = True
                        self.search_stats["solutions_found"] += 1
                        if value > best_value:
                            best_value = value
                            best_terminal = child
                        is_terminal = True
                        
                        if self.config.early_termination and value >= self.config.sure_threshold:
                            self._log_proposal_transition(
                                depth=depth + 1,
                                parent_trajectory=node.trajectory,
                                proposal=prop,
                                next_state=new_trajectory,
                                value=value,
                                is_terminal=is_terminal,
                                pruned=False,
                            )
                            return self._format_search_result(best_terminal, problem)
                    else:
                        is_terminal = False

                    pruned = value < self.config.impossible_threshold
                    if pruned:
                        self.search_stats["branches_pruned"] += 1

                    self._log_proposal_transition(
                        depth=depth + 1,
                        parent_trajectory=node.trajectory,
                        proposal=prop,
                        next_state=new_trajectory,
                        value=value,
                        is_terminal=is_terminal,
                        pruned=pruned,
                    )

                    if pruned:
                        continue
            
            # Keep top-k by value
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = [child for child, _ in candidates[:self.config.beam_width]]
            
            if not beam:
                break
        
        return self._format_search_result(best_terminal or best_candidate, problem)
    
    async def _dfs_search(self, task: str, problem: str, llm_server: Dict) -> Dict[str, Any]:
        """Depth-First Search implementation"""
        best_terminal = None
        best_value = 0.0
        best_candidate = None
        best_candidate_value = float('-inf')
        
        async def dfs(node: TreeNode, depth: int):
            nonlocal best_terminal, best_value
            
            if depth >= self.config.max_depth:
                return
            
            # Generate proposals
            proposals = await self.propose_next_steps(
                task,
                problem,
                node.trajectory,
                llm_server,
                self.config.branching_factor
            )
            node.proposals = proposals
            
            for prop in proposals:
                new_trajectory = f"{node.trajectory}\n{prop}" if node.trajectory else prop
                value = await self.evaluate_state(task, problem, new_trajectory, llm_server)
                
                child = TreeNode(
                    trajectory=new_trajectory,
                    depth=depth + 1,
                    value=value,
                    parent=node,
                )
                node.children.append(child)

                if value > best_candidate_value:
                    best_candidate = child
                    best_candidate_value = value
                self.search_stats["total_nodes_in_tree"] += 1
                
                if self._is_terminal(new_trajectory):
                    child.is_terminal = True
                    self.search_stats["solutions_found"] += 1
                    if value > best_value:
                        best_value = value
                        best_terminal = child
                    is_terminal = True
                    
                    if self.config.early_termination and value >= self.config.sure_threshold:
                        self._log_proposal_transition(
                            depth=depth + 1,
                            parent_trajectory=node.trajectory,
                            proposal=prop,
                            next_state=new_trajectory,
                            value=value,
                            is_terminal=is_terminal,
                            pruned=False,
                        )
                        return
                else:
                    is_terminal = False
                
                # Prune
                if value >= self.config.impossible_threshold:
                    self._log_proposal_transition(
                        depth=depth + 1,
                        parent_trajectory=node.trajectory,
                        proposal=prop,
                        next_state=new_trajectory,
                        value=value,
                        is_terminal=is_terminal,
                        pruned=False,
                    )
                    await dfs(child, depth + 1)
                else:
                    self.search_stats["branches_pruned"] += 1
                    self._log_proposal_transition(
                        depth=depth + 1,
                        parent_trajectory=node.trajectory,
                        proposal=prop,
                        next_state=new_trajectory,
                        value=value,
                        is_terminal=is_terminal,
                        pruned=True,
                    )
        
        await dfs(self.root, 0)
        return self._format_search_result(best_terminal or best_candidate, problem)
    
    # ===================== UTILITIES =====================
    
    def _is_terminal(self, trajectory: str) -> bool:
        """Check if trajectory represents a complete solution"""
        keywords = [
            "final answer",
            "reached goal",
            "solution:",
            "answer:",
            "conclusion:",
            "result:",
        ]
        trajectory_lower = trajectory.lower()
        return any(kw in trajectory_lower for kw in keywords)
    
    def _format_search_result(
        self,
        best_node: Optional[TreeNode],
        problem: str
    ) -> Dict[str, Any]:
        """Format search results for return"""
        if best_node:
            best_trajectory = best_node.trajectory
            best_value = best_node.value
        else:
            best_trajectory = ""
            best_value = 0.0
        
        return {
            "best_trajectory": best_trajectory,
            "best_value": best_value,
            "search_stats": self.search_stats,
            "decision_tree": self.decision_tree,
            "root_node": self.root,
        }
    
    async def _call_llm_streaming(
        self,
        llm_server: Dict,
        prompt: str
    ) -> str:
        """Call chat-completions endpoint and return the full response text."""
        import httpx

        payload = llm_server["payload"].copy()
        payload.pop("prompt", None)
        payload.pop("messages", None)
        payload["messages"] = [{"role": "user", "content": prompt}]
        payload["stream"] = False

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    llm_server["url"],
                    headers=llm_server["headers"],
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            raise

        choices = data.get("choices", [])
        if not choices:
            logger.warning("LLM response missing choices: %s", data.keys())
            return ""

        choice = choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message") or {}
            return msg.get("content") or choice.get("text", "")
        return str(choice)
    
    def get_decision_tree_json(self) -> str:
        """Export decision tree as JSON"""
        return json.dumps({
            "search_stats": self.search_stats,
            "decision_points": self.decision_tree,
            "num_decision_points": len(self.decision_tree),
        }, indent=2, default=str)

    def _serialize_node(self, node: Optional[TreeNode], max_depth: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Serialize tree nodes recursively for debugging/state inspection."""
        if node is None:
            return None

        if max_depth is not None and node.depth >= max_depth:
            return {
                "depth": node.depth,
                "value": node.value,
                "is_terminal": node.is_terminal,
                "trajectory": node.trajectory,
                "num_children": len(node.children),
                "children": [],
                "truncated": True,
            }

        return {
            "depth": node.depth,
            "value": node.value,
            "is_terminal": node.is_terminal,
            "trajectory": node.trajectory,
            "num_children": len(node.children),
            "children": [self._serialize_node(child, max_depth=max_depth) for child in node.children],
        }

    def get_state_snapshot(
        self,
        include_tree: bool = True,
        max_tree_depth: Optional[int] = None,
        decision_tail: Optional[int] = 50,
        include_cache_samples: bool = True,
        cache_sample_size: int = 5,
    ) -> Dict[str, Any]:
        """Return a comprehensive snapshot of the current ToT search state."""
        proposal_cache_keys = list(self.proposal_cache.keys())
        evaluation_cache_keys = list(self.evaluation_cache.keys())

        snapshot: Dict[str, Any] = {
            "config": {
                "branching_factor": self.config.branching_factor,
                "max_depth": self.config.max_depth,
                "search_method": self.config.search_method.value,
                "beam_width": self.config.beam_width,
                "sure_threshold": self.config.sure_threshold,
                "likely_threshold": self.config.likely_threshold,
                "impossible_threshold": self.config.impossible_threshold,
                "early_termination": self.config.early_termination,
                "cache_evaluations": self.config.cache_evaluations,
                "max_candidates_per_level": self.config.max_candidates_per_level,
            },
            "search_stats": dict(self.search_stats),
            "decision_tree_size": len(self.decision_tree),
            "cache_state": {
                "proposal_cache_size": len(self.proposal_cache),
                "evaluation_cache_size": len(self.evaluation_cache),
            },
            "root_present": self.root is not None,
            "root_depth": self.root.depth if self.root is not None else None,
            "root_value": self.root.value if self.root is not None else None,
            "root_is_terminal": self.root.is_terminal if self.root is not None else None,
        }

        if decision_tail is None:
            snapshot["decision_tree"] = self.decision_tree
        else:
            snapshot["decision_tree_tail"] = self.decision_tree[-decision_tail:]

        if include_cache_samples:
            sample_size = max(0, cache_sample_size)
            snapshot["cache_state"]["proposal_cache_key_samples"] = proposal_cache_keys[:sample_size]
            snapshot["cache_state"]["evaluation_cache_key_samples"] = evaluation_cache_keys[:sample_size]

        if include_tree:
            snapshot["tree"] = self._serialize_node(self.root, max_depth=max_tree_depth)

        return snapshot

    def get_state_snapshot_json(
        self,
        include_tree: bool = True,
        max_tree_depth: Optional[int] = None,
        decision_tail: Optional[int] = 50,
        include_cache_samples: bool = True,
        cache_sample_size: int = 5,
        indent: int = 2,
    ) -> str:
        """Return JSON string for the current ToT state snapshot."""
        snapshot = self.get_state_snapshot(
            include_tree=include_tree,
            max_tree_depth=max_tree_depth,
            decision_tail=decision_tail,
            include_cache_samples=include_cache_samples,
            cache_sample_size=cache_sample_size,
        )
        return json.dumps(snapshot, indent=indent, default=str)