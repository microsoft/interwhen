"""
Value prompts with few-shot examples for Tree of Thought evaluation across datasets.
"""

# ============================================================================
# GAME24 VALUE PROMPTS WITH FEW-SHOT EXAMPLES
# ============================================================================

GAME24_VALUE_PROMPT_WITH_FEWSHOT = """
Evaluate if given numbers can reach 24 (sure/likely/impossible).

PROBLEM STATEMENT:
{problem}

CURRENT TRAJECTORY:
{trajectory}

Here are examples of how to evaluate Game of 24 trajectories:

EXAMPLE 1 - SURE:
Numbers: 4 4 6 8
Trajectory: 4 + 8 = 12 (left: 4 6 12), 6 - 4 = 2 (left: 2 12), 2 * 12 = 24
Analysis: Reaches exactly 24 using each number exactly once.
Confidence: sure (9)

EXAMPLE 2 - SURE:
Numbers: 2 9 10 12
Trajectory: 12 * 2 = 24 (left: 9 10 24), then 24 * (10 - 9) = 24 * 1 = 24
Analysis: Valid path found with each number used once, equals 24.
Confidence: sure (9)

EXAMPLE 3 - SURE:
Numbers: 10 14
Trajectory: 10 + 14 = 24
Analysis: Direct calculation using both numbers reaches exactly 24.
Confidence: sure (9)

EXAMPLE 4 - SURE:
Numbers: 4 4 10
Trajectory: (10 - 4) * 4 = 6 * 4 = 24
Analysis: Uses all three numbers exactly once with factorization that reaches 24.
Confidence: sure (9)

EXAMPLE 5 - SURE:
Numbers: 4 9 11
Trajectory: 9 + 11 + 4 = 24
Analysis: All numbers used, arithmetic valid, equals 24.
Confidence: sure (9)

EXAMPLE 6 - LIKELY:
Numbers: 5 7 8
Trajectory: 5 + 7 + 8 = 20, or (8 - 5) * 7 = 21
Analysis: Cannot reach 24 immediately, but numbers in reasonable arithmetic range where 24 might be
achievable.
Confidence: likely (7)

EXAMPLE 7 - LIKELY:
Numbers: 5 6 6
Trajectory: 5 + 6 + 6 = 17, or (6 - 5) * 6 = 6
Analysis: Current attempts don't reach 24, but numbers are within reasonable range.
Confidence: likely (7)

EXAMPLE 8 - IMPOSSIBLE:
Numbers: 1 3 3
Trajectory: 1 * 3 * 3 = 9, or (1 + 3) * 3 = 12
Analysis: Maximum reachable with any operations is much less than 24. Numbers all too small.
Confidence: impossible (1)

EXAMPLE 9 - IMPOSSIBLE:
Numbers: 10 10 11
Trajectory: 10 + 10 + 11 = 31, or (11 - 10) * 10 = 10
Analysis: Sum exceeds 24, factorizations fall short. Cannot reach exactly 24.
Confidence: impossible (1)

EXAMPLE 10 - IMPOSSIBLE:
Numbers: 11 12
Trajectory: 11 + 12 = 23, or 12 - 11 = 1, or 11 * 12 = 132, or 11 / 12 ≈ 0.91
Analysis: No operation reaches 24. Sum close but not exact.
Confidence: impossible (1)

Rubric:
- sure (9): Reaches 24 using each number exactly once
- likely (7): Cannot reach 24 yet, but numbers in reasonable range
- possible (5): Uncertain if 24 is reachable
- unlikely (3): Numbers seem misaligned
- impossible (1): Numbers demonstrably cannot reach 24

Respond with "Confidence: <level>" followed by brief justification tied to arithmetic evaluations.
"""

GAME24_VALUE_PROMPT_SIMPLE = """Evaluate if given numbers can reach 24.

PROBLEM STATEMENT:
{problem}

CURRENT TRAJECTORY:
{trajectory}

Rate confidence on this rubric:
- sure (9): Reaches 24 using each number exactly once
- likely (7): Cannot reach 24 yet, but numbers in reasonable range
- possible (5): Uncertain if 24 is reachable
- unlikely (3): Numbers seem misaligned 
- impossible (1): Numbers cannot reach 24

Respond with "Confidence: <level>" and brief justification.
"""




# ============================================================================
# MAZE VALUE PROMPTS WITH FEW-SHOT EXAMPLES
# ============================================================================



MAZE_VALUE_PROMPT_WITH_FEWSHOT = """Verify a maze reasoning trace.

TASK PROMPT:
{problem}

MODEL TRAJECTORY:
{trajectory}

EXAMPLE 1 - SURE:
Question: Count right turns in path X from S to E
Trajectory: Carefully trace X-marked path. Starting at S, move UP (initial direction). 
Then RIGHT (90 degrees clockwise = right turn 1). Then DOWN (90 degrees clockwise = right turn 2).
Then RIGHT (90 degrees clockwise = right turn 3). Continuing pattern: 6 right turns total.
Answer: B (6 right turns)
Analysis: Systematic path tracing with correct turn geometry, defensible count.
Confidence: sure (9)

EXAMPLE 2 - SURE:
Question: What is the sequence of grid direction?
Trajectory: Following marked path from S: [0,0] then [0,1] (UP) then [1,1] (RIGHT) then [1,0]
(DOWN) then [2,0] (RIGHT). Each step verified against grid.
Answer: UP, RIGHT, DOWN, RIGHT
Analysis: Clear coordinate tracking, systematic verification.
Confidence: sure (9)

EXAMPLE 3 - LIKELY:
Question: Count right turns in path from S to E
Trajectory: Observing marked path shows mostly straight movements with a zigzag pattern. Zigzags
suggest mostly left turns. Likely 0-2 right turns based on pattern.
Answer: A (0 right turns)
Confidence: likely (7)
Analysis: Shows reasonable spatial intuition but lacks systematic verification.

EXAMPLE 4 - LIKELY:
Question: Is path continuous S to E?
Trajectory: I trace the marked path and it appears to connect from S all the way to E without
breaks. The X marks form a continuous line.
Answer: Yes
Confidence: likely (7)
Analysis: Reasonable assessment but could benefit from detailed step verification.

EXAMPLE 5 - POSSIBLE:
Question: Navigate maze from S to E
Trajectory: Following path X... I see turns but the specific sequence is unclear to me. Could be 3
or 4 right turns.
Answer: Uncertain between options
Confidence: possible (5)
Analysis: Recognizes task but cannot decisively trace path geometry.

EXAMPLE 6 - UNLIKELY:
Question: Count right turns in path X
Trajectory: Tracing marked path X from S. Moving DOWN, then RIGHT (left turn?), then DOWN, then
RIGHT (right turn?). I'm confused about turn geometry.
Answer: Some right turns but not sure
Confidence: unlikely (3)
Analysis: Confused about path following and angle identification.

EXAMPLE 7 - IMPOSSIBLE:
Question: Navigate from S to E following marked path
Trajectory: I move RIGHT, then up, then left, I'm not sure where the path goes. I think I hit a
wall.
Answer: I'm stuck
Confidence: impossible (1)
Analysis: Abandons task without following clearly marked X path provided.

Rubric: sure (9), likely (7), possible (5), unlikely (3), impossible (1)
Respond with "Confidence: <level>" + explanation referencing moves/directions.
"""

MAZE_VALUE_PROMPT_SIMPLE = """Verify a maze reasoning trace.

TASK PROMPT:
{problem}

MODEL TRAJECTORY:
{trajectory}

Judge if reasoning is consistent with maze/spatial relationships and if final answer is defensible.
Rubric: sure (9), likely (7), possible (5), unlikely (3), impossible (1)
Respond with "Confidence: <level>" + explanation referencing moves/directions.
"""




# ============================================================================
# SPATIAL REASONING VALUE PROMPTS WITH FEW-SHOT EXAMPLES
# ============================================================================




SPATIALMAP_VALUE_PROMPT_WITH_FEWSHOT = """
You are verifying a spatial reasoning multiple-choice trace.

TASK PROMPT:
{problem}

MODEL TRAJECTORY:
{trajectory}

Here are examples of how to evaluate spatial reasoning:

EXAMPLE 1:
Question: Based on the map, which location is northeast of the library?
Trajectory: I look at the map and see the library in the center. To the northeast means both north
AND east of that point. Looking at that quadrant, I see the museum is northeast of the library.
Answer: A (Museum)

Analysis: The student correctly understands the spatial direction (northeast = north AND east),
correctly identifies it on the map, and selects the correct option.
Confidence: sure/certain (9)
Justification: The reasoning correctly applies spatial relationships and identifies the appropriate
location.

EXAMPLE 2:
Question: Which building is closest to the park?
Trajectory: The park looks like it's in the middle of the map. Near it I see a building... looks
like it could be the school or maybe the library. I think the school is closer.
Answer: C (School)

Analysis: The student makes reasonable spatial observations but doesn't verify distances or compare
alternatives systematically.
Confidence: likely/probably (7)
Justification: The reasoning shows spatial awareness but lacks systematic comparison of distances
to verify the answer.

EXAMPLE 3:
Question: What is north of the train station?
Trajectory: I see the train station. North is... up on the map. I see some buildings, but I'm not
sure exactly which one. Could be the post office or the police station.
Answer: I'm not sure

Analysis: The student recognizes the direction but fails to identify the specific location clearly.
Confidence: possible/maybe (5)
Justification: The student understands the spatial direction but cannot decisively identify which
building is in that location.

EXAMPLE 4:
Question: If you're at the bank facing east, what's behind you?
Trajectory: At the bank facing east means I'm looking east. Behind me would be... west? I need to
think about what's west of the bank. I think there's a hotel or a store but I'm not sure.
Answer: Maybe the hotel

Analysis: The student correctly understands relative directions (east/behind = west) but isn't
certain about the specific feature.
Confidence: unlikely/doubtful (3)
Justification: While the directional reasoning is sound, the uncertainty about the specific
location makes this answer questionable.

Use the confidence rubric: sure/certain (9), likely/probably (7), possible/maybe (5),
unlikely/doubtful (3), impossible/blocked (1).
Respond with "Confidence: <category>" plus a concise explanation that references spatial
relationships and map features.
"""

SPATIALMAP_VALUE_PROMPT_SIMPLE = """You are verifying a spatial reasoning multiple-choice trace.

TASK PROMPT:
{problem}

MODEL TRAJECTORY:
{trajectory}

Judge if the reasoning correctly applies spatial relationships (north, south, east, west, near,
far, etc.) and whether the final \\boxed{{choice}} is defensible.
Use the confidence rubric: sure/certain (9), likely/probably (7), possible/maybe (5),
unlikely/doubtful (3), impossible/blocked (1).
Respond with "Confidence: <category>" plus a concise explanation that references the spatial
relationships and locations.
"""



# ============================================================================
# ZEBRA LOGIC VALUE PROMPTS WITH FEW-SHOT EXAMPLES
# ============================================================================




ZEBRALOGIC_VALUE_PROMPT_WITH_FEWSHOT = """Evaluate a Zebra Logic puzzle solution trajectory.

TASK PROMPT:
{problem}

MODEL TRAJECTORY:
{trajectory}

Here are examples of how to evaluate Zebra Logic trajectories:

EXAMPLE 1 - SURE:
Puzzle: Houses with colors, pets, beverages, and nationalities with clues about relationships.
Trajectory: I've systematically worked through the constraints. House 1 has British resident. Red
house owner has Panda. Coffee drinker speaks Japanese. Working through elimination, I've determined
all houses uniquely and the solution satisfies all clues without contradictions.
Analysis: Systematic constraint satisfaction with clear justification for each assignment. Solution
verifiable.
Confidence: sure (9)

EXAMPLE 2 - LIKELY:
Trajectory: Working through the clues methodically. I've identified several definite assignments
(House 2 has Swedish resident with bird). For the remaining houses, the constraints are narrowing
down possibilities and should lead to a unique solution.
Analysis: Reasonable progress using logic, but not yet complete verification of all constraints.
Confidence: likely (7)

EXAMPLE 3 - POSSIBLE:
Trajectory: I understand the puzzle structure. I'm working through clues but some deductions are
unclear to me. I think House 1 might have the British resident, but I'm not certain.
Analysis: Shows problem understanding but lacks decisive constraint application.
Confidence: possible (5)

EXAMPLE 4 - UNLIKELY:
Trajectory: I'm trying to assign attributes to houses. House 1 has red color and Swedish resident.
House 2 has green... wait, but green is next to red. I'm getting confused by the adjacency
constraints.
Analysis: Fundamental misunderstanding of spatial/logical constraints.
Confidence: unlikely (3)

EXAMPLE 5 - IMPOSSIBLE:
Trajectory: I'm going to assign all attributes randomly since I don't see how the clues relate to
each other.
Analysis: Abandons logical reasoning without attempting systematic constraint satisfaction.
Confidence: impossible (1)

Rubric for Zebra Logic:
- sure (9): Complete solution derived with clear constraint verification, all assignments justified
- likely (7): Systematic progress with mostly confident deductions, minor uncertainties remain
- possible (5): Some correct deductions but missing clear constraint application
- unlikely (3): Attempting logic but making errors in constraint application or showing confusion
- impossible (1): No meaningful attempt at systematic constraint satisfaction

Respond with "Confidence: <level>" followed by brief justification referencing the logical
deductions and constraint satisfaction.
"""

ZEBRALOGIC_VALUE_PROMPT_SIMPLE = """Evaluate a Zebra Logic puzzle solution trajectory.

TASK PROMPT:
{problem}

MODEL TRAJECTORY:
{trajectory}

Judge if the reasoning systematically applies logical constraints and whether the solution
assignments are well-justified.
Use the confidence rubric:
- sure (9): Complete solution with clear constraint verification
- likely (7): Systematic progress with mostly confident deductions
- possible (5): Some correct deductions with minor gaps
- unlikely (3): Attempting logic but making constraint errors
- impossible (1): No meaningful systematic reasoning

Respond with "Confidence: <level>" and brief justification referencing constraint satisfaction.
"""



# ============================================================================
# VERINA (LEAN 4 CODE GENERATION) VALUE PROMPTS WITH FEW-SHOT EXAMPLES
# ============================================================================





VERINA_VALUE_PROMPT_WITH_FEWSHOT = """Evaluate a Lean 4 code-generation reasoning trajectory.

PROBLEM:
{problem}

TRAJECTORY:
{trajectory}

You are evaluating whether this reasoning trajectory is making SOUND PROGRESS toward correct Lean 4
code.
The trajectory may or may not contain final code - that's fine. Judge the REASONING QUALITY.

## WHAT MAKES REASONING GOOD vs BAD

GOOD reasoning:
- Each step adds NEW concrete information (function name, operator, edge case, type)
- Mentions CORRECT Lean 4 constructs (foldl, match, recursion, List/Array methods)
- Identifies the right functional idioms (pure, immutable, no mutation)
- Builds toward a clear implementation path

BAD reasoning:
- REPETITION: Multiple steps saying the same thing ("use XOR", "XOR will work", "apply XOR")
- WRONG IDIOMS: Mentions imperative patterns (mutable variables, imperative iteration)
- WRONG OPERATORS: Plans to use `^` for XOR (it's exponentiation!), Python syntax
- VAGUE: "Process the list" without specifying HOW (fold? recursion? map?)
- STUCK: Goes in circles without making progress

## SCORING RUBRIC

Ask: "Is this reasoning on track to produce CORRECT, COMPILABLE Lean 4 code?"

- **sure (9)**: Clear path to correct code. Either has code, OR reasoning identifies correct Lean 4
constructs with enough detail to write code immediately. No wrong idioms.
- **likely (7)**: Right direction with specific Lean 4 constructs mentioned. Minor gaps but
approach is sound.
- **possible (5)**: Correct algorithm but vague on Lean 4 specifics. OR has repetition (multiple
steps, one idea).
- **unlikely (3)**: Reasoning uses WRONG patterns that won't work in pure Lean 4 function bodies
(mutable state, `^` for XOR, imperative style).
- **impossible (1)**: Wrong language, nonsense, or completely off-track.

## FEW-SHOT EXAMPLES

Problem: Find single number in list where all others appear twice.

EXAMPLE 1 — SURE (9):
Trajectory: "XOR all elements. XOR cancels duplicates (a⊕a=0). In Lean 4, use List.foldl with
Nat.xor and initial value 0."
Why SURE: Correct algorithm + correct Lean 4 construct (foldl, Nat.xor). Could write code now.

EXAMPLE 2 — SURE (9):
Trajectory: "Use foldl to accumulate XOR. Start with 0, apply Nat.xor to each element. The property
a⊕a=0 ensures only the unique element survives."
Why SURE: Same information, phrased differently but complete. Ready to implement.

EXAMPLE 3 — LIKELY (7):
Trajectory: "XOR all elements using some kind of fold operation. Lean has foldl for lists. Need to
find the right XOR function."
Why LIKELY: Knows the approach and foldl, but hasn't pinpointed Nat.xor yet. One detail away.

EXAMPLE 4 — POSSIBLE (5):
Trajectory: "XOR all numbers together. The duplicates will cancel out leaving the unique one."
Why POSSIBLE: Correct algorithm but NO Lean 4 specifics. How to XOR? What function?

EXAMPLE 5 — POSSIBLE (5) — REPETITION:
Trajectory: "XOR cancels duplicates. So XORing all elements gives the answer. The XOR operation is
the key insight here."
Why POSSIBLE: Three sentences, ONE idea repeated. No progress on HOW to implement in Lean 4.

EXAMPLE 6 — UNLIKELY (3):
Trajectory: "Loop through the list and XOR each element with an accumulator using ^. Something
like: for x in nums, result = result ^ x"
Why UNLIKELY: `^` is exponentiation in Lean 4, not XOR. Mutable accumulator pattern won't work in a
pure function body. This reasoning leads to compile errors.

EXAMPLE 7 — UNLIKELY (3):
Trajectory: "Create a mutable result variable initialized to 0. Iterate through nums, updating
result ^= x for each element."
Why UNLIKELY: Mutable variables and imperative iteration don't exist in pure Lean 4. Wrong paradigm
entirely.

---

Problem: Return maximum of two natural numbers.

EXAMPLE 8 — SURE (9):
Trajectory: "Use if-then-else: `if a >= b then a else b`. Nat has decidable ordering so comparison
works directly."
Why SURE: Exact syntax given, identifies why it works (decidable). Ready to implement.

EXAMPLE 9 — LIKELY (7):
Trajectory: "Compare a and b with if-then-else. Return the larger one. Lean's Nat supports direct
comparison."
Why LIKELY: Right approach, right construct, just needs exact syntax.

EXAMPLE 10 — POSSIBLE (5):
Trajectory: "Return whichever number is bigger."
Why POSSIBLE: Correct goal, but how? No Lean 4 specifics at all.

---

Problem: Compute longest common subsequence length.

EXAMPLE 11 — UNLIKELY (3):
Trajectory: "Build a 2D DP table. Use nested for loops: for i in 1..m, for j in 1..n, set dp[i][j]
based on matches."
Why UNLIKELY: Nested loops with mutable array updates don't work in pure Lean 4 function bodies.
Need functional approach (folds, recursion).

EXAMPLE 12 — LIKELY (7):
Trajectory: "Use dynamic programming with foldl. Outer fold over rows, inner fold over columns.
Build each row as a new List, tracking previous row values."
Why LIKELY: Correct functional approach (foldl, immutable row building). Needs details but sound
path.

---

## KEY PENALTIES

1. **REPETITION**: If 3 steps contain only 1 unique idea → score based on that 1 idea only
2. **WRONG OPERATORS**: `^` for XOR, Python's `max()`, etc. → unlikely (3) at best
3. **IMPERATIVE PATTERNS**: mutable state, imperative iteration in pure function bodies → unlikely
(3) at best
4. **VAGUE ABSTRACTION**: "process", "iterate", "handle" without Lean specifics → possible (5) max

## YOUR EVALUATION

Respond with "Confidence: <level>" followed by:
1. What NEW concrete information does each step add? (or note repetition)
2. Are the Lean 4 constructs mentioned correct?
3. Does the reasoning lead toward compilable code or toward errors?
"""

VERINA_VALUE_PROMPT_SIMPLE = """Evaluate a Lean 4 code-generation reasoning trajectory.

PROBLEM:
{problem}

TRAJECTORY:
{trajectory}

The trajectory contains step-by-step reasoning toward a Lean 4 function body.
Judge if the reasoning is on a sound path to producing valid Lean 4 that satisfies the
postcondition given the precondition.

Key scoring factors:
- Could you write the final code RIGHT NOW from this trajectory? If yes → sure (9).
- Steps that rephrase or "verify" earlier steps without adding new specifics (syntax, types, edge
cases) count as repetition. Repetitive trajectories should NEVER score sure.
- A single high-level idea without Lean 4 specifics should score possible (5) at most.
- Concrete Lean 4 syntax, function names, and postcondition reasoning push toward likely (7) or
sure (9).

Use the confidence rubric:
- sure (9): CODE-READY — correct Lean 4 syntax/functions AND postcondition logic, no gaps
- likely (7): Correct approach with specific Lean 4 constructs, minor gaps before code
- possible (5): Right direction but missing Lean 4 details, OR multi-step but repetitive
- unlikely (3): Fundamental issues in approach or idioms
- impossible (1): Completely off-track or wrong language

Respond with "Confidence: <level>" and brief justification. Note if steps are repetitive.
"""



# ============================================================================
# VERINA SPEC (LEAN 4 SPECIFICATION GENERATION) VALUE PROMPTS
# ============================================================================



VERINA_SPEC_VALUE_PROMPT_WITH_FEWSHOT = """
Evaluate a Lean 4 specification-generation reasoning trajectory.

PROBLEM:
{problem}

TRAJECTORY:
{trajectory}

You are evaluating whether this reasoning trajectory is making SOUND PROGRESS toward correct Lean 4
specifications (preconditions and postconditions as Prop expressions).

## WHAT MAKES SPEC REASONING GOOD vs BAD

GOOD reasoning:
- Each step adds NEW concrete information (a constraint, a quantifier, a Lean type)
- Uses correct Lean 4 Prop syntax: `∧`, `∨`, `→`, `¬`, `∀`, `∃`, `True`, `False`
- Mentions real Lean 4 functions: `List.count`, `List.length`, `Array.size`, etc.
- Reasons about SOUNDNESS (rejects bad inputs/outputs) and COMPLETENESS (accepts all valid ones)
- Produces actual Prop expressions, not English descriptions

BAD reasoning:
- NATURAL LANGUAGE: "The result must be..." instead of actual Lean syntax
- NON-EXISTENT FUNCTIONS: `lcsLength`, `maxSubarray`, made-up methods
- REPETITION: "The precondition should check X. Verify X is correct. Confirm X works."
- PYTHON SYNTAX: `result == max(a, b)`, `len(nums)`, `nums.count(x)`
- VAGUE: "The postcondition should ensure correctness" without specifics

## CRITICAL DISTINCTION: Props vs English

WRONG (English descriptions):
- "The input list must be non-empty"
- "The result should be the maximum value"
- "All elements must appear exactly twice"

RIGHT (Lean Props):
- `nums.length > 0`
- `result >= a ∧ result >= b ∧ (result = a ∨ result = b)`
- `∀ x, x ∈ nums → (List.count x nums = 1 ∨ List.count x nums = 2)`

## SCORING RUBRIC

Ask: "Could I write compilable [PRECOND]...[/PRECOND] and [POSTCOND]...[/POSTCOND] from this
reasoning?"

- **sure (9)**: Has (or is ready to produce) valid Lean Prop syntax. Mentions correct operators (∧,
∨, ∀, ∃). Uses real Lean functions.
- **likely (7)**: Right direction with specific Lean constructs identified. Needs exact syntax but
approach is sound.
- **possible (5)**: Correct intuition about what to specify, but vague on Lean syntax. OR
repetitive reasoning.
- **unlikely (3)**: Uses English descriptions, made-up functions, or Python syntax. Won't compile.
- **impossible (1)**: Wrong language, nonsense, or completely off-track.

## FEW-SHOT EXAMPLES

Problem: Find single number in list where all others appear twice.

EXAMPLE 1 — SURE (9):
Trajectory: "Precondition: exactly one element has count 1, others have count 2. Use ∃ x, x ∈ nums
∧ List.count x nums = 1 ∧ (∀ y, y ∈ nums → y ≠ x → List.count y nums = 2). Postcondition:
List.count result nums = 1."
Why SURE: Valid Lean syntax with ∃, ∀, ∧, →. Uses real function List.count. Membership guard (∈
nums) ensures quantifiers are well-scoped. Ready to write.

EXAMPLE 2 — LIKELY (7):
Trajectory: "Need to say exactly one element appears once. Use existential quantifier and
List.count. The result should be that unique element."
Why LIKELY: Knows the constructs (∃, List.count) but hasn't assembled exact syntax yet.

EXAMPLE 3 — POSSIBLE (5):
Trajectory: "The precondition should check that the list has a unique element. The postcondition
should verify the result is that element."
Why POSSIBLE: Right idea but NO Lean syntax. Just English descriptions.

EXAMPLE 4 — UNLIKELY (3):
Trajectory: "The input list must be non-empty, and there must be exactly one element that appears
exactly once in the list."
Why UNLIKELY: Pure English prose. "The input list must be..." won't compile. Needs Prop syntax.

EXAMPLE 5 — UNLIKELY (3):
Trajectory: "Postcondition: result = nums.findUnique() where findUnique returns the single
element."
Why UNLIKELY: `findUnique()` doesn't exist. Made-up method.

---

Problem: Return maximum of two natural numbers.

EXAMPLE 6 — SURE (9):
Trajectory: "Precondition: True (no restrictions on Nat). Postcondition: result >= a ∧ result >= b
∧ (result = a ∨ result = b). This is sound: rejects values not equal to a or b. Complete: actual
max satisfies all conjuncts."
Why SURE: Complete Prop with soundness/completeness reasoning. Exact syntax ready.

EXAMPLE 7 — LIKELY (7):
Trajectory: "Postcondition needs three parts: result >= a, result >= b, and result equals one of
them. Use conjunction ∧ and disjunction ∨."
Why LIKELY: Knows the pieces and operators, just needs to assemble.

EXAMPLE 8 — POSSIBLE (5):
Trajectory: "The postcondition should ensure the result is the maximum."
Why POSSIBLE: Right goal, but "the maximum" is English, not Lean. What IS maximum in Prop terms?

EXAMPLE 9 — UNLIKELY (3):
Trajectory: "Postcondition: result == max(a, b)"
Why UNLIKELY: `==` is boolean equality (BEq), not propositional equality — use `=` for Props.
`max(a, b)` uses parenthesized call syntax — use `max a b` or `Nat.max a b` in Lean.

---

Problem: Longest common subsequence length.

EXAMPLE 10 — UNLIKELY (3):
Trajectory: "Postcondition: result = lcsLength a b where lcsLength computes the LCS."
Why UNLIKELY: `lcsLength` doesn't exist in Lean's standard library. Made-up function.

EXAMPLE 11 — LIKELY (7):
Trajectory: "Need to define what LCS means. In POSTCOND_AUX, define a recursive function for
subsequences. Then postcondition says result = length of longest common one."
Why LIKELY: Acknowledges need for auxiliary definition. Approach is sound even if details missing.

EXAMPLE 12 — POSSIBLE (5) — REPETITION:
Trajectory: "The LCS length should be computed correctly. Verify the computation is correct.
Confirm the result matches the LCS definition."
Why POSSIBLE: Three sentences, ONE idea. No actual Lean syntax or helper definition.

---

## KEY PENALTIES

1. **ENGLISH PROSE**: "The X must be Y" → unlikely (3) unless followed by Lean syntax
2. **MADE-UP FUNCTIONS**: Anything not in Lean stdlib without defining it → unlikely (3)
3. **PYTHON SYNTAX**: `==`, `len()`, `max()`, `.count()` without module → unlikely (3)
4. **REPETITION**: Same property phrased multiple ways → score based on unique content only
5. **NO SOUNDNESS/COMPLETENESS**: Specs without reasoning about what they accept/reject → likely
(7) max

## YOUR EVALUATION

Respond with "Confidence: <level>" followed by:
1. Does the reasoning produce actual Lean Prop syntax or just English?
2. Are the Lean constructs mentioned real (List.count, ∧, ∀) or made-up?
3. Is there soundness/completeness reasoning?
4. Is there repetition?
"""

VERINA_SPEC_VALUE_PROMPT_SIMPLE = """
Evaluate a Lean 4 specification-generation reasoning trajectory.

PROBLEM:
{problem}

TRAJECTORY:
{trajectory}

The trajectory contains step-by-step reasoning toward Lean 4 preconditions and postconditions.
Judge if the reasoning is on a sound path to producing valid, sound, and complete specifications.

Key scoring factors:
- Could you write the final [PRECOND] and [POSTCOND] RIGHT NOW from this trajectory? If yes → sure
(9).
- Steps that rephrase or "verify" earlier steps without adding new specifics count as repetition.
Repetitive trajectories should NEVER score sure.
- A single high-level idea without Lean 4 Prop specifics should score possible (5) at most.
- Concrete Lean 4 Prop syntax (∧, ∨, ¬, ∀, ∃) and soundness/completeness reasoning push toward
likely (7) or sure (9).

Use the confidence rubric:
- sure (9): SPEC-READY — correct Lean 4 Prop syntax AND soundness+completeness reasoning, no gaps
- likely (7): Correct approach with specific Lean 4 Prop constructs, minor gaps
- possible (5): Right direction but missing Lean 4 Prop details, OR multi-step but repetitive
- unlikely (3): Fundamental issues in approach or syntax
- impossible (1): Completely off-track or wrong language

Respond with "Confidence: <level>" and brief justification. Note if steps are repetitive.
"""

# ============================================================================
# GENERIC VALUE PROMPT
# ============================================================================

GENERIC_VALUE_PROMPT_WITH_FEWSHOT = """
Evaluate how close the following trajectory is to solving the problem.

PROBLEM:
{problem}

CURRENT TRAJECTORY:
{trajectory}

Here are examples of trajectory evaluations:

EXAMPLE 1 (Strong progress):
Trajectory: I've broken down the problem into steps, identified key constraints, and I'm halfway
through the solution with correct logic so far.
Confidence: likely/probably (7)
Justification: Clear methodology and correct intermediate progress toward the solution.

EXAMPLE 2 (Uncertain progress):
Trajectory: I've started the problem and my approach seems reasonable, but I'm not confident about
the next steps.
Confidence: possible/maybe (5)
Justification: Direction is sound but execution and completeness require verification.

EXAMPLE 3 (Low chance of success):
Trajectory: I tried an approach but it seems to have led to a contradiction.
Confidence: unlikely/doubtful (3)
Justification: The approach has fundamental issues that need to be reconsidered.

Rate the state on the scale: sure/certain (9), likely/probably (7), possible/maybe (5),
unlikely/doubtful (3), impossible/blocked (1).
Respond with "Confidence: <category>" and a short rationale.
"""

GENERIC_VALUE_PROMPT_SIMPLE = """
Evaluate how close the following trajectory is to solving the problem.

PROBLEM:
{problem}

CURRENT TRAJECTORY:
{trajectory}

Rate the state on the scale: sure/certain (9), likely/probably (7), possible/maybe (5),
unlikely/doubtful (3), impossible/blocked (1).
Respond with "Confidence: <category>" and a short rationale.
"""


def build_game24_value_prompt(
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """Build game24 value prompt with or without few-shot examples."""
    if use_fewshot:
        return GAME24_VALUE_PROMPT_WITH_FEWSHOT.format(
            problem=problem, trajectory=trajectory
        )
    else:
        return GAME24_VALUE_PROMPT_SIMPLE.format(
            problem=problem, trajectory=trajectory
        )


def build_mcq_value_prompt(
    problem: str,
    trajectory: str,
    task_name: str,
    use_fewshot: bool = True
) -> str:
    """Build MCQ (maze/spatial) value prompt with or without few-shot examples."""
    if task_name.lower() == "maze":
        if use_fewshot:
            return MAZE_VALUE_PROMPT_WITH_FEWSHOT.format(
                problem=problem, trajectory=trajectory
            )
        else:
            return MAZE_VALUE_PROMPT_SIMPLE.format(
                problem=problem, trajectory=trajectory
            )
    elif task_name.lower() in (
        "spatial", "spatialmap", "spatial reasoning"
    ):
        if use_fewshot:
            return SPATIALMAP_VALUE_PROMPT_WITH_FEWSHOT.format(
                problem=problem, trajectory=trajectory
            )
        else:
            return SPATIALMAP_VALUE_PROMPT_SIMPLE.format(
                problem=problem, trajectory=trajectory
            )
    else:
        # Default MCQ template
        if use_fewshot:
            return MAZE_VALUE_PROMPT_WITH_FEWSHOT.format(
                problem=problem, trajectory=trajectory
            )
        else:
            return MAZE_VALUE_PROMPT_SIMPLE.format(
                problem=problem, trajectory=trajectory
            )


def build_generic_value_prompt(
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """Build generic value prompt with or without few-shot examples."""
    if use_fewshot:
        return GENERIC_VALUE_PROMPT_WITH_FEWSHOT.format(
            problem=problem, trajectory=trajectory
        )
    else:
        return GENERIC_VALUE_PROMPT_SIMPLE.format(
            problem=problem, trajectory=trajectory
        )


def build_zebralogic_value_prompt(
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """Build zebralogic value prompt with or without few-shot examples."""
    if use_fewshot:
        return ZEBRALOGIC_VALUE_PROMPT_WITH_FEWSHOT.format(
            problem=problem, trajectory=trajectory
        )
    else:
        return ZEBRALOGIC_VALUE_PROMPT_SIMPLE.format(
            problem=problem, trajectory=trajectory
        )


def build_verina_value_prompt(
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """Build verina (Lean 4 code generation) value prompt
    with or without few-shot examples."""
    if use_fewshot:
        return VERINA_VALUE_PROMPT_WITH_FEWSHOT.format(
            problem=problem, trajectory=trajectory
        )
    else:
        return VERINA_VALUE_PROMPT_SIMPLE.format(
            problem=problem, trajectory=trajectory
        )


def build_verina_spec_value_prompt(
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """Build verina spec (Lean 4 spec generation) value prompt
    with or without few-shot examples."""
    if use_fewshot:
        return VERINA_SPEC_VALUE_PROMPT_WITH_FEWSHOT.format(
            problem=problem, trajectory=trajectory
        )
    else:
        return VERINA_SPEC_VALUE_PROMPT_SIMPLE.format(
            problem=problem, trajectory=trajectory
        )


def build_tot_value_prompt(
    task: str,
    problem: str,
    trajectory: str,
    use_fewshot: bool = True
) -> str:
    """
    Build value prompt for Tree of Thought evaluation.

    Args:
        task: The task type
            (e.g., "game24", "maze", "spatialmap",
            "zebralogic", "verina")
        problem: The original problem statement
        trajectory: Current partial solution
        use_fewshot: Whether to use few-shot examples
            (default True)

    Returns:
        Formatted value prompt
    """
    
    if task == "game24":
        return build_game24_value_prompt(
            problem, trajectory, use_fewshot
        )
    

    if task == "maze":
        return build_mcq_value_prompt(
            problem, trajectory, "maze", use_fewshot
        )


    if task == "spatialmap":
        return build_mcq_value_prompt(
            problem, trajectory, "spatial reasoning", use_fewshot
        )


    if task == "zebralogic":
        return build_zebralogic_value_prompt(
            problem, trajectory, use_fewshot
        )


    if task == "verina":
        return build_verina_value_prompt(
            problem, trajectory, use_fewshot
        )


    if task == "verina_spec":
        return build_verina_spec_value_prompt(
            problem, trajectory, use_fewshot
        )


    return build_generic_value_prompt(
        problem, trajectory, use_fewshot
    )