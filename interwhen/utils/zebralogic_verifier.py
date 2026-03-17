"""
ZebraLogic Verifier

Z3-based constraint solver for verifying ZebraLogic house grid problem assignments.
Uses the ZebraLogic dataset from WildEval/ZebraLogic.
"""

import re
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from importlib import resources
from z3 import Solver, And, Or, Not, Bool, PbEq


# ============== ZebraLogicProblem ==============

class ZebraLogicProblem:
    """Z3-based constraint solver for ZebraLogic house grid problems.

    Encodes the problem constraints using Z3 boolean variables and supports
    checking satisfiability after adding candidate assignments.

    Args:
        problem: Dict with keys 'n_houses', 'features', 'clues', 'clue_irs'.
    """

    def __init__(self, problem: dict):
        self.solver = Solver()

        self.n_houses = problem['n_houses']
        self.houses = list(range(self.n_houses))
        self.features = self._make_features(problem['features'])
        self.clue_texts = problem['clues']

    def _make_features(self, features_domains: dict) -> dict:
        """Create Z3 boolean variables for each feature/value/house combination.

        Returns:
            dict[feature][value][house] -> Bool
        """

        features = defaultdict(lambda: defaultdict(dict))
        for feature_name, feature_domain in features_domains.items():
            for value in feature_domain:
                for h in self.houses:
                    features[feature_name][value][h] = Bool(f"{feature_name}={value}@{h}")

        features = dict(features)
        for feature_name in features.keys():
            features[feature_name] = dict(features[feature_name])

        for feature_name, feature in features.items():
            # Exactly one value per house
            for h in self.houses:
                self.solver.add(
                    PbEq([(feature[v][h], 1) for v in feature.keys()], 1)
                )
            # Exactly one house per value
            for v in feature.keys():
                self.solver.add(
                    PbEq([(feature[v][h], 1) for h in self.houses], 1)
                )

        return features

    def compile_constraint(self, ir: dict):
        """Compile an intermediate representation (IR) constraint into a Z3 BoolRef.

        Supported constraint types:
            place: Entity at specific position
            not_place: Entity NOT at specific position
            same_house: Two entities at same position
            pos_relation: Spatial relationship between entities
        """
        from z3 import And, Or, Not

        t = ir["type"]

        if t == "place":
            f, v = ir["entity"]
            h = ir["pos"] - 1  # Convert 1-indexed to 0-indexed
            return self.features[f][v][h]

        if t == "not_place":
            f, v = ir["entity"]
            h = ir["pos"] - 1
            return Not(self.features[f][v][h])

        if t == "same_house":
            (f1, v1) = ir["A"]
            (f2, v2) = ir["B"]
            return And(*[
                self.features[f1][v1][h] == self.features[f2][v2][h]
                for h in self.houses
            ])

        if t == "pos_relation":
            (f1, v1) = ir["A"]
            (f2, v2) = ir["B"]
            direction = ir["direction"]
            dist = ir["dist"]

            def A(h): return self.features[f1][v1][h]
            def B(h): return self.features[f2][v2][h]

            def k_to_left(k):
                clauses = []
                if k == "?":
                    for h1 in self.houses:
                        for h2 in self.houses:
                            if h1 < h2:
                                clauses.append(And(A(h1), B(h2)))
                else:
                    k = int(k)
                    for h in range(0, self.n_houses - k):
                        clauses.append(And(A(h), B(h + k)))
                return clauses

            def k_to_right(k):
                clauses = []
                if k == "?":
                    for h1 in self.houses:
                        for h2 in self.houses:
                            if h1 > h2:
                                clauses.append(And(A(h1), B(h2)))
                else:
                    k = int(k)
                    for h in range(k, self.n_houses):
                        clauses.append(And(A(h), B(h - k)))
                return clauses

            if direction == "left":
                clauses = k_to_left(dist)
            elif direction == "right":
                clauses = k_to_right(dist)
            elif direction == "any":
                clauses = k_to_left(dist) + k_to_right(dist)
            else:
                raise ValueError(f"Unknown direction: {direction}")

            return Or(*clauses)

        raise ValueError(f"Unknown IR type: {t}")

    def apply_ir(self, ir):
        """Apply a list of IR constraints to the solver (SOP format)."""

        if len(ir) == 0:
            return
        if isinstance(ir, list) and isinstance(ir[0], dict):
            ir = [ir]

        compiled_disjuncts = []
        for disjunct in ir:
            assert isinstance(disjunct, list), "Each disjunct must be a list of conjuncts"
            compiled_conjuncts = [self.compile_constraint(c) for c in disjunct]
            compiled_disjuncts.append(And(*compiled_conjuncts))
        self.solver.add(Or(*compiled_disjuncts))

    @property
    def is_satisfiable(self) -> bool:
        """Check if the current constraints are satisfiable."""
        from z3 import sat
        return self.solver.check() == sat
