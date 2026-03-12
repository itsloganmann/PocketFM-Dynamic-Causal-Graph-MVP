"""
Causal propagation module.

Responsible for propagating belief updates through the causal graph structure.
This ensures that changes in one belief (the antecedent) influence causally
downstream beliefs (the consequent).

Mechanism:
    For each causal link A -> B with weight w:
    delta_L(B) = rate * w * tanh(L(A) / 2)

    This means:
    - If A is strongly believed (L(A) > 0), B receives positive evidence.
    - If A is strongly disbelieved (L(A) < 0), B receives negative evidence.
    - If A is uncertain (L(A) ~ 0), B is unaffected.
"""

import math
from typing import Dict, List, Any, Optional

from core.data_structures import CharacterState, BeliefNode
from reasoning.belief_update import resolve_belief_conflicts

def propagate_causal_effects(
    state: CharacterState,
    prev_log_odds: Optional[Dict[str, float]] = None,
    propagation_rate: float = 0.1
) -> None:
    """
    Apply causal propagation rules to the belief network.

    Iterates through the character's causal_links and updates the log-odds
    of consequent beliefs based on the strength of antecedent beliefs.

    Preconditions
    -------------
    state : CharacterState
        Must contain initialized beliefs and causal_links.

    Procedure
    ---------
    1. For each belief that has children in the causal graph:
       a. Compute Δ = current log_odds - previous log_odds
       b. For each child link with weight w:
          child.log_odds += propagation_rate * w * Δ
    2. Resolve any belief conflicts after all updates.

    Postconditions
    --------------
    state.beliefs updated in place.

    Parameters
    ----------
    state : CharacterState
    propagation_rate : float
        Scalar to control the speed of propagation per turn.
    """
    if not state.causal_links:
        return
    if prev_log_odds is None:
        return
    updates: Dict[str, float] = {}

    # Iterate over all beliefs that might have changed
    for prop, node in state.beliefs.items():
        prev_val = prev_log_odds.get(prop)
        if prev_val is None:
            continue

        # Compute how much this belief changed this turn
        delta = node.log_odds - prev_val

        # If nothing changed, no propagation needed
        if abs(delta) < 1e-9:
            continue

        # Propagate delta to all children via get_children()
        for link in state.get_children(prop):
            cons_name = link["consequent"]
            weight = link.get("weight", 1.0)

            impact = propagation_rate * weight * delta

            updates[cons_name] = updates.get(cons_name, 0.0) + impact

    # Apply accumulated updates
    for prop, delta in updates.items():
        node = state.get_belief(prop)
        # double-check valid endpoints 
        if node is None:
            continue
        node.log_odds += delta

        # Mark provenance if update is significant
        if abs(delta) > 0.01:
            node.add_evidence("causal_propagation")

    # Ensure beliefs remain logically consistent
    resolve_belief_conflicts(state.beliefs)
def snapshot_belief_log_odds(state: CharacterState) -> Dict[str, float]:
    return {
        prop: node.log_odds
        for prop, node in state.beliefs.items()
    }

