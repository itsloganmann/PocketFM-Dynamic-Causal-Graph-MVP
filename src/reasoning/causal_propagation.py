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
    propagation_rate: float = 0.1
) -> None:
    """
    Apply causal propagation rules to the belief network.

    Iterates through the character's causal_links and updates the log-odds
    of consequent beliefs based on the strength of antecedent beliefs.

    Mechanism:
        For each causal link A -> B with weight w:
        delta_L(B) = propagation_rate * w * tanh(L(A) / 2)

    Preconditions
    -------------
    state : CharacterState
        Must contain initialized beliefs and causal_links.

    Procedure
    ---------
    1. For each causal link:
       a. Retrieve antecedent belief A and consequent belief B.
       b. Compute impact = propagation_rate * weight * tanh(L(A) / 2).
       c. Accumulate impacts for all consequents.
    2. Apply impacts to belief log-odds.
    3. Resolve any belief conflicts after all updates.

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

    updates: Dict[str, float] = {}

    # Iterate over all causal links
    for link in state.causal_links:
        ant_name = link["antecedent"]
        cons_name = link["consequent"]
        weight = link.get("weight", 1.0)

        ant_node = state.get_belief(ant_name)
        if ant_node is None:
            continue

        # Compute impact using tanh(L(A)/2)
        # This reflects the strength of the belief A influencing B.
        strength = math.tanh(ant_node.log_odds / 2.0)
        impact = propagation_rate * weight * strength

        updates[cons_name] = updates.get(cons_name, 0.0) + impact

    # Apply accumulated updates
    for prop, delta in updates.items():
        node = state.get_belief(prop)
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

