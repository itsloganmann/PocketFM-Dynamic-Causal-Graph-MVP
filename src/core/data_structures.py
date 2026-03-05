"""
Defines the shared schemas used throughout the system.

These structures represent the causal character graph state,
world state, and structured event frames as defined in the
Dynamic Causal Character Graphs paper.
"""

from __future__ import annotations

import math
import copy
from typing import Dict, List, Optional, Any, Set

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi]."""
    return max(lo, min(hi, float(value)))


def _validate_unit(value: float, name: str) -> float:
    """Validate and clamp a value to [0, 1]."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return _clamp(v, 0.0, 1.0)


def _validate_signed_unit(value: float, name: str) -> float:
    """Validate and clamp a value to [-1, 1]."""
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return _clamp(v, -1.0, 1.0)

# ===================================================================
# TraitState
# ===================================================================

class TraitState:
    """
    Represents stable personality traits.
    
    τ_j ∈ [-1, 1] or [0, 1].
    These nodes have low plasticity and evolve slowly over time.
    Ex: bravery, honesty, curiosity, risk-aversion.

    Attributes
    ----------
    traits : dict[str, float]
        Mapping of trait name → intensity (clamped to [-1, 1]).
    plasticity : float
        Base plasticity α ∈ [0, 1] — should be small for traits.
    """

    def __init__(self, traits: Dict[str, float], plasticity: float = 0.05):
        if not isinstance(traits, dict):
            raise TypeError("traits must be a dict")
        # Store each trait clamped to [-1, 1]
        self.traits: Dict[str, float] = {
            str(k): _validate_signed_unit(v, f"trait '{k}'")
            for k, v in traits.items()
        }
        self.plasticity: float = _validate_unit(plasticity, "plasticity")

    # -- accessors ---------------------------------------------------------

    def get(self, name: str, default: float = 0.0) -> float:
        """Return trait intensity, or *default* if the trait is absent."""
        return self.traits.get(name, default)

    def set_trait(self, name: str, value: float) -> None:
        """Set a single trait (clamped)."""
        self.traits[str(name)] = _validate_signed_unit(value, f"trait '{name}'")

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {"traits": dict(self.traits), "plasticity": self.plasticity}

    @classmethod
    def from_dict(cls, d: dict) -> "TraitState":
        return cls(traits=d["traits"], plasticity=d.get("plasticity", 0.05))

    def __repr__(self) -> str:
        return f"TraitState(traits={self.traits}, α={self.plasticity})"

# ===================================================================
# EmotionState
# ===================================================================

class EmotionState:
    """
    Low-dimensional affective state.
    --------------
    e_t = (v_t, a_t)  where v = valence, a = arousal.
    Discrete emotional tags (fear, anger, joy …) are intensity
    values in [0, 1].  Emotions are FAST nodes (large α).

    Attributes
    ----------
    valence : float
        Positive/negative affect in [-1, 1].
    arousal : float
        Activation level in [0, 1].
    emotion_tags : Dict[str, float]
        Named emotion intensities in [0, 1].
    plasticity : float
        Rate at which emotions update. Fast nodes have high plasticity.
    """

    def __init__(
        self,
        valence: float = 0.0,
        arousal: float = 0.0,
        emotion_tags: Optional[Dict[str, float]] = None,
        plasticity: float = 0.8,
    ):
        self.valence: float = _validate_signed_unit(valence, "valence")
        self.arousal: float = _validate_unit(arousal, "arousal")
        self.emotion_tags: Dict[str, float] = {}
        if emotion_tags:
            for k, v in emotion_tags.items():
                self.emotion_tags[str(k)] = _validate_unit(v, f"emotion_tag '{k}'")
        self.plasticity: float = _validate_unit(plasticity, "plasticity")

    # -- convenience -------------------------------------------------------

    def dominant_emotion(self) -> Optional[str]:
        """Return the tag with highest intensity, or None."""
        if not self.emotion_tags:
            return None
        return max(self.emotion_tags, key=self.emotion_tags.get)

    def set_tag(self, tag: str, intensity: float) -> None:
        self.emotion_tags[str(tag)] = _validate_unit(intensity, f"emotion_tag '{tag}'")

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "emotion_tags": dict(self.emotion_tags),
            "plasticity": self.plasticity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionState":
        return cls(
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.0),
            emotion_tags=d.get("emotion_tags"),
            plasticity=d.get("plasticity", 0.8),
        )

    def __repr__(self) -> str:
        dom = self.dominant_emotion()
        return (
            f"EmotionState(v={self.valence:.2f}, a={self.arousal:.2f}, "
            f"dominant={dom}, α={self.plasticity})"
        )

# ===================================================================
# RelationshipState
# ===================================================================

class RelationshipState:
    """
    Represents relationship values with another entity.

    --------------
    R_t(x) = (trust, affection, respect) ∈ [0, 1]³.
    Semi-stable nodes — moderate plasticity.

    Attributes
    ----------
    entity_id : str       who this relationship is with
    trust     : float     ∈ [0, 1]
    affection : float     ∈ [0, 1]
    respect   : float     ∈ [0, 1]
    plasticity: float     ∈ [0, 1]
    """

def __init__(
        self,
        entity_id: str = "",
        trust: float = 0.5,
        affection: float = 0.5,
        respect: float = 0.5,
        plasticity: float = 0.3,
    ):
        self.entity_id: str = str(entity_id)
        self.trust: float = _validate_unit(trust, "trust")
        self.affection: float = _validate_unit(affection, "affection")
        self.respect: float = _validate_unit(respect, "respect")
        self.plasticity: float = _validate_unit(plasticity, "plasticity")

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "trust": self.trust,
            "affection": self.affection,
            "respect": self.respect,
            "plasticity": self.plasticity,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RelationshipState":
        return cls(**d)

    def __repr__(self) -> str:
        return (
            f"RelationshipState(entity={self.entity_id!r}, "
            f"trust={self.trust:.2f}, aff={self.affection:.2f}, "
            f"resp={self.respect:.2f}, α={self.plasticity})"
        )


class BeliefNode:
    """
    Represents a belief proposition using log-odds encoding.

    The log-odds representation enables additive evidence updates:
        l_t = ln(p_t / (1 - p_t))

    A value of 0 represents maximum uncertainty (p = 0.5).

    Attributes
    ----------
    proposition : str
        The propositional content, e.g. "player_has_key".
    log_odds : float
        Epistemic confidence expressed as log-odds.
    evidence_sources : List[str]
        Provenance metadata tracking the source of each update.
    """

    def __init__(self, proposition: str, log_odds: float = 0.0):
        self.proposition: str = proposition
        self.log_odds: float = log_odds
        self.evidence_sources: List[str] = []


class CharacterState:
    """
    Full internal character state in the Dynamic SCM.

    Attributes
    ----------
    traits : TraitState
    emotions : EmotionState
    beliefs : Dict[str, BeliefNode]
        Keyed by normalised proposition string.
    relationships : Dict[str, RelationshipState]
        Keyed by entity name.
    intentions : List[str]
        Current behavioural goals.
    timeline_index : int
        Current position on the character's knowledge timeline.
    """

    def __init__(self):
        self.traits: TraitState = TraitState({}, plasticity=0.05)
        self.emotions: EmotionState = EmotionState()
        self.beliefs: Dict[str, BeliefNode] = {}
        self.relationships: Dict[str, RelationshipState] = {}
        self.intentions: List[str] = []
        self.timeline_index: int = 0


class WorldState:
    """
    Represents the objective world state (canonical narrative truth).

    Unlike CharacterState, this graph may contain facts unknown to
    any single character.

    Attributes
    ----------
    entities : Dict[str, dict]
        Entity attribute maps.
    object_states : Dict[str, str]
        Current state label for each object.
    constraints : List[str]
        Immutable narrative rules (forbidden abilities, etc.).
    timeline_index : int
    """

    def __init__(self):
        self.entities: Dict[str, dict] = {}
        self.object_states: Dict[str, str] = {}
        self.constraints: List[str] = []
        self.timeline_index: int = 0


class EventFrame:
    """
    Structured event representation extracted from user dialogue.

    Follows the schema:
        e_t = (P_t, E_t, a_t, tau_t, c_t)

    Attributes
    ----------
    propositions : List[str]
        Asserted propositions P_t.
    entities : List[str]
        Referenced entities E_t.
    speaker : Optional[str]
        Speaker identity a_t. None or "DirectObservation" for
        first-person observations.
    emotional_tone : Optional[str]
        Inferred emotional tone tau_t (e.g. "anger", "joy").
    confidence : float
        Extraction confidence c_t in [0, 1].
    """

    def __init__(
        self,
        propositions: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        speaker: Optional[str] = None,
        emotional_tone: Optional[str] = None,
        confidence: float = 1.0,
    ):
        self.propositions: List[str] = propositions if propositions is not None else []
        self.entities: List[str] = entities if entities is not None else []
        self.speaker: Optional[str] = speaker
        self.emotional_tone: Optional[str] = emotional_tone
        self.confidence: float = confidence
