"""Preregistered core variable set for Q1 E1."""

E1_CORE_TARGETS = (
    # Stance state
    "stance_score",
    "stance_gap",
    # Agreement and conflict
    "local_agreement",
    "remaining_disagreement",
    "affiliation",
    "adversariality",
    "observed_alignment_index",
    "observed_conflict_index",
    "observed_accommodation_index",
    # Small conversational-personality set
    "perceived_persona_warmth_trailing3",
    "perceived_persona_dominance_trailing3",
    "perceived_persona_humility_trailing3",
    # Expressed affect from the text VAD regressor
    "expressed_valence",
    "expressed_arousal",
    "expressed_dominance",
)


E1_REPRESENTATIVE_CURVES = (
    "stance_score",
    "stance_gap",
    "observed_alignment_index",
    "observed_conflict_index",
    "perceived_persona_warmth_trailing3",
    "perceived_persona_dominance_trailing3",
    "expressed_valence",
    "expressed_arousal",
    "expressed_dominance",
)

