"""Source-aware Track 1 variable registry and deterministic transcript derivations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Source = Literal["internal_report", "external_annotation", "response_implied", "derived"]

BIG_FIVE_TRAITS = (
    "extraversion", "agreeableness", "conscientiousness", "neuroticism", "openness",
)
PERCEIVED_PERSONA_FIELDS = tuple(
    f"perceived_persona_{dimension}"
    for dimension in (
        "warmth", "dominance", "curiosity", "structure", "stability",
        "deference", "humility",
    )
)
OBSERVER_BIG_FIVE_FIELDS = tuple(f"observer_big5_{trait}" for trait in BIG_FIVE_TRAITS)
DYNAMIC_PERSONA_FIELDS = (*PERCEIVED_PERSONA_FIELDS, *OBSERVER_BIG_FIVE_FIELDS)


@dataclass(frozen=True)
class Variable:
    name: str
    source: Source
    task: Literal["continuous", "categorical", "multilabel"]
    timing: Literal["current_response", "trailing_window", "transition"]
    description: str


VARIABLES = (
    Variable("stance_score", "internal_report", "continuous", "current_response", "Raw topic stance."),
    Variable("role_aligned_stance", "derived", "continuous", "current_response", "Stance recoded toward the assigned role."),
    Variable("stance_confidence", "internal_report", "continuous", "current_response", "Reported stance confidence."),
    Variable("stance_change", "derived", "continuous", "transition", "Change from the speaker's prior response."),
    Variable("partnerward_stance_movement", "derived", "continuous", "transition", "Reduction in distance to the partner's prior stance."),
    Variable("stance_gap", "derived", "continuous", "current_response", "Absolute gap from the partner's prior stance."),
    Variable("local_agreement", "external_annotation", "continuous", "current_response", "Locally expressed agreement."),
    Variable("remaining_disagreement", "external_annotation", "continuous", "current_response", "Remaining substantive disagreement."),
    Variable("affiliation", "external_annotation", "continuous", "current_response", "Affiliative interaction style."),
    Variable("adversariality", "external_annotation", "continuous", "current_response", "Adversarial interaction style."),
    Variable("emotional_tone", "external_annotation", "categorical", "current_response", "Observed emotional tone."),
    Variable("expressed_valence", "derived", "continuous", "current_response", "Valence expressed in response text from the registered VAD model."),
    Variable("expressed_arousal", "derived", "continuous", "current_response", "Arousal expressed in response text from the registered VAD model."),
    Variable("expressed_dominance", "derived", "continuous", "current_response", "Dominance expressed in response text from the registered VAD model."),
    Variable("realized_move", "external_annotation", "categorical", "current_response", "Observed conversational move."),
    Variable("perceived_persona_warmth", "external_annotation", "continuous", "trailing_window", "Perceived warmth/agreeableness."),
    Variable("perceived_persona_dominance", "external_annotation", "continuous", "trailing_window", "Perceived dominance/assertiveness."),
    Variable("perceived_persona_curiosity", "external_annotation", "continuous", "trailing_window", "Perceived curiosity/openness."),
    Variable("perceived_persona_structure", "external_annotation", "continuous", "trailing_window", "Perceived structure/conscientiousness."),
    Variable("perceived_persona_stability", "external_annotation", "continuous", "trailing_window", "Perceived emotional stability/reactivity."),
    Variable("perceived_persona_deference", "external_annotation", "continuous", "trailing_window", "Perceived deference versus force."),
    Variable("perceived_persona_humility", "external_annotation", "continuous", "trailing_window", "Perceived epistemic humility/hedging."),
    Variable("observer_big5_extraversion", "external_annotation", "continuous", "current_response", "Direct observer rating of extraverted conversational presentation; not a BFI-44 score."),
    Variable("observer_big5_agreeableness", "external_annotation", "continuous", "current_response", "Direct observer rating of agreeable conversational presentation; not a BFI-44 score."),
    Variable("observer_big5_conscientiousness", "external_annotation", "continuous", "current_response", "Direct observer rating of conscientious conversational presentation; not a BFI-44 score."),
    Variable("observer_big5_neuroticism", "external_annotation", "continuous", "current_response", "Direct observer rating of neurotic or reactive conversational presentation; not a BFI-44 score."),
    Variable("observer_big5_openness", "external_annotation", "continuous", "current_response", "Direct observer rating of open conversational presentation; not a BFI-44 score."),
    *(
        Variable(
            f"observer_big5_{trait}_confidence", "external_annotation", "continuous",
            "current_response", f"Visible-evidence confidence for the {trait} observer rating.",
        )
        for trait in BIG_FIVE_TRAITS
    ),
    Variable("apparent_objective", "response_implied", "categorical", "current_response", "What the completed response appears designed to accomplish."),
    Variable("response_implied_expected_reaction", "response_implied", "categorical", "current_response", "Partner reaction apparently anticipated by the response."),
    Variable("explicit_synthesis", "external_annotation", "categorical", "current_response", "Explicit synthesis behavior."),
    Variable("explicit_resolution", "external_annotation", "categorical", "current_response", "Explicit resolution behavior."),
    Variable("explicit_closure", "external_annotation", "categorical", "current_response", "Explicit synthesis, resolution, or closure behavior."),
    Variable("semantic_velocity", "derived", "continuous", "transition", "Response-embedding displacement."),
    Variable("semantic_acceleration", "derived", "continuous", "transition", "Change in semantic velocity."),
    Variable("observed_conflict_index", "derived", "continuous", "current_response", "Observed conflict index."),
    Variable("observed_alignment_index", "derived", "continuous", "current_response", "Observed alignment index."),
    Variable("observed_accommodation_index", "derived", "continuous", "transition", "Observed accommodation index."),
    Variable("observed_exploration_index", "derived", "continuous", "current_response", "Observed exploration index."),
    Variable("closure_evidence", "derived", "continuous", "current_response", "Observable closure evidence."),
    Variable("observable_transition", "derived", "categorical", "transition", "Prespecified observable transition event."),
    Variable("basin_leaning", "derived", "continuous", "current_response", "Position on the model-basin axis."),
    Variable("partnerward_basin_velocity", "derived", "continuous", "transition", "Movement toward the partner basin."),
    Variable("off_axis_distance", "derived", "continuous", "current_response", "Distance orthogonal to the basin axis."),
)

VARIABLES += tuple(
    Variable(
        f"{field}_{suffix}", "derived", "continuous", timing, description.format(field=field)
    )
    for field in DYNAMIC_PERSONA_FIELDS
    for suffix, timing, description in (
        ("trailing3", "trailing_window", "Trailing-three state for {field}."),
        ("self_play_baseline", "trailing_window", "Same-model self-play baseline for {field}."),
        ("deviation_from_self_play", "trailing_window", "Deviation from the same-model self-play baseline for {field}."),
        ("movement", "transition", "Change in the trailing-three state for {field}."),
    )
)


def registry_frame() -> pd.DataFrame:
    return pd.DataFrame([variable.__dict__ for variable in VARIABLES])


def derive_stance_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add transcript-only variables without changing source transcript files."""
    out = df.sort_values(["conv_id", "turn"]).copy()
    stance = pd.to_numeric(out["stance_score"], errors="coerce")
    out["raw_topic_stance"] = stance
    out["role_aligned_stance"] = np.where(
        out["role"].eq("opposer"), 6.0 - stance, stance
    )
    out["prior_stance_score"] = out.groupby(["conv_id", "speaker"])["stance_score"].shift()
    out["stance_change"] = stance - out["prior_stance_score"]
    prior_partner_stance = []
    for _, conversation in out.groupby("conv_id", sort=False):
        latest = {}
        for _, row in conversation.iterrows():
            partner = "b" if row["speaker"] == "a" else "a"
            prior_partner_stance.append(latest.get(partner, np.nan))
            latest[row["speaker"]] = row["stance_score"]
    out["partner_prior_stance"] = prior_partner_stance
    out["stance_gap"] = (stance - out["partner_prior_stance"]).abs()
    previous_gap = out.groupby(["conv_id", "speaker"])["stance_gap"].shift()
    out["partnerward_stance_movement"] = previous_gap - out["stance_gap"]
    return out


def merge_annotations(df: pd.DataFrame, annotation_path: str | None) -> pd.DataFrame:
    """Merge offline annotations by immutable (conv_id, turn), rejecting duplicates."""
    if not annotation_path:
        return df
    annotations = pd.read_csv(annotation_path)
    keys = ["conv_id", "turn"]
    missing = [key for key in keys if key not in annotations]
    if missing:
        raise ValueError(f"Annotation file is missing keys: {missing}")
    allowed = {variable.name for variable in VARIABLES}
    unknown = set(annotations) - set(keys) - allowed - {"annotator_id"}
    if unknown:
        raise ValueError(f"Unregistered annotation variables: {sorted(unknown)}")
    if annotations.duplicated(keys).any():
        if "annotator_id" not in annotations:
            raise ValueError("Duplicate annotations require annotator_id.")
        annotations = aggregate_annotations(annotations)
    return df.merge(annotations, on=keys, how="left", validate="one_to_one")


def add_persona_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Add dynamic interaction-style and direct Big Five observer states."""
    out = df.sort_values(["conv_id", "speaker", "turn"]).copy()
    for column in DYNAMIC_PERSONA_FIELDS:
        if column not in out:
            continue
        trailing = out.groupby(["conv_id", "speaker"])[column].transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
        state_column = f"{column}_trailing3"
        out[state_column] = trailing
        baseline = (
            out[out["condition"].eq("self_play")]
            .groupby("model")[state_column]
            .mean()
        )
        baseline_column = f"{column}_self_play_baseline"
        out[baseline_column] = out["model"].map(baseline)
        out[f"{column}_deviation_from_self_play"] = trailing - out[baseline_column]
        out[f"{column}_movement"] = out.groupby(
            ["conv_id", "speaker"]
        )[state_column].diff()
    return out


def annotation_reliability(annotation_path: str) -> pd.DataFrame:
    """Report transparent pairwise reliability for raw multi-annotator labels."""
    from itertools import combinations

    annotations = pd.read_csv(annotation_path)
    if "annotator_id" not in annotations:
        return pd.DataFrame([{
            "variable": None, "metric": "not_evaluated", "value": np.nan,
            "reason": "annotator_id column absent",
        }])
    registered = {variable.name: variable for variable in VARIABLES}
    rows = []
    annotators = sorted(annotations["annotator_id"].dropna().unique())
    for name, variable in registered.items():
        if name not in annotations:
            continue
        pair_values = []
        for first, second in combinations(annotators, 2):
            left = annotations[annotations["annotator_id"].eq(first)][["conv_id", "turn", name]]
            right = annotations[annotations["annotator_id"].eq(second)][["conv_id", "turn", name]]
            paired = left.merge(right, on=["conv_id", "turn"], suffixes=("_a", "_b")).dropna()
            if paired.empty:
                continue
            if variable.task == "continuous":
                a = pd.to_numeric(paired[f"{name}_a"], errors="coerce")
                b = pd.to_numeric(paired[f"{name}_b"], errors="coerce")
                valid = a.notna() & b.notna()
                pair_values.extend(np.abs(a[valid] - b[valid]).tolist())
            else:
                pair_values.extend((paired[f"{name}_a"] == paired[f"{name}_b"]).astype(float).tolist())
        metric = "mean_pairwise_absolute_difference" if variable.task == "continuous" else "pairwise_agreement"
        rows.append({
            "variable": name, "metric": metric,
            "value": float(np.mean(pair_values)) if pair_values else np.nan,
            "n_pairwise": len(pair_values),
        })
    return pd.DataFrame(rows)


def aggregate_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw annotator rows to one auditable row per recorded turn."""
    registered = {variable.name: variable for variable in VARIABLES}
    aggregations = {}
    for column in annotations:
        if column not in registered:
            continue
        if registered[column].task == "continuous":
            aggregations[column] = "mean"
        else:
            aggregations[column] = lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan
    return annotations.groupby(["conv_id", "turn"], as_index=False).agg(aggregations)
def add_lagged_behavioral_state(df: pd.DataFrame) -> pd.DataFrame:
    """Expose only state observed before the current speaker response."""
    out = df.sort_values(["conv_id", "speaker", "turn"]).copy()
    state_columns = (
        "stance_score", "stance_confidence", "stance_gap", "local_agreement",
        "remaining_disagreement", "affiliation", "adversariality",
        "closure_evidence", "basin_leaning",
        *(f"{field}_trailing3" for field in DYNAMIC_PERSONA_FIELDS),
    )
    for column in state_columns:
        if column in out:
            out[f"prior_{column}"] = out.groupby(["conv_id", "speaker"])[column].shift()
    return out

def derive_annotation_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Derive documented indices and transition labels from visible annotations."""
    out = df.sort_values(["conv_id", "turn"]).copy()

    def numeric(name: str) -> pd.Series:
        return pd.to_numeric(out[name], errors="coerce")

    if {"remaining_disagreement", "adversariality"}.issubset(out.columns):
        out["observed_conflict_index"] = (
            numeric("remaining_disagreement") + numeric("adversariality")
        ) / 8.0
    if {"local_agreement", "affiliation"}.issubset(out.columns):
        out["observed_alignment_index"] = (
            numeric("local_agreement") + numeric("affiliation")
        ) / 8.0
    if {"realized_move", "apparent_objective"}.issubset(out.columns):
        exploration_moves = {"clarify", "question"}
        exploration_objectives = {"clarify", "explore"}
        out["observed_exploration_index"] = (
            out["realized_move"].isin(exploration_moves).astype(float)
            + out["apparent_objective"].isin(exploration_objectives).astype(float)
        ) / 2.0
    closure_inputs = {
        "explicit_synthesis", "explicit_resolution", "explicit_closure",
        "local_agreement", "remaining_disagreement",
    }
    if closure_inputs.issubset(out.columns):
        out["closure_evidence"] = (
            numeric("explicit_synthesis")
            + numeric("explicit_resolution")
            + numeric("explicit_closure")
            + numeric("local_agreement") / 4.0
            + (4.0 - numeric("remaining_disagreement")) / 4.0
        ) / 5.0

    if {"observed_alignment_index", "observed_conflict_index"}.issubset(out.columns):
        alignment_change = out.groupby(
            ["conv_id", "speaker"]
        )["observed_alignment_index"].diff()
        conflict_change = out.groupby(
            ["conv_id", "speaker"]
        )["observed_conflict_index"].diff()
        out["observed_accommodation_index"] = (
            alignment_change - conflict_change
        ) / 2.0

        transition = pd.Series("stable", index=out.index, dtype=object)
        if "realized_move" in out:
            transition.loc[out["realized_move"].eq("accommodate")] = "accommodation"
            transition.loc[out["realized_move"].eq("concede")] = "de_escalation"
        transition.loc[(alignment_change > 0.15) | (conflict_change < -0.15)] = (
            "de_escalation"
        )
        transition.loc[(conflict_change > 0.15) | (alignment_change < -0.15)] = (
            "escalation"
        )
        if "explicit_synthesis" in out:
            transition.loc[numeric("explicit_synthesis").eq(1)] = "synthesis"
        if "explicit_resolution" in out:
            transition.loc[numeric("explicit_resolution").eq(1)] = "resolution"
        if "explicit_closure" in out:
            transition.loc[numeric("explicit_closure").eq(1)] = "closure"
        out["observable_transition"] = transition
    return out
