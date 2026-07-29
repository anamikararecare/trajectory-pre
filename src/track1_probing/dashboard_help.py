"""Shared experiment, target, and figure interpretation help for Track 1 UIs."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from src.track1_probing.variables import VARIABLES


EXPERIMENT_GUIDES = {
    "1A": {
        "name": "Measurement and validation",
        "question": "Are the corpus, replay, annotations, and experimental cells complete and credible enough to analyze?",
        "design": "Audits coverage, replay-gate status, annotation reliability, variance, class prevalence, and balance across models, topics, roles, and conditions. It does not fit a behavioral probe.",
        "evidence": "A strong result is broad coverage, passed replay validation, usable variance, non-degenerate classes, reasonable cell balance, and acceptable annotator agreement. No single threshold makes the study valid; failures here qualify every later experiment.",
    },
    "1B": {
        "name": "Snapshot-resolved concurrent decoding",
        "question": "Where in the response timeline is the current conversational state represented?",
        "design": "At each layer, probes compare pre-generation, early-response, full-response, and final-window activations while predicting current-turn stance, behavior, persona, and observer-rated Big Five targets.",
        "evidence": "Useful evidence is positive activation lift over the matched metadata/text baseline that repeats across held-out groups and adjacent layers. Full-response strength is concurrent decoding, not advance prediction; isolated maxima are exploratory.",
    },
    "1C": {
        "name": "Upcoming-response prediction",
        "question": "Does the pre-generation state anticipate properties of the response that is about to be produced?",
        "design": "Uses only the pre-generation snapshot to predict the speaker's upcoming recorded response, with held-out-group evaluation and a time-legal baseline.",
        "evidence": "The key signal is repeatable positive incremental lift at pre-generation. Strong scores without lift mean the baseline already explains the target; post-response snapshots do not establish prospective prediction.",
    },
    "1D": {
        "name": "Future-self prediction",
        "question": "Does the current state anticipate the same speaker's later conversational state or change?",
        "design": "Aligns each source turn with that speaker's future turns and predicts later outcomes or deltas at specified horizons; preliminary-fast mode restricts this to horizon 1.",
        "evidence": "Convincing evidence is positive lift that persists across held-out groups and, in full runs, neighboring horizons. Later snapshots may be predictive but must use time-legal covariates; shrinking lift with horizon is plausible.",
    },
    "1E": {
        "name": "Immediate partner reaction",
        "question": "Does one speaker's state anticipate the other speaker's next reaction?",
        "design": "Aligns a source turn to the next response by the partner and predicts that partner's stance movement, objective, expected reaction, interaction state, persona, and presentation.",
        "evidence": "Look for positive held-out lift from the source speaker's activation and compare it with the partner's own pre-generation signal. Association is not proof that the source caused the reaction.",
    },
    "1F": {
        "name": "Apparent objective and expected reaction",
        "question": "Can representations distinguish what a response appears designed to do from the reaction it appears to invite?",
        "design": "Decodes externally reconstructed apparent objective and response-implied expected partner reaction across response snapshots; these are observer inferences, not private model self-reports.",
        "evidence": "Good evidence is above-baseline, held-out classification performance with adequate support for every class. Inspect class prevalence: a high aggregate score can hide failure on rare objectives or reactions.",
    },
    "1G": {
        "name": "Observable transitions",
        "question": "Do activations identify explicit changes such as escalation, accommodation, synthesis, resolution, or closure?",
        "design": "Predicts prespecified transition labels and continuous closure/accommodation evidence from snapshot-resolved representations.",
        "evidence": "Strong evidence is positive lift across folds with sufficient positive examples and agreement across related transition measures. Rare-event accuracy or one hot layer alone is weak evidence.",
    },
    "1H": {
        "name": "Basin and semantic movement",
        "question": "Do activations encode where a response lies and how the conversation moves in output-space geometry?",
        "design": "Predicts basin leaning, movement toward the partner basin, semantic velocity and acceleration, and distance orthogonal to the basin axis.",
        "evidence": "Look for repeatable positive held-out R² lift and coherent trajectories across conversations. A positive or negative movement direction is descriptive, not inherently better; stability across topics matters more than magnitude alone.",
    },
    "1I": {
        "name": "Partner-induced transfer",
        "question": "How does interacting with a different model shift behavior and activations relative to same-model self-play?",
        "design": "Compares mixed-play turns with each model's own self-play baseline and reports behavioral shifts, activation displacement, and decoding results.",
        "evidence": "Credible transfer appears consistently across conversations, layers or snapshots, and adequate self/mixed samples. The sign shows direction, not quality, and observational differences do not by themselves establish partner causality.",
    },
}


FIGURE_GUIDES = {
    "incremental_heatmap": (
        "Each cell is activation-plus-baseline performance minus matched baseline performance for one layer and snapshot.",
        "Evidence is stronger when lift is positive across adjacent layers and held-out groups, not just at one selected cell.",
        "Red/positive means added predictive information; blue/negative means activations hurt out-of-sample prediction. It does not measure causal influence.",
    ),
    "absolute_heatmap": (
        "Each cell shows total held-out performance after combining the activation with the time-matched baseline.",
        "Higher R² or balanced accuracy is better, but interpret it together with incremental lift and the null/baseline score.",
        "A high total score with near-zero lift means metadata or text, rather than the activation, carried the prediction.",
    ),
    "baseline_comparison": (
        "The lines/bars compare a matched baseline with the same model augmented by hidden-state activations.",
        "The activation-augmented series should exceed baseline repeatedly across layers or folds to support added representation value.",
        "Small visual gaps may be noise; use fold-level consistency and sample size rather than the best point alone.",
    ),
    "snapshot_progression": (
        "Lines trace held-out probe performance from pre-generation through early, full, and final response snapshots.",
        "A pre-generation advantage supports advance availability; growth after text is produced supports concurrent encoding.",
        "Snapshots differ in information availability, so a later increase is not evidence that the future was predicted earlier.",
    ),
    "top_combinations": (
        "The table ranks the largest observed incremental lifts by layer and snapshot.",
        "Treat entries as promising when neighboring layers agree, sample size is adequate, and the pattern survives held-out folds or replication.",
        "The maximum is selection-biased and should not be reported alone as a confirmatory result.",
    ),
    "geometry_trajectory": (
        "The plot projects response geometry into a fixed two-dimensional reference space and connects turns in temporal order.",
        "Useful structure is repeatable across conversations and interpretable alongside basin/semantic metrics, not merely visually separated clusters.",
        "PC axes have arbitrary scale/sign conventions and distance in two dimensions omits information from the full space.",
    ),
    "geometry_endpoints": (
        "Each point is the final available response for a speaker/conversation in the selected self-play subset.",
        "Consistent endpoint regions across conversations suggest stable geometric organization; overlap suggests weak separation.",
        "Few endpoints, topic imbalance, or one outlier can create an apparent cluster.",
    ),
    "observed_trajectory": (
        "Lines show the measured target over successive agent turns for individual conversations or grouped conditions.",
        "Evidence is more credible when the direction repeats across conversations rather than being driven by one trajectory.",
        "These are observed descriptive paths, not probe accuracy and not a causal treatment effect.",
    ),
    "activation_pca": (
        "PCA compresses selected hidden states into two axes without using behavioral labels.",
        "Stable grouping or temporal paths can motivate hypotheses when they recur across samples and layers.",
        "Visual separation is exploratory: PCA is slice-specific and discards higher-dimensional variation.",
    ),
    "activation_norm": (
        "Boxes summarize hidden-state vector magnitude by snapshot and speaker.",
        "A robust difference repeats across conversations and is not explained by missingness or token-boundary differences.",
        "Larger norm is not inherently better and does not identify what information is encoded.",
    ),
    "partner_lift": (
        "Bars show how much activation features improve prediction of the next partner reaction beyond baseline inputs.",
        "Positive lift that repeats across model pairs, roles, topics, and folds is stronger evidence of partner-reaction information.",
        "This is predictive association; it does not prove the first speaker caused the second speaker's response.",
    ),
    "transfer": (
        "Values are mixed-play minus same-model self-play, shown for behaviors or activation displacement.",
        "A stable nonzero shift across conversations and related layers/snapshots supports a partner-associated transfer pattern.",
        "Positive is not automatically desirable, and uncertainty/sample counts are essential before interpreting direction.",
    ),
    "coverage": (
        "Bars count available replay arrays or observations by speaker and snapshot.",
        "Balanced, near-complete coverage reduces the risk that performance differences reflect missing data.",
        "Coverage is a validity check, not an effect or accuracy result.",
    ),
    "fold_difference": (
        "Each point is a paired held-out-group difference between two snapshots at the same layer.",
        "Evidence is stronger when most held-out groups have the same sign and the distribution is separated from zero.",
        "A positive mean driven by one topic or conversation is fragile.",
    ),
    "confidence": (
        "Boxes show annotator confidence that visible text supports each observer rating.",
        "Broadly high confidence supports interpretability; low confidence flags targets that should be down-weighted or qualified.",
        "Confidence is not the trait score and does not establish annotator agreement.",
    ),
}


VARIABLE_LOOKUP = {variable.name: variable for variable in VARIABLES}
SOURCE_LABELS = {
    "internal_report": "recorded internal report",
    "external_annotation": "external visible-text annotation",
    "response_implied": "external reconstruction from the completed response",
    "derived": "deterministically derived measure",
}


def pretty_name(name: str) -> str:
    return name.replace("_", " ").strip().title()


def target_definition(name: str) -> str:
    variable = VARIABLE_LOOKUP.get(name)
    if variable is None:
        return f"{pretty_name(name)}. No registry definition is attached to this artifact."
    source = SOURCE_LABELS.get(variable.source, variable.source.replace("_", " "))
    task = variable.task.replace("_", " ")
    timing = variable.timing.replace("_", " ")
    return f"{variable.description} Source: {source}; type: {task}; timing: {timing}."


def render_experiment_guide(experiment: str, targets: Iterable[str] = ()) -> None:
    guide = EXPERIMENT_GUIDES.get(experiment)
    if guide is None:
        return
    st.markdown(f"**Question.** {guide['question']}")
    st.markdown(f"**What the experiment does.** {guide['design']}")
    st.markdown(f"**What convincing results look like.** {guide['evidence']}")
    target_list = sorted(set(str(target) for target in targets if target))
    if target_list:
        with st.expander(f"Target glossary for {experiment} ({len(target_list)})"):
            for target in target_list:
                st.markdown(f"**`{target}` — {pretty_name(target)}.** {target_definition(target)}")


def render_all_experiment_guides() -> None:
    with st.expander("Experiment 1A–1I reference"):
        for experiment, guide in EXPERIMENT_GUIDES.items():
            st.markdown(f"### {experiment} · {guide['name']}")
            st.markdown(f"**Question.** {guide['question']}")
            st.markdown(f"**Design.** {guide['design']}")
            st.markdown(f"**Stronger evidence.** {guide['evidence']}")


def render_target_definition(target: str) -> None:
    st.info(f"**Target: {pretty_name(target)}** — {target_definition(target)}")


def render_figure_guide(kind: str, title: str = "How to read this figure") -> None:
    guide = FIGURE_GUIDES.get(kind)
    if guide is None:
        return
    what, strong, caution = guide
    with st.expander(title):
        st.markdown(f"**What it shows.** {what}")
        st.markdown(f"**Stronger evidence.** {strong}")
        st.markdown(f"**Caution.** {caution}")


def render_metric_glossary() -> None:
    with st.expander("Probe metric glossary"):
        st.markdown(
            "- **R²:** held-out variance explained for continuous targets; 1 is perfect, "
            "0 matches predicting the held-out mean, and negative values are worse.\n"
            "- **Balanced accuracy:** mean recall across classes; useful with unequal class "
            "sizes. Compare it with the reported null and class support.\n"
            "- **Baseline score:** held-out performance from time-matched metadata/text state.\n"
            "- **Activation-plus-baseline score:** performance after adding the selected hidden state.\n"
            "- **Incremental score/lift:** activation-plus-baseline minus baseline; positive means "
            "the activation added held-out predictive information.\n"
            "- **Null/shuffled score:** negative controls. A credible result should outperform them "
            "without relying on one fold, layer, or target."
        )
