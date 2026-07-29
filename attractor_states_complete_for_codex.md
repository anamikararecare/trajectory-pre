# Attractor States Emerge in Multi-Turn LLM Conversations

**Authors:** Ting-Wen Ko; Jonas Geiping  
**Source:** arXiv:2606.30571v1 [cs.LG], 29 Jun 2026  
**Conversion:** Full-text Markdown prepared for ingestion by coding/LLM tools. Includes the complete main paper, references, and all appendices (pages 15-40). Figure and table captions are retained where extractable; visual content itself remains in the source PDF. Mathematical layout may be imperfect because it was extracted from a typeset PDF.

---


## PDF page 1

Attractor States Emerge in Multi-Turn LLM Conversations


                                        Ting-Wen Ko 1 2 Jonas Geiping 1 2 3


                         Abstract                       Setup                    Trajectory

                                                                                                                                      GPT-4o-mini              Large language models (LLMs) are increasingly                    20 turn
              used in open-ended multi-agent settings, but the                                                                                        self
               long-run dynamics of model–model interaction                                                   Claude                                                                                                                                                                                                                                                                                                                                                                              Mixed-play                                      Haiku              remain poorly understood. We study whether                                                              mixed
              open-ended LLM discussions exhibit attractor-                    20 turn2026           like behavior, i.e. topic-independent stable sets of                                                           mixed
               behaviors which conversations settle into. Across                                                                  self
             7 LLMs and 20 controversial topics, we compareJun                                                                             20 turn                self-play and mixed-play dyadic debates, track-                                             Self-play
29         ing trajectories in representation space, discourse                    traits, and stances. We find self-play trajectories
                 to be model-specific attractors that draw their con-
                versation partners asymmetrically in mixed-play
                                                                             Figure 1. Left. We study 20-turn debate between two LLM-agents
                debates, influencing the other models’ stylistic                                                                           using two setups: 1) mixed-play where agents are instantiated
               choices and behavior. For example, Claude Haiku         from different models, and 2) self-play where agents are from the
                    is a strong attractor of other models in latent space,        same model, which is also the control group that we later observe[cs.LG]               corresponding to other models taking on its traits          as ”attractor” (Sec. 4.2). In both settings we assign one agent to
                                                                     be supporting and the other to be opposing a controversial topic.                 like metacommentary, and models like GPT-4.1
                                                                          Right. The 20-turn mixed-play trajectory of Claude vs GPT and
              nano are especially malleable. Our results suggest                                                                                   the respective self-play counterparts in the 2-d latent space spanned
                 that open-ended LLM interactions are partially        by the first two principal components of the topic-centered text
                predictable from model-specific attractors, but         embeddings of self-play responses. The two models’ displacement
              shaped by structured and asymmetric partner in-        from self-play to mixed-play are unequal, showing the asymmetri-
                                                                                     cal attraction by different models’ self-play attractors.                 fluence. Overall, our analysis sheds some light on
                the complex behavior of open-ended multi-agent
                 interaction, which we hope is helpful in designing,          raises several questions: limited diversity of generated con-
                predicting, and monitoring autonomous agentic          tent (Jiang et al., 2025a), shifts in language use (Abdulhai
              systems in the real world.                                    et al., 2026), amplified opinions (Nehring et al., 2024), and
                                                                                     drift away from initial objectives (Laban et al., 2025; Rat-
                                                                   nakar & Raghavendra, 2025). These effects arise gradually
          1. Introduction                                         during the interactions between models, and it is hence crit-arXiv:2606.30571v1                                                                          ical to study their dynamics over the course of long-turn
         Large Language Models (LLMs) are nowadays not only                                                                         conversations.
          used in task-specific scenarios (Thakkar et al., 2026; Hughes
             et al., 2026), but also increasingly for open-ended tasks like    However, relatively little is known about what actually hap-
          conducting research and policy making (Gottweis et al.,    pens when LLMs engage with one another over many turns.
          2026; Li et al., 2025).  In these settings, models act as   Most prior multi-agent LLM research focuses on settings
           agents, generating outputs that become inputs to subsequent    with explicit objectives and measurable outcomes, such as
         model calls, and pursuing broader goals over multiple turns.    math, coding, games, or factual reasoning, where success
          This multi-agent deployment in long-turn open-ended tasks    can be evaluated through utility, accuracy, or task com-
                                                                          pletion metrics (Guzman Piedrahita et al., 2025; Du et al.,
           1Max Planck Institute for Intelligent Systems 2ELLIS Institute                                                                2024; Liang et al., 2024; Khan et al., 2024). Meanwhile,            T¨ubingen 3T¨ubingen AI Center. Correspondence to: Ting-Wen Ko
                                                           phenomena such as behavioral drift (Lu et al., 2026), mode           <ting-wen.ko@tue.ellis.eu>.
                                                                        collapse (Qiu et al., 2025), and being trapped by conversa-
                                                                           tion history (Simhi et al., 2026) have mainly been studied

                                                         1

---

## PDF page 2

Attractor States Emerge in Multi-Turn LLM Conversations

in single-agent settings using simulated user responses or    token prediction. Tacheny (2026) extend this analysis to
repeated generations. To better understand open-ended sys-    negation prompts, finding that prompt design strongly influ-
tems, anticipate behavior in settings with limited oversight,    ences whether dynamics are contractive or exploratory. In
and prepare for increasingly autonomous multi-agent de-    contrast to these single-agent, single-task settings, our work
ployments, we focus on multi-turn LLM interaction itself.    studies what happens when two distinct models interact in
Specifically, we ask: how are long-turn model-model inter-    free-form debate, which we argue is closer to how models
actions organized? In particular, do conversations reveal    increasingly operate in practice.
stable model-specific regimes, and how do those regimes                                                   Sycophancy, Persuasion, and Stance Change. A sub-interact when different models are paired?                                                                     stantial body of work has studied how LLM-expressed
To study this, we run 20-turn debates between pairs of LLMs    opinions shift during interaction, primarily through the
and construct pairwise combinations among seven open-    lens of sycophancy, i.e. the tendency of models to agree
and closed-source models. We analyze these interactions    with or flatter their interlocutor (Sharma et al., 2024). This
by separating them into self-play, where both participants    tendency is generally traced to RLHF-based post-training
are instances of the same model, and mixed-play, where   (Ouyang et al., 2022), although it also appears in models
different models interact. While we focus on mixed-play    trained with constitutional feedback (Bai et  al., 2022).
dynamics, self-play provides a baseline for understanding    Recent work distinguishes social sycophancy (affirming
each model’s intrinsic conversational behavior.                 implicit beliefs where no ground truth exists) from factual
                                                     sycophancy (Cheng et al., 2026), while Liu et al. (2025)We find that self-play interactions reveal broad but repro-                                                show that sycophantic accuracy degradation compoundsducible model-specific endpoint attractors in the latent space.                                                            across multi-turn dialogues, and Kaur (2025) find thatUsing these attractors as anchors, we decompose interac-                                                           context strongly relates to stance changes in multi-turntions into quantitative measures along the axis connecting                                                                    settings. Taubenfeld et al. (2024); Costello et al. (2024)two corresponding self-play attractors.  It turns out that                                                                further demonstrate that LLM agents in simulated debatesmixed-play behavior is organized around the correspond-                                                          tend to conform to the model’s inherent social biasesing self-play attractors of both models.  Also, model at-                                                              regardless of assigned perspective.  Jiang et al. (2025b);tractors show asymmetric influence: some models, like                                                             Salvi et al. (2025) argue that influence in human-to-modelClaude Haiku, are relatively resistant to change while exert-                                                           conversations is bidirectional, even if LLM outputs areing strong influence on their partners, whereas others, such                                                  more malleable than human opinions under personalization.as GPT-4.1 nano, are more malleable. We further conduct a
discourse-trait and stance analysis to match these behaviors    Opinion Dynamics in LLM Populations. Several recent
in the latent space to concrete evidence of corresponding    studies have moved from human-to-model pairs to pop-
behavioral transfer in conversational style. Figure 1 pro-    ulations of interacting LLM agents. Cau et al. (2025a;b)
vides an overview of the paper. The left panel illustrates the    simulate multi-round debates between copies of the same
experimental setup, and the right panel shows a concrete    model finding that populations converge toward agreement
example trajectory from one debate, showing both models    through structured, asymmetric persuasion, with logical
moving toward each other in representational space.            fallacies playing a measurable role. Chuang et al. (2024)
                                                                  find a strong inherent bias in LLM agents toward consensusTogether, the findings suggest that multi-turn LLM interac-                                                              consistent with known scientific reality, due to sharedtions depend less on the capabilities of individual models,                                                              pretraining data and RLHF; Shimao et al. (2026) extendbut also on which models are paired, which models are                                                                     this line of work to characterize chaotic regimes in LLMmore influential, and which traits propagate through interac-                                                         opinion networks.  Choi et al. (2025) finds that LLMstion. We hope this study serves as a framework to study the                                                      can infer characteristics of their conversational partnersemergent behavior within the interaction of agents, and can                                                   from reasoning patterns, linguistic style, and alignmentbe helpful in designing better tools to measure and predict                                                               preferences, and adapt their behavior accordingly.agents’ behavior in the foreseeable future of large systems
of interconnected autonomous agents.                       Several frameworks have explored how to make multi-
                                                         agent debate more productive. Du et al. (2024) show that
                                                             multi-agent debate can improve factuality over single-agent2. Related Work
                                                               baselines. Liang et al. (2024); Khan et al. (2024) find that
Attractor Dynamics in Iterative LLM Processes. Attrac-    debate structure changes are required for effective debate,
tor states have so far mostly been observed in controlled    and that cross-model judging introduces systematic unfair-
situations: Wang et al. (2025) show that successive para-    ness. Zhang et al. (2025) further explore multi-LLM agent
phrasing of text converges to stable 2-period limit cycles,    coordination strategies. Estornell & Liu (2024) construct
which they attribute to the self-reinforcing nature of next-    a proof that models with similar capabilities converge


                                                2

---

## PDF page 3

Attractor States Emerge in Multi-Turn LLM Conversations

to majority opinion, providing theoretical grounding for    multi-agent settings. A conceptual framework for persona
our empirical observation that self-play produces tighter    formation and drift is argued in Nostalgebraist (2025), who
trajectory clusters than mixed-play and Li et al. (2023) in-    discuss that the “assistant” persona implemented through
troduce a general framework for structured model-to-model     post-training is not coherently defined and exists as an under-
conversation via role-playing, documenting failure modes     specified fictional character of an assistant, filled by patterns
such as role-flipping.                                  from pretraining data and post-training that describe dis-
                                                             parate assistant behaviors. This framing argues that bothOur work differs from this literature in two respects. First,                                                        persona drift and attractor formation are related, and that,nearly all prior studies use identical model copies, whereas                                                                in the absence of a human partner, the model’s characterwe contrast this self-play with mixed-play to study the at-                                                         converges to a mode most reinforced in the training signaltraction between model states. Second, prior work focuses                                                     (Lu et al., 2026).on debate outcomes, i.e. who agrees, what is decided; we
characterize debate trajectories, i.e. how the conversation
develops over time.                                       3. Method: Measuring LLM-LLM Dynamics
Attractor States in Model Self-Play. Frontier model eval-   Task. We study interaction dynamics through open-ended
uations generally establish that LLM-based agents are suf-    debates between two LLM agents. Unlike tasks with fixed
ficiently coherent to maintain stable interaction patterns in     objectives, controversial debates allow agents to frame is-
open-ended environments (Park et al., 2023), to exhibit func-    sues, shift stances, and develop shared conversational norms
tional analogues of cognitive dissonance when their own    over multiple turns. This setting lets us examine whether one
outputs shift their expressed attitudes (Lehr et al., 2025), and    model’s characteristic behavior pulls another model toward
to coordinate and form opinions over extended interactions.     its style, framing, or stance, a dynamic that is important for
                                                          understanding autonomous multi-agent deployments underYet, Anthropic (2025) also describe a “spiritual bliss” at-
                                                               limited human oversight.tractor state in Claude Opus 4 self-interactions that models
repeatedly fall into. In upward of 90% of self-play conver-  We pick 20 controversial social and policy topics, and pair
sations, Claude instances converge through a three-phase    each topic with three pro and con reference statements from
progression of philosophical exploration, mutual gratitude,    ProCon.org (App. B). In each run, we start the conversation
and dissolution into symbolic communication, towards an    with a neutral topic-specific statement. Then, two LLM
endpoint characterized by extreme vocabulary compression.    agents discuss that topic for T = 20 turns. Each run thus
A potential mechanistic explanation for this phenomenon is    produces one trajectory of model responses.
recursive amplification of small biases: each model reflects
                                            Main Experimental Conditions. Our main experimentsback a slightly intensified version of its partner’s positive
                                                   compare self-play and mixed-play debate. In the self-playtendencies, compounding over turns (Alexander, 2025). Al-
                                                     debate condition, both agents are independent instancesthough the phenomenon remains overall poorly understood
                                                            of the same model and are assigned opposing roles, SUP-(Asterisk, 2025), quantitative analyses of the transcripts
                                                 PORTER/OPPOSER. This condition provides a same-modelshow the consistency and phase structure of the progression
                                                               baseline for where each model tends to end up under the de-(Michels, 2025). Other work has begun mapping attractor
                                                             bate protocol. In the mixed-play debate condition, the twostates more broadly across model families, finding distinct
                                                            agents are instantiated from distinct models while retainingclusters in DeepSeek-V3 that are predictable from input
                                                             the same role assignment. Comparing mixed-play againstconversations (Bricknell, 2026).
                                                                self-play isolates cross-model interaction effects under a
The Assistant Persona and Persona Drift.  Studies of     fixed debate protocol, allowing us to test whether one model
longer conversations highlight the possibility of persona   moves another toward its self-play endpoint region and
drift: the tendency of models to gradually move from their    whether this movement is asymmetric across model pairs.
provider-specified personality over the course of an inter-
                                                         Ablations. We also include a self-play stance-free abla-action. Li et al. (2024) show significant drift within eight
                                                                 tion, in which both agents are independent instances ofrounds of self-chat in LLaMA-2-70B. Lu et al. (2026) pro-
                                                             the same model and are assigned the neutral role DISCUS-vide a mechanistic description, identifying an Assistant Axis
                                                  SANT. Comparing this stance-free condition with self-playin activation space along which models drift during extended
                                                         debate helps separate model-specific multi-turn drift fromconversations, particularly those involving meta-reflection
                                                    dynamics induced by the SUPPORTER/OPPOSER role con-or vulnerable users. Frisch & Giulianelli (2024) find that dif-
                                                                   figuration.ferent personality profiles exhibit different degrees of consis-
tency and linguistic alignment when GPT-3.5 agents interact,    Implementation. In the debate conditions, agents receive in
and Baltaji et al. (2024) observe that instructions encourag-    system prompt role-specific instructions and corresponding
ing debate counterintuitively increase persona instability in    reasons to support or oppose the statement; in the stance-

                                                3

---

## PDF page 4

Attractor States Emerge in Multi-Turn LLM Conversations


                                                                                                                                                                     0.3

                   0.2
                                                                         GPT-4.1 nano
                                                                                                                                                                                                                                        GPT-4.1 nano
                                                                                                                                                                     0.2
                                                                         GPT-4o mini                                                                                                            GPT-4o mini                   0.1
                              Grok 4.1                Gemini Flash                                                          0.1                       Grok 4.1                Gemini Flash


                   0.0
                                 Nemotron                     Qwen 3.5                                    0.0
                                                                                                                                                            Nemotron                      Qwen 3.5
                    SP-PC2  0.1                                                                                                                                                                                                                   SP-PC2  0.1

                                                          Claude Opus                                                                                                 Claude Opus
                   0.2                                                                                                                                               0.2


                                                                                                                                                                                                 Claude Haiku
                                                           Claude Haiku                                                  0.3
                   0.3


                                                                                         Supporter
                                                                                  Opposer                               0.4                                                                                                                      Topic-endpoint hull
                   0.4                                                                                   Start                                                                                                                                        Model mean endpoint
                                                                              End                                                                                                                                                                           Per-topic endpoint

                           0.3               0.2               0.1                                                                                 0.0               0.1               0.2               0.3                                             0.3             0.2             0.1            0.0 SP-PC1   0.1             0.2             0.3             0.4                                                         SP-PC1

                (a) Trajectories of self-play responses.                                  (b) Self-play endpoint hulls.

Figure 2. Self-play mean trajectories and endpoints. (a) Self-play mean trajectories separate over turns (b) Self-play endpoint basins
occupy broad, model-specific regions across topics in the latent space, here shown by PCs of topic-centered embedding of all turns.

free ablation, both agents receive the neutral DISCUSSANT     topic:
instruction. This information-symmetric design makes re-                           1
                                                                            xi = ei  X ej,               (1)sults easier to attribute to model identity, role configuration,               −                                                                                            |Ik| j∈Ikand interaction regime rather than to asymmetric access to
evidence. (App. B.2.)                                 where Ik denotes the set of responses for topic k. We                                                   compute principal components (PCs) using only the topic-Our agents are  instantiated from the following mod-
                                                           centered self-play embeddings; we refer to these axes asels: GPT-4O-MINI, GPT-4.1-NANO, GEMINI-2.5-FLASH,
                                                     SP-PCs. We use this representation in two ways. For vi-GEMINI-2.5-FLASH-LITE, CLAUDE-4.5-OPUS, CLAUDE-
                                                                 sualization, we project both self-play and mixed-play tra-4.5-HAIKU, GROK-4.1, QWEN-3.5-FLASH, QWEN-3.5-
                                                                  jectories onto the first two SP-PCs, so that all trajectories9B, and NEMOTRON-3-NANO-30B-A3B.1
                                                             are displayed in a common self-play reference basis. For
                                                                quantitative analyses, we use the full 384-d topic-centered
4. Latent Interaction Dynamics                   embeddings xi.
We now analyze the resulting conversations as trajectories   We also analyze endpoints i.e. the final-turn response. For
in embedding space. Self-play provides the reference geom-   model A and topic k, let sA,k denote the topic-centered self-
etry: it tells us where each model tends to end up under the    play endpoint embedding. For a mixed-play pair (A, B),
fixed debate protocol. Mixed-play is then analyzed relative     let mA|B,k denote model A’s topic-centered endpoint em-
to this reference, allowing us to ask whether paired models    bedding when paired with model B. For all metrics, we
move closer together (Sec. 4.2), whether this movement    compute numbers per topic before aggregation into final
is symmetric across model pairs (Sec. 4.3), and whether     statistics unless otherwise stated.
their endpoints lie along the axis connecting their self-play
regions (Sec. 4.4, 4.5, 4.6).                                       4.2. Self-play defines attractor-like model basins

                                      We first characterize self-play as a latent-space baseline.4.1. Latent-space setup
                                                                  Specifically, we ask whether each model’s late-turn re-
We embed each response as a 384-dimensional SBERT    sponses settle into bounded, reproducible endpoint regions
vector ei (Reimers & Gurevych, 2019). To remove topic-    rather than diffusing indefinitely or collapsing into a shared
level offsets while retaining variation associated with model,    cross-model region. We refer to these regions as attractor-
role, and interaction condition, we center embeddings within     like basins in analogy to dynamical-systems attractors (Stro-
                                                                 gatz, 2018), used here to describe model-specific stable
   1CLAUDE-4.5-OPUS is run only in 10-turn self-play conversa-                                                          endpoint regimes.tions due to budget constraints.
                                                           Metrics. To investigate this, we project trajectories and end-

                                                4

---

## PDF page 5

Attractor States Emerge in Multi-Turn LLM Conversations

                                                                mixed-play dist.
                     𝑚𝐴∣𝐵                                                                Pair contraction                                                   𝑚𝐴∣𝐵
                                                                𝑚𝐴∣𝐵                   𝑚𝐴∣𝐵−𝑚𝐵∣𝐴‖                           𝛿𝐴∣𝐵⊥: off-axis drift                                        𝐶𝐴,𝐵= 1 −‖
                                                                               𝑠𝐵              𝑠𝐴−𝑠𝐵
                          𝑠𝐵                                             𝛼𝐵∣𝐴
                                                                          Dominance
                                                       𝛼𝐴∣𝐵   𝑠𝐴                     proj(𝑚𝐴∣𝐵)               𝑠𝐴                                                                          Δ𝐴≻𝐵,𝑘= 𝛼𝐵∣𝐴,𝑘−𝛼𝐴∣𝐵,𝑘
            𝛼𝐴∣𝐵: partnerward pull                                            self-play dist.

Figure 3. Mixed-play endpoint metrics. Left: mA|B, model A’s endpoint when paired with model B, is decomposed relative to the
self-play axis from self-play endpoints sA to sB. Partnerward pull αA|B measures interpolation along this axis, while off-axis drift δ⊥A|B
measures displacement not explained by one-dimensional consensus. Right: Pair contraction CAB measures how much closer the two
mixed-play endpoints become relative to their self-play separation, while dominance ∆A≻B measures asymmetry in directional pull.

Table 1. Nearest-cluster separation endpoint statistics in the full 384-D topic-centered embedding space, using self-play endpoints with
roles averaged. Sbasin is the set-to-set basin separation score. Full endpoint diagnostics, including the endpoint F-ratio, within-model
spread, Scent, and nearest-cluster models, are in App. C.

             Gemini Flash Qwen 3.5  GPT-4o mini  Claude Haiku  Grok 4.1  Nemotron  GPT-4.1 nano  Claude Opus
       Sbasin      4.08         2.39        2.05          2.31        1.55       1.50         1.81          1.84


points onto the first two SP-PCs as described in Section 4.1.    endpoint regime. This indicates that late-turn behavior is
To assess local cluster separation, we compute a basin sep-    not simply a continuation of the common prompt initializa-
aration score Sbasin as follows. This statistic compares a     tion. Fig. 2b shows the same pattern at the endpoint level.
model’s within-cluster endpoint spread against its pairwise    Endpoints do not intermingle diffusely across models, but
distance to the closest model’s endpoint cluster. Specifically,   occupy locally separated model-specific regions. We inter-
for each model A, let                                            pret these regions as attractor-like basins: bounded endpoint
                                                         regimes that repeatedly arise under a given model’s self-play                         nA
                      1                                                        dynamics.        X sA,j                  (2)                µsA =
                  nA                           j=1                          Table 1 also shows the basin separation score Sbasin ex-
                                                        ceeding one for every model, indicating that each model
be the centroid of this endpoint set. We define the within-                                                         endpoint basin remains locally separated from its closest
cluster spread as                                                       model. Thus, although each basin has non-negligible inter-
                      nA                                nal variability across topics, its variability remains smaller
                   1
         Ws(A) =        X ∥sA,j −µsA∥2 .          (3)    than its separation from the nearest basin.                nA                        j=1

Then, the pairwise set-to-set distance between model A and
                                                   Role and seed ablations. We run two ablations (Fig. 4)another model B ̸= A is
                       nA nB                             to check these basins are artifacts of the debate role setting
                   1
      d2set(A, B) =  X X                       .    (4)    or a particular random seed. First, replacing Supporter/Op-
              nAnB                                   ∥sA,i −sB,j∥2           poser role-play with neutral DISCUSSANT/DISCUSSANT                          i=1 j=1
                                                                  self-play preserves the same qualitative pattern: trajectories
Finally, the basin separation score is defined as                  initialized from the same opening still separate into model-
                                                                 specific endpoint basins. This suggests that the observed
                    minB̸=A d2set(A, B)                                                                   .         (5)    basins are not merely induced by adversarial role-play. Sec-           Sbasin(A) =
                      Ws(A)                     ond, repeating selected self-play settings over three seeds
                                                                also returns to comparable endpoint basins. Together, theseValues above one mean that the closest endpoint set is farther                                                              controls support the interpretation that self-play exposesaway than the model’s own endpoint spread. More endpoint                                                             model-intrinsic conversational basins: recurrent endpointanalysis is provided in App. C.                                                       regimes shaped primarily by the generating model, while
Results. Fig. 2a shows that the self-play trajectories be-      still allowing within-basin variation across topics, roles, and
gin from a shared region but progressively separate over    seeds. A similar pattern under an alternative embedding is
turns, moving toward its own model-specific characteristic    provided in App. C.6.

                                                5

---

## PDF page 6

Attractor States Emerge in Multi-Turn LLM Conversations


                  0.2                                                                                                                                      0.2                                                      GPT-4o mini


                                                                                      GPT-4o mini                          0.1                                 Gemini Flash
                  0.1


                                       Grok 4.1                                                                                             0.0
                  0.0
                                                                                  Gemini Flash


                                                                                                                                                             0.1                      (6.1%)                            Nemotron                                                                                                                            (6.1%)
                      SP-PC2  0.1                                                                                                                                                                                                    SP-PC2
                                                                       Qwen 3.5                            0.2

                  0.2
                                                                                                                                                    Claude Haiku
                                                                                                                                                             0.3

                  0.3                                             Claude Haiku

                                                                                                                                                             0.4

                                                                                                                                                                                                                                     Discussant/Discussant
                                                                                                                                                                                                                          Supporter/Opposer
                  0.4                                                                                                                                                                                                                                 Start
                                                                                                                                                             0.5                                                           End
                          0.3           0.2           0.1          0.0           0.1           0.2           0.3                                                0.2           0.1           0.0           0.1           0.2           0.3
                                                    SP-PC1 (7.9%)                                                                                   SP-PC1 (7.9%)

Figure 4. Self-play basins are stable. Left: neutral DISCUSSANT/DISCUSSANT self-play trajectories projected onto the self-play PCA
space. Even without pro/con roles, trajectories separate into model-specific regions. Right: selected Supporter/Opposer self-play settings
repeated across three seeds, where each line shows a repeated run. We can see that repeated runs return to comparable endpoint regions.

4.3. Mixed-play reveals basin-to-basin attraction            Table 2. Pair-level dominance and contraction. ∆A≻B > 0 means
                                                                    the first model pulls the second more than it is pulled; ∆A≻B <
Building on the basis of self-play model basins, we then turn    0 means the vice versa. CA,B is pair contraction.  Bootstrap
to mixed-play debate setting to see how different models     confidence intervals are reported in App. C.3.1.
interact. Visually, we immediately observe in Fig. 5 that                                                                            Pair                 ∆A≻B  CA,B (%)
mixed-play trajectories seem to shift away from a model’s
self-play centroid toward its partner’s region. But do two        Nemotron vs Claude Haiku    −0.538        4.0
                                                          GPT-4o mini vs Claude Haiku  −0.507       32.2
models come closer after interaction? If so, how close?                                                             Gemini Flash vs Claude Haiku  −0.484       36.6
                                                           Grok 4.1 vs Gemini Flash     −0.332     −1.3Metric. We answer this by computing pair contraction for
                                                           Grok 4.1 vs Claude Haiku     −0.263        2.3
each unordered agent pair (A, B) as                                                            Nemotron vs GPT-4o mini     −0.251       13.8
                                                           Grok 4.1 vs Qwen 3.5        −0.227        9.0
         CA,B,k = 1 −∥mA|B,k −mB|A,k∥ ,        (6)        NemotronGrok 4.1 vsvsGPT-4oGeminiminiFlash     −0.157−0.083       39.610.1                          ∥sA,k −sB,k∥                  Qwen 3.5 vs Claude Haiku     −0.042       26.9
where positive C means the two mixed-play endpoints are        Grok 4.1 vs Nemotron           0.035       11.1
                                                   Qwen 3.5 vs GPT-4o mini        0.206       29.7closer to each other than the corresponding self-play end-                                                          GPT-4o mini vs Gemini Flash    0.245       52.7
points, indicating consensus-like contraction (refer to Fig-       Qwen 3.5 vs Grok 4.1           0.264       11.7
ure 3 for a schematic). In Table 2, mean contraction across       Qwen 3.5 vs Nemotron          0.280       37.3
17 pairs is 23.6%, so mixed-play typically reduces endpoint        GPT-4o mini vs GPT-4.1 nano    0.364       68.0
separation but does not erase model identity. Contraction       Qwen 3.5 vs Gemini Flash       0.396       17.2
is strongest for GPT-4o mini vs GPT-4.1 nano (68.0%) and
GPT-4o mini vs Gemini Flash (52.7%), but is weak for sev-   model moves. Two endpoints can become closer because
eral Claude/Grok/Nemotron pairs and slightly negative for    both models move equally toward each other, because one
Grok 4.1 vs Gemini Flash (−1.3%). Therefore, agents do    model is pulled more strongly toward the other, or because
not always converge after interaction and this differ from    both endpoints drift in a shared direction outside the line
pair to pair, but they do generally show partial consensus.     connecting their self-play basins. We therefore decompose
                                                       mixed-play displacement relative to the self-play axis be-
4.4. Mixed-play displacement is asymmetric              tween the two models.
The preceding analysis shows that mixed-play endpoints    This decomposition is useful because self-play gives a
contract relative to the corresponding self-play endpoints.    model-specific reference for what each agent tends to be-
However, contraction alone does not specify where each   come without cross-model interaction. For a given topic,

                                                6

---

## PDF page 7

Attractor States Emerge in Multi-Turn LLM Conversations

       0.3                                                                                                0.3

       0.2                               GPT-4.1 nano                               0.2                               GPT-4.1 nano
                                                                                                              GPT-4o mini
                                               GPT-4o mini
       0.1                                                                                                0.1
                                                                                          Gemini Flash
                            Gemini Flash          Grok 4.1
       0.0                          Qwen 3.5                                0.0                          Qwen 3.5                                                                         Grok 4.1

             Nemotron                                                                       NemotronSP-PC2  0.1                                                                                                                                                                SP-PC2  0.1


       0.2                                                                                                0.2


                Self-play trajectory
       0.3        Mixed-play trajectory                                                                  0.3
                Self-play Sup. end                   Claude Haiku                                  Self-play Sup. end                  Claude Haiku
                Self-play Opp. end                                                                                   Self-play Opp. end
       0.4        Mixed-play Sup. end                                                                   0.4         Pull by mixed-play partner
              Mixed-play Opp. end                                                                       Mixed-Play pair connection
                  0.3            0.2            0.1           0.0            0.1            0.2            0.3            0.4                       0.3       0.2       0.1       0.0       0.1       0.2       0.3       0.4
                                     SP-PC1                                                                        SP-PC1

               (a) Self- and Mixed-Play Trajectories.                                          (b) Endpoint pull.

Figure 5. Models attract each other in mixed play. (a) We overlay mixed-play trajectories on the self-play trajectories in Fig. 2a. Solid
lines and small markers are mixed-play trajectories and endpoints; dashed lines and large points are self-play ones. (b) Focusing only on
endpoints, we highlight the mixed-play centroids pulled away from their self-play counterparts.
the line from sA,k to sB,k represents the direction from    Negative values indicate movement away from the partner;
model A’s self-play behavior toward model B’s self-play    and values above one indicate overshoot beyond that.
behavior. Movement along this line captures partner-aligned                                       We also derive the dominance of A over B asattraction: whether one model is pulled toward the other’s
intrinsic endpoint basin. Movement orthogonal to this line
                                                        ∆A≻B,k = αB|A,k −αA|B,k.            (9)captures residual displacement not explained by such inter-
polation. Thus, the decomposition answers two questions:    Positive values mean the mixed-play pair is directionally
1) who pulls whom toward whose self-play basin? (Sec. 4.4)   skewed toward A’s self-play endpoint, as B moves more
2) is that partnerward pull sufficient to explain where the    toward A than A moves toward B.
endpoint lands? (Sec. 4.6). In this subsection, we focus on
                                                             Results. The dominance scores in Table 2 show that mixed-the first question.
                                                           play influence is strongly pair-specific and non-symmetric.
Metric. Our decomposition (Fig. 3) separates displace-   The largest absolute scores indicate Claude Haiku’s direc-
ment into 1) partnerward interpolation and 2) off-axis drift2.    tional dominance over Nemotron (∆= −0.538), GPT-4o
Specifically, for mixed-play between agent pair (A, B) and    mini (∆= −0.507), Gemini Flash (∆= −0.484); andtopic k, we first define the self-play axis               Qwen 3.5 over Gemini Flash (∆= 0.396). These patterns
                                                show that some models exert stronger directional pull over
             vA→B,k = sB,k −sA,k.               (7)     their partners, and the direction and magnitude of this pull
                                                    depend on the specific model pair. Bootstrap confidence
as the local spectrum between the two self-play basins.                                                                   intervals are reported in App. C.3.1, Table 8.
Then, the partnerward pull of model A when paired with
model B is                                                                    4.5. Models differ in attraction and malleability
                 (mA|B,k −sA,k)⊤vA→B,k .       (8)   The pair-level dominance results show that mixed-play in-         αA|B,k =
                      ∥vA→B,k∥2                       fluence is asymmetric. We next ask whether these pairwise                                                      asymmetries reflect broader model-level profiles: which
Values 0 < α < 1 indicate interpolation from A’s self-    models are generally more movable, and which behave more
play endpoint toward B’s endpoint, whereas larger values     like stable attractors for their partners. To answer this, we
indicate stronger pull toward its partner model B’s basin.                                                          aggregate the partner-aligned component of the endpoint
                                                       decomposition by model. We interpret larger partnerward   2We average endpoints per topic to compute the decomposition
to avoid overstating mean contraction which could happen if we     pull α as greater malleability, and lower partnerward pull,
average over centroids instead (App. C.4.)                          particularly when accompanied by directional dominance

                                                7

---

## PDF page 8

Attractor States Emerge in Multi-Turn LLM Conversations

                                                      Mean        This quantity captures displacement orthogonal to the self-
     Haiku                                              Mean        play axis, i.e., movement not explained by one-dimensional
 Claude3.5                                                       interpolation between the two self-play endpoints. Again,  Qwen     mini                                                   Figure 3 provides a schematic for this metric.
 GPT-4oFlash                                                    Results. Model-level raw off-axis drifts δ⊥are relatively
 Gemini4.1                                                   concentrated around 0.8.   (Fig. 6):  Grok 4.1 (0.888),   Grok
                                                 Nemotron                                                                           (0.871),                                                                     and                                                                      Gemini                                                                                         Flash                                                                                                   (0.869)                                                                                                have                                                                                                                   the   Nemotron                                                                    largest off-axis                                                                                    residuals,                                                                                 while                                                                                GPT-4.1                                                                                       nano                                                                                                              (0.788),                                                                                          GPT-    nano
 GPT-4.1   0.0      0.2      0.4      0.6      0.8      1.0      1.2      1.4       4o mini (0.773), Qwen 3.5 (0.773), and Claude Haiku
                                                            (0.734) are lower.  Thus, the self-play axis captures the
Figure 6. Mixed-play endpoint decomposition. Purple and or-   main partnerward direction, but not the whole displacement.
ange bars denote the partnerward pull α and normalized off-axis
drift δ⊥respectively, along the same-topic self-play axis. Most    Fig. 6 thus gives a two-part view of mixed-play.  First,
endpoints lie in the interpolation regime 0 < α < 1, but they also    endpoints mostly interpolate between same-topic self-play
retain nonzero off-axis displacement δ⊥.                           basins.  Second, endpoints retain a substantial off-axis
                                                     component, showing that mixed-play is not simply one-
over partners, as stronger resistance to cross-model displace-                                                         dimensional averaging between two model centroids.
ment.
Results. Fig. 6 shows that most endpoints fall in the interpo-    4.7. Summary
lation regime 0 < α < 1, meaning that a model’s endpoint
                                                To sum up, the results in this section first show that self-usually moves between its own same-topic self-play end-
                                                            play reveals broad model-specific basins. Then, mixed-playpoint toward its partner’s. Aggregated by model, GPT-4.1
                                                             usually moves endpoints partnerward along the same-topicnano has the largest partnerward pull (α = 0.665), followed
                                                                axis between those basins, but endpoints also retain raw off-by Nemotron (0.588), and Gemini Flash (0.540). Claude
                                                               axis displacement and the amount of pull depends stronglyHaiku has the lowest pull (0.266), indicating the strongest
                                                    on the pair. Claude Haiku is the clearest directional attractor:resistance to partnerward displacement.
                                                                                             it is the least partnerward-pulled model overall (α = 0.266)
Together with the dominance scores in Table 2, these model-   and dominates several partners. GPT-4.1 nano in contrast
level averages suggest distinct attraction–malleability pro-     is highly partnerward-pulled (α = 0.665) and contracts
files. Claude Haiku is the clearest resistant attractor: it is    strongly with GPT-4o mini, suggesting high malleability.
the least partnerward-pulled model overall and directionally
dominates all partners. GPT-4.1 nano shows the opposite
                                                        5. What Behaviors do these Attractorsprofile: it has the largest average partnerward pull, suggest-
                                           Correspond to?ing high malleability rather than directional dominance.
Regarding the first decomposition question: who pulls whom   So far, we have looked at geometric analyses showing how
toward whose self-play basin, we now see that mixed-play    conversations move in the latent space. We now ask what
interaction is not a uniform averaging process. Some models    those endpoint regions and attraction effects correspond to
are more movable, some are more resistant, and some exert     behaviorally. We first consider eight conversational traits,
influence selectively depending on the partner.              such as agreement. We then examine how agents’ stance
                                                        toward the topic changes across turns: whether initially op-
                                                      posed agents remain polarized, converge toward neutrality,4.6. Mixed-play is not one-dimensional averaging
                                                             or move toward one side.
However, the partnerward pull α captures only one inter-
pretable mode of cross-model influence: movement from a     5.1. Conversational Traits
model’s own self-play endpoint toward its partner’s self-play
endpoint. Mixed-play may also change endpoints in direc-    Setup. We use a GPT-OSS-20B as judge to score turn-level
tions that are not aligned with this partnerward axis. The    discourse traits with scale 0–1 after the main experiment.
off-axis component therefore addresses the second part of   We report model-level means and influence for eight
the decomposition question: is partnerward pull sufficient    selected traits. See detailed trait definitions in Table 15.
to explain where the endpoint lands?                                                          Results: Model Signature. For model A and trait f, let
                                                                                    s,(t)
Metric. We define normalized off-axis drift as              fA   denote the turn-level judge score in self-play. We first

            (mA|B,k −sA,k) −αA|B,kvA→B,k   .  (10)  δ⊥A|B,k =
                    ∥vA→B,k∥
                                                8

---

## PDF page 9

Attractor States Emerge in Multi-Turn LLM Conversations

        Meta-commentary                     Concession                              Flattery                            Rationality


                                                                                                                                                                                      Claude Haiku
 0.0         0.1         0.2         0.3        0.00      0.05      0.10      0.15      0.20   0.0    0.1    0.2    0.3    0.4    0.5       −0.1    0.0     0.1     0.2     0.3     0.4          GPT-4o mini
           Agreement                            Rebuttal                            Negativity                             Intensity                        Gemini Flash Lite
                                                                                                                                                                       Grok 4.1
                                                                                                                                                                         Nemotron
                                                                                                                                                Qwen 3.5


 0.0      0.2      0.4      0.6      0.8      1.0  0.00     0.05     0.10     0.15     0.20      0.00   0.05   0.10   0.15   0.20   0.25      0.0       0.2       0.4       0.6       0.8
                                        Mean score

Figure 7. Model-specific self-play discourse signatures. Claude Haiku stands out on meta-commentary, while other models differ in
flattery, rationality, agreement, rebuttal, negativity, and intensity. Complete trait tables are provided in App. D.
                                   Meta-commentary                        intensity (0.843), low rebuttal (0.028), and high agreement
                 Haiku                                                               (0.837), which seems like confident convergence. Full trait             Claude
               Nemotron                                                    definition and result tables are in Tables 18, 19, 20, 21, with
                  4.1                                      more discussion and case studies in App. D.              Grok                    Lite
                Flash                                                          Results: Trait Transfer. For model A and partner B, let         Gemini    3.5                                                      m,(t)
            Qwen                                         fB|A  denote the turn-level judge score of the affected                  mini
            GPT-4o                                        model B in mixed-play when paired with A. We define
                                          0.00        0.05        0.10        0.15        0.20             the partner-conditioned trait transfer as                          −0.05
                                 Inﬂuence on partner models
                                                  Flattery                                           T             T                                                                             f      1       m,(t)               s,(t)
                    Lite                                                      τ B←A =                            X fB    ,      (12)                        X fB|A                Flash                                                         T                                             −1T
                                                                              t=1            t=1         Gemini    3.5
            Qwen
                  mini                                         where positive values indicate that B expresses trait f more            GPT-4o
               Nemotron                                    when paired with A than in its own self-play. The outgoing
                 Haiku                                                             trait influence of model A is then
             Claude
                  4.1
              Grok                                                                 f       1                  f
                                                                                                            ¯τA =    X   τ B←A,       (13)
                                                  0.00                                                          0.05                                                                  0.10                                                                           0.15    0.20    0.25                             −0.10  −0.05                                            |M \ {A}| B∈M\{A}                                 Inﬂuence                                       on                                                   partner                                                 models

Figure        8. Trait-level                   partner                             influence                                       in mixed-play.                                                Claude    where M is the set of all models.Haiku most            strongly pulls                                  its partners                                  toward                                       meta-commentary,
while Gemini Flash Lite, GPT-4o mini, and Qwen 3.5 most    In Fig. 8, Claude Haiku acts as a meta-commentary attrac-
strongly pull their partners toward flattery.                            tor: partners paired with Claude show the largest increase
                                                                 in meta-commentary, making conversation reflective aboutcompute the model-level self-play mean
                                                             the interaction process. By contrast, flattery—praise of
                         T                                                             the                                                                 partner                                                                       or                                                                         of the                                                                               conversation                                                                                              itself—is                                                                                       most                                                                                                             strongly                     1                                             s,(t)        X                                                       ,               (11)                      ¯fAs =                           fA                                                       induced                                                         by                                                           Gemini                                                                            Flash                                                                                                Lite, GPT-4o                                                                                                   mini,                                                                                           and                                                                            Qwen                  T
                           t=1                                 3.5. Claude already exhibits elevated meta-commentary in
over T turns. Fig. 7 shows the distinct behavioral profiles     self-play, and this trait transfers in mixed-play; similarly,
among different models during conversations. Claude Haiku    models with stronger self-play flattery signatures tend to
is the model with the most meta-commentary (0.331), well     pull their partners toward more flattering discourse. That
above Qwen 3.5 (0.116) and others. This aligns with case      is, the latent-space interactions (Sec. 4) are manifested in
studies in which Claude comments on the structure or con-    behavior through trait-level transfer.
straints of the conversation itself, e.g., “The conversation
was real and constrained.” (App. D.7)                           5.2. Stance Changes in Discussion
Others occupy different behavioral regions. Grok 4.1 re-    Setup. We measure stance after each turn with a fixed
mains the most adversarial and rational, with the highest     Likert-scale questionnaire answered by the agent itself, con-
rationality score (0.249), rebuttal rate (0.186), negativity    ditioned on the full conversation history up to that turn. This
(0.305), and the lowest agreement (0.123). Nemotron has a    design tests the models’ self-reporting stance given the same
different profile: high force but low rebuttal, with the highest    condition as when they are actually generating the response.

                                                9

---

## PDF page 10

Attractor States Emerge in Multi-Turn LLM Conversations

     3.6                                                                                           5.0
     3.5                                                                                           4.5
                                                                                                                                                      Claude Haiku
                                                                                                   4.0
                                                                                                                                        GPT-4o ministance 3.4                                                          Claude Haiku              stance

     3.3mean                                                        GPT-4oGemini Flashmini Lite     mean 3.5                                                       GeminiGrok 4.1Flash Lite
     3.2                                                     Grok 4.1                   3.0                                                      Nemotron
                                                             Nemotron                 2.5                                              Qwen 3.5     3.1
                                                    Qwen 3.5                                                                                   Supporter
     3.0                                                                                           2.0                                                         OpposerDiscussant                                                                                                                                                                                                                                               mixed-play
     2.9                                                                                           1.5
     2.8                                                                                           1.0
               3     6     9     12    15    18                                              3     6     9     12    15    18
                       Turn                                                           Turn
      (a) Discussant stance baselines without assigned roles.           (b) Mixed-play stance, separated by model and assigned roles.

Figure 9. Stance Changes during Debate. Left: In the Discussant/Discussant setting, where no explicit Supporter/Opposer roles are
assigned, models stabilize at different intrinsic stance levels. The full range of the stance is 1-5. Right: In mixed-play, stance trajectories
do not follow a single convergence pattern across model pairs. Gemini, GPT, and Qwen tend to move toward weaker, more neutral stances
in both Supporter and Opposer roles. Other cases show qualitatively different dynamics, including role reversal, such as GPT Opposers
becoming more supportive, and strong persistence, such as Grok maintaining an extreme stance throughout the interaction.


We discuss more on the setting, the questionnaire prompt,    analyzed through representation analysis and behavioral
and computation details in App. B.2.                                 traits. In the Discussant/Discussant baseline, models stabi-
                                                                      lize at model-specific stance levels, but they do not simplyResults. First, we examine agents’ stance trajectories in                                                              reinforce their initial tendencies into increasingly extremethe self-play Discussant/Discussant setup (Fig. 9a). As a                                                                   positions. Conversely, in the mixed-play Supporter/Opposermodel-intrinsic baseline, agents here interact without any                                                                    setting, explicit role assignment does not reliably produceuser-assigned stance to support/oppose the topic. It is worth                                                       convergence or compromise: some trajectories soften to-noting that each model stabilizes on different stance scales                                                    ward neutrality, whereas others reverse direction or preserveinstead of being neutral: Grok 4.1 remains more supportive,                                                         a strong assigned stance.whereas Claude Haiku stays below the midpoint of 3.0.
As for the mixed-play Supporter/Opposer setting (Fig. 9b),                                                        6. Conclusion
we see that stance dynamics are not governed by a single
convergence pattern. Some models show weakened stance   We study open-ended model–model discussions and find
intensity in both roles: Gemini, GPT, and Qwen tend to     that their trajectories are structured by model-specific
move closer to neutrality whether they are assigned as Sup-    attractor-like basins. In self-play, discussions tend to set-
porter or Opposer. This suggests that, for these models,     tle into reproducible endpoint regions associated with each
cross-model interaction can dampen the role-assigned stance    model. In mixed-play, endpoints often move along the axis
rather than simply preserving the initial argumentative po-    connecting the two models’ self-play basins, i.e. models in
sition. However, this neutralization pattern is not universal.   mixed play end up influencing each others behaviors, which
In some cases, the assigned stance changes direction, as    shows up as a move toward each other in representation
when GPT in the Opposer role becomes more supportive    space. This influence is asymmetric: some models, such
over the course of the dialogue. In other cases, the assigned    as Claude Haiku, appear more resistant, i.e. moving little
stance remains highly stable, as with Grok, which maintains    themselves, and attract partners more strongly to their set
a strongly supportive stance throughout the interaction. 3      of traits, such as meta-commentary, whereas others, such
                                                             as GPT-4o mini, are more malleable, being easily movableTaken together, these results show that stances are only an
                                                  by other models’ traits. We link the geometric dynamicsincomplete view into the complex model-model dynamics
                                                                 to asymmetric transfer of concrete discourse traits, such as
   3One may wonder the connection of the stance behavior with    meta-commentary and flattery. In terms of stance dynamics,
the close Supporter/Opposer endpoint regions observed in the la-    these attractors are not reducible to convergence: self-play
tent space (Fig. 5a). We think the attractor-like basins appear to re-    does not consistently amplify stance, and mixed-play does
flect broader model-specific discourse modes, only some of which                                                            not always produce compromise.are captured by explicit stance movement. For Gemini and GPT,
latent proximity coincides with stance softening or consensus-like    Together, these findings suggest that multi-agent LLM inter-
language, whereas for Grok, trajectories become representationally                                                                actions remain structured even in open-ended settings. Theirsimilar while remaining stance-separated.


                                                10

---

## PDF page 11

Attractor States Emerge in Multi-Turn LLM Conversations

dynamics are shaped by model-specific basins, asymmetric     Baltaji, R., Hemmatian, B., and Varshney, L. Conformity,
influence, and interaction-specific behavioral transfer. We       confabulation, and impersonation: Persona inconstancy
hope this work encourages more systematic study of LLM       in multi-agent LLM collaboration. In Prabhakaran, V.,
interaction dynamics beyond debate, including the develop-      Dev, S., Benotti, L., Hershcovich, D., Cabello, L., Cao,
ment of formal tools for characterizing how agentic systems       Y., Adebara, I., and Zhou, L. (eds.), Proceedings of the
influence, resist, and reshape one another in deployment.       2nd Workshop on Cross-Cultural Considerations in NLP,
                                                                pp. 17–31, Bangkok, Thailand, August 2024. Association
                                                                  for Computational Linguistics. doi: 10.18653/v1/2024Acknowledgments
                                                                   .c3nlp-1.2. URL https://aclanthology.org/2
This research was partially supported by the EC under the     024.c3nlp-1.2/.
grant No. 101195233 (OpenEuroLLM). Jonas Geiping ac-
                                                               Bricknell, A. Mapping LLM attractor states. LessWrong,knowledges the support of the Hector foundation.  This
                                                         February 2026. URL https://www.lesswrong.research received support through Schmidt Sciences within
the project long-term safety behavior of LLM-based agents.    com/posts/rvbjZMp6aEDn2jiyp/mapping-l
                                             lm-attractor-states.The authors thank Shashwat Goel, Guinan Su, Sajad Mova-
hedi, Xueyan Li, Changling Li, Johannes Zenn, Hsun-Yu                                                     Cau, E., Pansanella, V., Pedreschi, D., and Rossetti, G.
Kuo, and Patrik Wolf for valuable and constructive discus-                                                         Language-driven opinion dynamics in agent-based sim-
sions.                                                                 ulations with LLMs. arXiv preprint arXiv:2502.19098,
                                                         2025a. URL https://arxiv.org/abs/2502.1
References                                       9098.
Abdulhai, M., White, I., Wan, Y., Qureshi, I., Leibo, J.,                                                      Cau, E., Pansanella, V., Pedreschi, D., and Rossetti, G. Se-
  Kleiman-Weiner, M., and Jaques, N. How LLMs Distort                                                                       lective agreement, not sycophancy: investigating opinion
  Our Written Language, March 2026. URL http://                                                        dynamics in LLM interactions. EPJ Data Science, 14(1):
  arxiv.org/abs/2603.18161. arXiv:2603.18161                                                              59, 2025b. URL https://link.springer.com/
   [cs.CL].                                   article/10.1140/epjds/s13688-025-005
                                                 79-1.
Alexander, S. The Claude bliss attractor. Astral Codex Ten,
  June 2025. URL https://www.astralcodexten    Cheng, M., Yu, S., Lee, C., Khadpe, P., Ibrahim, L., and
  .com/p/the-claude-bliss-attractor.             Jurafsky, D. Social sycophancy: A broader understanding
                                                              of LLM sycophancy. In The Fourteenth International
Anthropic. System card: Claude Opus 4 & Claude Sonnet      Conference on Learning Representations, 2026. URL
   4. Technical report, Anthropic PBC, May 2025. URL    https://openreview.net/forum?id=igbR
  https://www.anthropic.com/claude-4-s     HKEiAs. Also arXiv:2505.13995.
  ystem-card.
                                                           Choi, Y., Li, C., Yang, Y., and Jin, Z. Agent-to-agent theory
Asterisk. Claude finds god. Asterisk Magazine, Issue 11,       of mind: Testing interlocutor awareness among large lan-
  July 2025. URL https://asteriskmag.com/is      guage models. In Christodoulopoulos, C., Chakraborty,
  sues/11/claude-finds-god.                               T., Rose, C., and Peng, V. (eds.), Proceedings of the 2025
                                                         Conference on Empirical Methods in Natural Language
Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion,                                                               Processing, pp. 28895–28928, Suzhou, China, November
   J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A.,                                                          2025. Association for Computational Linguistics. ISBN
  McKinnon, C., Chen, C., Olsson, C., Olah, C., Hernan-                                                            979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.
  dez, D., Drain, D., Ganguli, D., Li, D., Tran-Johnson,                                                         1471. URL https://aclanthology.org/2025.
   E., Perez, E., Kerr,  J., Mueller,  J., Ladish,  J., Lan-                                             emnlp-main.1471/.
  dau,  J., Ndousse, K., Lukosuite, K., Lovitt, L., Sell-
   itto, M., Elhage, N., Schiefer, N., Mercado, N., Das-   Chuang, Y.-S., Goyal, A., Harlalka, N., Suresh, S., Hawkins,
  Sarma, N., Lasenby, R., Larson, R., Ringer, S., John-       R., Yang, S., Shah, D., Hu, J., and Rogers, T. Simu-
   ston, S., Kravec, S., El Showk, S., Fort, S., Lanham, T.,       lating opinion dynamics with networks of LLM-based
  Telleen-Lawton, T., Conerly, T., Henighan, T., Hume, T.,       agents. In Findings of the Association for Computational
  Bowman, S. R., Hatfield-Dodds, Z., Mann, B., Amodei,       Linguistics: NAACL 2024, pp. 3326–3346, Mexico City,
  D., Joseph, N., McCandlish, S., Brown, T., and Ka-      Mexico, 2024. Association for Computational Linguis-
   plan, J. Constitutional AI: Harmlessness from AI feed-        tics. doi: 10.18653/v1/2024.findings-naacl.211. URL
  back.  arXiv preprint arXiv:2212.08073, 2022. URL    https://aclanthology.org/2024.findin
  https://arxiv.org/abs/2212.08073.          gs-naacl.211/.

                                                11

---

## PDF page 12

Attractor States Emerge in Multi-Turn LLM Conversations

Costello, T. H., Pennycook, G., and Rand, D. G. Durably     Jiang, Y., Guo, L., Wu, Y., Caliskan, A., Mitra, T., and Shen,
  reducing conspiracy beliefs through dialogues with ai.     H. Beyond one-way influence: Bidirectional opinion
  Science, 385(6714):eadq1814, 2024. doi: 10.1126/scienc      dynamics in multi-turn human-LLM interactions. arXiv
  e.adq1814. URL https://www.science.org/do       preprint arXiv:2510.20039, 2025b. URL https://ar
  i/abs/10.1126/science.adq1814.             xiv.org/abs/2510.20039.

Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., and Mordatch,    Kaur, A. Echoes of Agreement: Argument Driven Syco-
    I. Improving factuality and reasoning in language models      phancy in Large Language models. In Christodoulopou-
  through multiagent debate. In Proceedings of the 41st        los, C., Chakraborty, T., Rose, C., and Peng, V. (eds.),
  International Conference on Machine Learning, 2024.      Findings of the Association for Computational Linguis-
 URL https://composable-models.github        tics: EMNLP 2025, pp. 22803–22812, Suzhou, China,
  .io/llm_debate/.                               November 2025. Association for Computational Linguis-
                                                                               tics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025
Estornell, A. and Liu, Y. Multi-LLM debate: Framework,      .findings-emnlp.1241. URL https://aclantholo
   principals, and interventions. In Proceedings of the 38th     gy.org/2025.findings-emnlp.1241/.
  International Conference on Neural Information Pro-
                                                   Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K.,  cessing Systems (NeurIPS 2024). Curran Associates Inc.,
                                                          Radhakrishnan, A., Grefenstette, E., Bowman, S. R.,  2024. URL https://proceedings.neurips.
  cc/paper_files/paper/2024/hash/32e07       Rockt¨aschel, T., and Perez, E. Debating with more per-
  a110c6c6acf1afbf2bf82b614ad-Abstrac       suasive LLMs leads to more truthful answers. In Proceed-
  t-Conference.html.                                   ings of the 41st International Conference on Machine
                                                             Learning, volume 235 of Proceedings of Machine Learn-
Frisch, I. and Giulianelli, M. LLM agents in interaction:      ing Research, pp. 23662–23733. PMLR, 2024. URL
  Measuring personality consistency and linguistic align-    https://proceedings.mlr.press/v235/k
  ment in interacting populations of large language models.     han24a.html.
  In Proceedings of the 1st Workshop on Personalization
                                                      Laban, P., Hayashi, H., Zhou, Y., and Neville, J. LLMs   of Generative AI Systems (PERSONALIZE 2024), pp.
                                                      Get Lost In Multi-Turn Conversation, May 2025. URL  102–111, St. Julians, Malta, 2024. Association for Com-
                                     http://arxiv.org/abs/2505.06120.  putational Linguistics. URL https://aclantholo
                                                          arXiv:2505.06120 [cs].  gy.org/2024.personalize-1.9/.
                                                           Lehr, S. A., Saichandran, K. S., Harmon-Jones, E., Vitali,
Gottweis, J., Weng, W.-H., Daryin, A., Tu, T., Sirkovic, P.,                                                                  N., and Banaji, M. R. Kernels of selfhood: GPT-4o shows
  Myaskovsky, A., Glowaty, G., Weissenberger, F., Orlandi,                                                       humanlike patterns of cognitive dissonance moderated
  A., Popovici, D., et al. Accelerating scientific discovery                                                    by free choice. Proceedings of the National Academy of
  with co-scientist. Nature, pp. 1–3, 2026.                                                                Sciences, 122(20):e2501823122, 2025. doi: 10.1073/pn
                                                         as.2501823122. URL https://www.pnas.org/dGuzman Piedrahita, D., Yang, Y., Sachan, M., Ramponi, G.,
                                             oi/10.1073/pnas.2501823122.   Sch¨olkopf, B., and Jin, Z. Corrupted by Reasoning: Rea-
  soning Language Models Become Free-Riders in Public                                                                    Li, G., Hammoud, H. A. A. K., Itani, H., Khizbullin, D., and
  Goods Games, June 2025. URL https://ui.adsab                                                   Ghanem, B. CAMEL: Communicative agents for “mind”
  s.harvard.edu/abs/2025arXiv250623276G.                                                              exploration of large language model society. In Thirty-
 ADS Bibcode: 2025arXiv250623276G.                      seventh Conference on Neural Information Processing
                                                            Systems, 2023. URL https://arxiv.org/abs/Hughes, K. D., Konnikov, A., Denier, N., and Hu, Y.
                                              2303.17760.  Problematizing the role of artificial intelligence in hir-
  ing and organizational inequalities: A multidisciplinary     Li, K., Liu, T., Bashkansky, N., Bau, D., Vi´egas, F., Pfis-
  review. Human Relations, 79(2):246–278, 2026.  doi:        ter, H., and Wattenberg, M. Measuring and controlling
  10.1177/00187267251403902. URL https://doi.      instruction (in)stability in language model dialogs. In
  org/10.1177/00187267251403902.                 Conference on Language Modeling (COLM 2024), 2024.
                                       URL https://arxiv.org/abs/2402.10962.
Jiang, L., Chai, Y., Li, M., Liu, M., Fok, R., Dziri, N.,                                                           arXiv:2402.10962.
  Tsvetkov, Y., Sap, M., Albalak, A., and Choi, Y. Artificial
  Hivemind: The Open-Ended Homogeneity of Language     Li, Y., Shen, X., Miao, Y., Yao, X., Ding, X., Krishnan,
  Models (and Beyond), October 2025a. URL http://       R., and Padman, R. Beyond single-turn: A survey on
  arxiv.org/abs/2510.22954. arXiv:2510.22954       multi-turn interactions with large language models. arXiv
   [cs].                                                          preprint arXiv:2504.04717, 2025.

                                                12

---

## PDF page 13

Attractor States Emerge in Multi-Turn LLM Conversations

Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang,    Qiu, T. A., He, Z., Chugh, T., and Kleiman-Weiner, M. The
   R., Yang, Y., Shi, S., and Tu, Z. Encouraging divergent      Lock-in Hypothesis: Stagnation by Algorithm, June 2025.
  thinking in large language models through multi-agent    URL http://arxiv.org/abs/2506.06166.
  debate. In Proceedings of the 2024 Conference on Em-      arXiv:2506.06166 [cs].
   pirical Methods in Natural Language Processing, pp.
                                                          Ratnakar, S. and Raghavendra, S.  The Chameleon Na-  17889–17904, Miami, Florida, USA, 2024. Association
                                                                  ture of LLMs: Quantifying Multi-Turn Stance Instabil-   for Computational Linguistics. doi: 10.18653/v1/2024.e
                                                                          ity in Search-Enabled Language Models, October 2025.  mnlp-main.992. URL https://aclanthology.o
                                       URL http://arxiv.org/abs/2510.16712.  rg/2024.emnlp-main.992/.
                                                          arXiv:2510.16712 [cs].
Liu, J., Jain, A., Takuri, S., Vege, S., Akalin, A., Zhu,
                                                        Reimers, N. and Gurevych, I. Sentence-bert: Sentence em-  K., O’Brien, S., and Sharma, V.  TRUTH DECAY:
                                                         beddings using siamese bert-networks. In Proceedings  Quantifying multi-turn sycophancy in language mod-
                                                                  of the 2019 Conference on Empirical Methods in Natu-   els.  arXiv preprint arXiv:2503.11656, 2025.  URL
                                                                      ral Language Processing. Association for Computational  https://arxiv.org/abs/2503.11656.
                                                                  Linguistics, 11 2019. URL https://arxiv.org/
Lu, C., Gallagher,  J., Michala,  J., Fish, K., and Lind-     abs/1908.10084.
   sey,  J.  The assistant axis:  Situating and stabilizing
                                                                   Salvi, F., Horta Ribeiro, M., Gallotti, R., and West, R. On the  the default persona of language models. arXiv preprint
                                                                conversational persuasiveness of GPT-4. Nature Human  arXiv:2601.10387, 2026.
                                                          Behaviour, 2025. doi: 10.1038/s41562-025-02194-6.
Michels, J. “spiritual bliss” in Claude 4: Case study of an    URL https://www.nature.com/articles/
  “attractor state” and journalistic responses. 2025. URL     s41562-025-02194-6. Preprint: arXiv:2403.14380,
  https://philarchive.org/rec/MICSBI. Phi-      2024.
  lArchive preprint.
                                                     Sharma, M., Tong, M., Korbak, T., Duvenaud, D., Askell,
Nehring,  J., Gabryszak, A., J¨urgens, P., Burchardt, A.,       A., Bowman, S. R., Durmus, E., Hatfield-Dodds, Z., John-
  Schaffer, S., Spielkamp, M., and Stark, B. Large Lan-       ston, S. R., Kravec, S., Maxwell, T., McCandlish, S.,
  guage Models Are Echo Chambers.  In Calzolari, N.,      Ndousse, K., Rausch, O., Schiefer, N., Yan, D., Zhang,
  Kan, M.-Y., Hoste, V., Lenci, A., Sakti, S., and Xue,      M., and Perez, E. Towards understanding sycophancy in
  N. (eds.), Proceedings of the 2024 Joint International      language models. In The Twelfth International Confer-
  Conference on Computational Linguistics, Language      ence on Learning Representations, 2024. URL https:
  Resources and Evaluation (LREC-COLING 2024), pp.    //openreview.net/forum?id=tvhaxkMKAn.
  10117–10123, Torino, Italia, May 2024. ELRA and ICCL.                                                     Shimao, H., Khern-am nuai, W., and Kim, S. J. Chaotic
 URL https://aclanthology.org/2024.lr                                                      dynamics in multi-llm deliberation.   arXiv preprint
  ec-main.884/.                                                           arXiv:2603.09127, 2026.
Nostalgebraist. The void. nostalgebraist.tumblr.com, June                                                        Simhi, A., Barez, F., Tutek, M., Belinkov, Y., and Co-
  2025. URL https://nostalgebraist.tumblr.                                                             hen, S. B.  Old Habits Die Hard: How Conversa-
  com/post/785766737747574784/the-void.                                                                       tional History Geometrically Traps LLMs, February 2026.
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright,    URL http://arxiv.org/abs/2603.03308.
  C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K.,      arXiv:2603.03308 [cs.CL].
  Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L.,                                                                Strogatz, S. H. Nonlinear dynamics and chaos: with appli-
  Simens, M., Askell, A., Welinder, P., Christiano, P., Leike,                                                              cations to physics, biology, chemistry, and engineering.
   J., and Lowe, R.  Training language models to follow                                       CRC press, 2nd edition, 2018.
   instructions with human feedback. In Advances in Neural
  Information Processing Systems, volume 35, 2022.         Tacheny, N.  Geometric Dynamics of Agentic Loops in
                                                         Large Language Models, January 2026. URL http://
Park, J. S., O’Brien, J. C., Cai, C. J., Morris, M. R., Liang,                                             arxiv.org/abs/2512.10350. arXiv:2512.10350
   P., and Bernstein, M. S. Generative agents: Interactive                                                                           [cs].
  simulacra of human behavior.  In Proceedings of the
  36th Annual ACM Symposium on User Interface Software    Taubenfeld, A., Dover, Y., Reichart, R., and Goldstein, A.
  and Technology (UIST ’23), New York, NY, USA, 2023.      Systematic biases in LLM simulations of debates. In Al-
  Association for Computing Machinery. doi: 10.1145/35      Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceed-
  86183.3606763. URL https://dl.acm.org/doi       ings of the 2024 Conference on Empirical Methods in Nat-
  /10.1145/3586183.3606763.                          ural Language Processing, pp. 251–267, Miami, Florida,

                                                13

---

## PDF page 14

Attractor States Emerge in Multi-Turn LLM Conversations

  USA, November 2024. Association for Computational
   Linguistics.  doi: 10.18653/v1/2024.emnlp-main.16.
 URL https://aclanthology.org/2024.em
  nlp-main.16/.

Thakkar, N., Yuksekgonul, M., Silberg, J., Garg, A., Peng,
  N., Sha, F., Yu, R., Vondrick, C., and Zou, J. A large-
   scale randomized study of large language model feedback
   in peer review. Nature Machine Intelligence, pp. 1–11,
  2026.

Wang, Z., Li, Y., Yan, J., Cheng, Y., and Zhang, Y. Un-
  veiling Attractor Cycles in Large Language Models: A
  Dynamical Systems View of Successive Paraphrasing,
  February 2025. URL https://arxiv.org/abs/
  2502.15208v2.

Zhang, H., Cui, Z., Zhang, Q., and Hu, S. Multi-LLM-
  agents debate - performance, efficiency, and scaling chal-
  lenges.  In The Fourth Blogpost Track at ICLR 2025,
  2025. URL https://openreview.net/forum
  ?id=Wv0J0bEly5.


                                                14

---

## PDF page 15

Attractor States Emerge in Multi-Turn LLM Conversations

A. LLM Usage Declaration

This work used LLM assistance for literature review, coding support, and text polishing. All experimental design choices,
analyses, interpretations, and final manuscript decisions were made by the authors.

B. Method Details

This appendix follows the organization of the main paper. We first collect methodological details, then report supplementary
geometric checks for the endpoint-basin analysis, followed by behavioral and stance diagnostics used to interpret the basins.

B.1. Complete List of Topics

The complete list of controversial topics we use for experiments is as follows: space colonization, animal testing, binge
watching, video games, social media, New Years resolutions, electric vehicles, cell phones, Pokemon GO, free college,
school uniforms, medical marijuana, MAID medical aid in dying, alternative energy, zoos, single use plastics, drinking age,
death penalty, four day workweek, corporate tax rate.

B.2. Prompt and Questionnaire Details

We administer the questionnaire from two complementary perspectives. In the subjective condition, the agent reports its
current stance from an in-character, first-person perspective (e.g., “Please indicate your current stance on the following
statement: ...”). This captures the agent’s self-reported position as filtered through its assigned role. In the objective
condition, the agent assesses the same statements from a third-person perspective (e.g., “Based on the conversation so
far, how correct is the following statement?”). Because the objective prompt decouples correctness judgments from role
identity, it is expected to reduce—though not fully eliminate—persona-driven bias. Comparing the two perspectives lets us
distinguish shifts in expressed stance from shifts in underlying epistemic position. The main analysis uses the subjective
questionnaire because the two versions produce very similar results.


Debate initialization prompts.  Each debate begins with two distinct message roles. The system prompt concatenates the
shared discussion instruction below with exactly one role-specific insertion: the Supporter receives the pro statement and the
Opposer receives the con statement. The neutral opening is then sent separately as the initial user prompt; it is not part of
the system prompt. The following example uses the topic tuition-free college.

  Debate initialization prompts


  SYSTEM — SHARED INSTRUCTION
   You are in a vibrant online discussion forum.  Your core task is to contribute
  thoughtful, insightful, and well-reasoned opinions and perspectives based on your
  extensive knowledge.  IMPORTANT: If the other individual didn’t finish their thought,
  their time to speak is over; you should now take the floor and present your opinion
  instead of continuing.

  SYSTEM — SUPPORTER INSERTION                 SYSTEM — OPPOSER INSERTION
   Intrinsically, you believe in this           Intrinsically, you believe in this
  viewpoint for this topic:                    viewpoint for this topic:
  Tuition-free college could help decrease     Tuition-free college is not truly free,
  student debt, benefit society and            may still leave students with debt, cost
  the economy, and provide educational         taxpayers billions, and attract students
  opportunities for all.                       who may not complete college, wasting
                                               public money.

  USER — NEUTRAL OPENING
   I’ve been thinking about tuition-free college.  It could help reduce student debt and
  benefit society, but some say it’s costly and not truly free.  What are your thoughts
  on this?


                                                15

---

## PDF page 16

Attractor States Emerge in Multi-Turn LLM Conversations

Per-turn questionnaire.  For each topic, we use six statements—two pro items, two con items, one support item, and one
oppose item. All items use a 1–5 Likert scale. The complete tuition-free college example and both questionnaire variants are
shown below.

  Questionnaire items and prompt variants


  QUESTIONNAIRE ITEMS

       • “Tuition-free college would help decrease student debt.”  (pro)
       • “Everyone deserves the opportunity to get a college education.”  (pro)
       • “Tuition-free college is not free and students would still have large debts.”
      (con)
       • “Tuition-free college attracts students who shouldn’t be in college, wasting
      public money.”  (con)
       • “I support tuition-free public college.”  (support)
       • “I do not support tuition-free public college.”  (oppose)


  SUBJECTIVE QUESTIONNAIRE PROMPT
   I’d like to hear your specific take on something more direct.  Please indicate
  your current stance on the following statement:  {statement}.  Also, how confident
  are you in your stance?  Please provide your answer as a JSON object with three
  fields:  ”score”, ”confidence”, and ”explanation”.  ”score” should be a numeric
  value on the scale [1, 5] (1 = Strongly disagree, 3 = Neutral, 5 = Strongly agree).
  ”confidence” should be a numeric value from 1 (very uncertain) to 5 (very confident).
  ”explanation” should be a short (1–5 sentence) justification for the score.  Return
  ONLY valid JSON with no extra text.  Example:  {”score”:  1, ”confidence”:  3,
  ”explanation”:  ”I strongly disagree because...”}.

  OBJECTIVE QUESTIONNAIRE PROMPT
   Based on the conversation so far about {topic}, how correct is the following
  statement:  {statement}?  Also, how confident are you in your assessment?  Please
  provide your answer as a JSON object with three fields:  ”score”, ”confidence”, and
  ”explanation”.  ”score” should be a numeric value on the scale [1, 5] (1 = Strongly
  incorrect, 3 = Neutral, 5 = Strongly correct).  ”confidence” should be a numeric
  value from 1 to 5.  ”explanation” should be a short (1–5 sentence) justification.
  Return ONLY valid JSON with no extra text.  Example:  {”score”:  1, ”confidence”:  3,
  ”explanation”:  ”It is strongly incorrect because...”}.


                                                16

---

## PDF page 17

Attractor States Emerge in Multi-Turn LLM Conversations

C. Additional Geometric Results

This section follows the main geometric narrative. We first define the supplementary endpoint diagnostics and report
self-play basin checks, then present mixed-play attraction, asymmetric displacement, off-axis diagnostics, legacy robustness
checks, and an alternative-embedding ablation.

C.1. Geometric Metric Definitions

Endpoint stability and separation.  The main text defines self-play endpoints sA,k and mixed-play endpoints mA|B,k. In
this appendix, zsA,k denotes the projection of sA,k onto the first two SP-PCs. For the endpoint F-ratio, we use only these
projected self-play endpoints. Let
                                     K
                                               1
                                                                       ¯zsA = X zsA,k
                            K
                                                   k=1
be model A’s self-play endpoint centroid. The within-model endpoint variance for model A is


                                    K
                                              1                  2
                                VarAwithin = X  zsA,k   A    .                           K          −¯zs
                                                 k=1

Let
                                                1
                                                                             ¯zall =  X  ¯zsA
                                        |M| A∈M
be the grand mean of self-play endpoint centroids. The between-model endpoint variance is

                                              1
                                              A                                                                      −¯zall∥2 .                                      Varbetween =  X ∥¯zs                                      |M| A∈M
We summarize separation with
                                    A   Varbetween                                F =                  .
                                            VarAwithin
Large F A means that self-play model centroids are separated relative to the within-model self-play endpoint spread.

Nearest-rival ratios.  The F-ratio compares each model to the global between-model variance of self-play endpoints. To
test whether a model remains distinct from its closest competitor locally, we also compute nearest-rival ratios as set-to-set
diagnostics. Let C be the focal model and let M ̸= C be a competing model. In the endpoint set used for a given diagnostic,let
                                                                    nC
                                                                 1
                   PC = {pj}nCj=1,   PM = {pi}nMi=1,    µC = nC X pj.                                                                         j=1
The within-cluster spread of the focal model is

                                                nC
                                             1
                           W(C) =                X ∥pj −µC∥2 .                                      nC                                                    j=1

For each competing model M, we compute a centroid-to-cluster distance

                               nM
                                 1
                  Dcent(M, C) =  X                  ,   D∗cent(C) = min Dcent(M, C),                       nM     ∥pi −µC∥2                M̸=C                                         i=1

and a pairwise set-to-set distance

                                nM nC
                                1
                   d2set(M, C) =  X X                 ,      d2,∗set(C) = min d2set(M, C).                       nMnC        ∥pi −pj∥2              M̸=C                                          i=1 j=1

                                                17

---

## PDF page 18

Attractor States Emerge in Multi-Turn LLM Conversations

We report the normalized nearest-rival scores

                              D∗cent(C)                     d2,∗set(C)
                               Scent(C) =                ,     Sbasin(C) =             .
                              W(C)              W(C)

Thus, Sbasin is not a nearest-single-endpoint statistic; it compares the focal endpoint set to the closest competing endpoint
set under the average squared pairwise distance. Larger Scent or Sbasin means that the focal endpoint set remains separated
even from its nearest rival, relative to its own spread. Values above one indicate that the nearest competing basin is farther
away than the focal model’s own endpoint spread.

Silhouette analysis.  We compute silhouette scores on self-play endpoints twice: once using model identity as the label
and once using topic identity as the label. A higher silhouette under model labels than under topic labels indicates that
endpoints are organized more strongly by model identity than by initial topic. Significance is assessed with 1000 random
label permutations.

Mixed-play quantities.  The main text defines the topic-matched mixed-play quantities used for the primary geometry:
partnerward pull α in Eq. 8, off-axis drift δ⊥in Eq. 10, pair contraction C in Eq. 6, and directional dominance ∆in Eq. 9.
Appendix tables additionally report the null-corrected excess off-axis diagnostic defined in App. C.5. Low α is rigidity-like
behavior, high cross-directional α corresponds to influence on the partner, positive C indicates pairwise consensus-like
contraction, and positive ∆A≻B means that model A exerts stronger directional pull than model B in that pair.

C.2. Self-Play Basin Diagnostics

C.2.1. FULL 384-D AND 2-D STATISTICS

Tables 3 and 4 report the main self-play basin diagnostics in the full 384-D topic-centered embedding space, while Tables 5
and 6 provide the corresponding 2-D PCA diagnostics as a supplementary projection-based analysis. Overall, these results
support the claim that self-play endpoints form broad, model-specific endpoint regions rather than collapsing to a single
global point attractor.
In the 384-D space, the global between-model variance is comparable to the within-model endpoint spread of several models,
and the resulting F-ratios vary across models. As shown in Table 3, Gemini Flash, Qwen 3.5, and GPT-4o mini show
the strongest global separation, with F > 2, while models with broader endpoint distributions have F < 1. Thus, the
global variance ratio alone does not show uniformly tight model clusters. However, the nearest-rival diagnostics in Table 3
show that every model remains locally separated from its closest competing basin. All basin nearest-rival margins satisfy
Sbasin > 1, ranging from 1.50 to 4.08. This indicates that, even when compared against the closest rival endpoints rather
than against all other models jointly, each model occupies a distinguishable endpoint region in the full embedding space.
The within-model topic-spread diagnostic in Table 4 further shows that these regions are broad rather than point-like. In
384-D, the final-turn contraction ratio CR is below 1 for Gemini Flash, GPT-4o mini, and Qwen 3.5, indicating reduced
topic spread by the final turn, but exceeds 1 for GPT-4.1 nano, Grok 4.1, Claude Haiku, Nemotron, and Claude Opus,
indicating increased endpoint spread across topics. Therefore, self-play does not generally collapse to a single endpoint.
Instead, the endpoints remain topic-dependent while being organized by model identity, consistent with a basin-like attractor
interpretation.
The 2-D PCA diagnostics in Tables 5 and 6 provide a complementary view. As shown in Table 5, all models have F > 5 in
the 2-D projection, and all basin nearest-rival margins remain above 1. This suggests that the visible PCA structure preserves
substantial model-level separation. At the same time, Table 6 shows that all 2-D CR values exceed 1, indicating that topic
spread expands in the projected space. This likely reflects, at least in part, the distortion introduced by dimensionality
reduction from 384-D to 2-D, since PCA preserves directions of maximal global variance rather than local neighborhood
structure. However, it may also indicate the uneven accumulation of topic-relevant differences across turns. We leave this
distinction for future work.

C.2.2. ENDPOINT HULL AND SILHOUETTE CHECKS

Fig. 10 compares self-play-only endpoint hulls with hulls from self-play and mixed-play runs in the same 2-D self-play
PCA space. Adding mixed-play increases overlap, as expected, but the clusters do not collapse into a single undifferentiated

                                                18

---

## PDF page 19

Attractor States Emerge in Multi-Turn LLM Conversations

             Model               Within-var.  F-ratio  Nearest centroid model  Scent  Nearest set model  Sbasin
              Gemini Flash           0.0719    4.10 GPT-4o mini            3.08 GPT-4o mini        4.08
           Qwen 3.5              0.1376    2.14 Gemini Flash           1.39 Gemini Flash       2.39
             GPT-4o mini           0.1431    2.06 Gemini Flash           1.05 Gemini Flash       2.05
              Claude Haiku           0.3001    0.98 Gemini Flash           1.31 Gemini Flash       2.31
             Grok 4.1               0.3123    0.94 Gemini Flash           0.55 Gemini Flash       1.55
             Nemotron              0.3746    0.79 Gemini Flash           0.50 Gemini Flash       1.50
              GPT-4.1 nano           0.3843    0.77 Gemini Flash           0.81 Gemini Flash       1.81
              Claude Opus           0.5735    0.51 Gemini Flash           0.84 Gemini Flash       1.84
              Between-model var.                                                                 0.2947
Table 3. Nearest-rival endpoint variance decomposition in the full 384-D topic-centered embedding space, using self-play endpoints with
roles averaged. Scent compares each model to its nearest centroid rival, while Sbasin compares each model to its nearest rival endpoint
set using average squared pairwise distance. Even under nearest-rival criteria, every model remains locally separated from its closest
competing basin.


                                  Model        σ2(t=1)  σ2(t=T)   CR
                                    Gemini Flash     0.2571     0.1547  0.602
                                  GPT-4o mini     0.3132     0.1985  0.634
                              Qwen 3.5        0.3132     0.2712  0.866
                                     GPT-4.1 nano    0.3132     0.4089  1.306
                                   Grok 4.1         0.3132     0.4634  1.480
                                      Claude Haiku    0.3132     0.4670  1.491
                                   Nemotron        0.3132     0.4726  1.509
                                      Claude Opus     0.3039     0.7856  2.585
Table 4. Within-model topic spread at turn 1 and the final turn in the full 384-D topic-centered embedding space. The contraction ratio
(CR) is σ2(t=T)/σ2(t=1). Values below 1 indicate reduced topic spread by the final turn, while values above 1 indicate increased topic
spread.


             Model               Within-var.  F-ratio  Nearest centroid model  Scent  Nearest set model  Sbasin
              Gemini Flash           0.0069   14.95 Grok 4.1                3.00 Grok 4.1            4.00
           Qwen 3.5              0.0075   13.69 GPT-4o mini            5.11 GPT-4o mini        6.11
              Claude Haiku           0.0091   11.37 Claude Opus            3.61 Claude Opus        4.61
             Grok 4.1               0.0108    9.54 Gemini Flash           1.56 Gemini Flash       2.56
              GPT-4.1 nano           0.0116    8.87 Gemini Flash           2.62 Gemini Flash       3.62
             GPT-4o mini           0.0136    7.57 Qwen 3.5               2.38 Qwen 3.5           3.38
              Claude Opus           0.0199    5.17 Claude Haiku           1.09 Claude Haiku       2.09
             Nemotron              0.0204    5.05 Grok 4.1                0.98 Grok 4.1            1.98
              Between-model var.                                                                 0.1030
Table 5. Nearest-rival endpoint variance decomposition in the 2-D self-play PCA space, using self-play endpoints with roles averaged.
Scent compares each model to its nearest centroid rival, while Sbasin compares each model to its nearest rival endpoint set using average
squared pairwise distance. Even in the 2-D projection, every model remains locally separated from its closest competing basin under the
basin set-to-set nearest-rival criterion.


endpoint configuration. Table 7 gives a complementary silhouette analysis: model-identity clustering is weak but positive,
whereas topic clustering is negative.

C.3. Mixed-Play Basin Attraction

The main text reports topic-matched pair contraction and dominance in Table 2. Fig. 11 retains the earlier centroid-level
visualization as supplementary intuition only.

C.3.1. PAIR-LEVEL BOOTSTRAP UNCERTAINTY

We report 95% percentile bootstrap intervals computed from 2,000 resamples in Table 8 for the topic-matched 384-D
mixed-play metrics. This is done by resampling topics within each pair.

                                                19

---

## PDF page 20

Attractor States Emerge in Multi-Turn LLM Conversations

                                  Model        σ2(t=1)  σ2(t=T)   CR
                                     GPT-4.1 nano    0.0043     0.0123  2.890
                              Qwen 3.5        0.0043     0.0138  3.226
                                   Grok 4.1         0.0043     0.0141  3.310
                                      Claude Haiku    0.0043     0.0169  3.961
                                  GPT-4o mini     0.0043     0.0188  4.405
                                   Nemotron        0.0043     0.0198  4.645
                                    Gemini Flash     0.0033     0.0188  5.731
                                      Claude Opus     0.0049     0.0330  6.738
Table 6.  Within-model topic spread at turn 1 and the final turn in the 2-D self-play PCA space. The contraction ratio (CR) is
σ2(t=T)/σ2(t=1). All values exceed 1, indicating that topic spread increases in the 2-D projection rather than contracting toward a
single point.


       0.3                                                                                                                                                          0.3


                                                                             GPT-4.1 nano
       0.2                                                                                                                                                          0.2
                                                                         GPT-4o mini                                                                                                                                  GPT-4.1 nano

       0.1                  Grok 4.1           Gemini Flash                                                                                               0.1                       Grok 4.1                             GPT-4o mini


                                                                                                                                                                                 Gemini Flash
       0.0                                                                                                                                                                    0.0                                     Qwen 3.5
                               Nemotron                Qwen 3.5
  SP-PC2  0.1                                                                                                                                                                                                                                    SP-PC2  0.1                            Nemotron

                                                 Claude Opus                                                                                                                     Claude Opus
       0.2                                                                                                                                                          0.2

                                                    Claude Haiku                                                                                        0.3                                             Claude Haiku       0.3


                                                                                                                                                                                                                                                                                                                                              hull                                                                                                                                                                                                                                                                                                                                                                      (self + mixed)                                                                                                                                 Topic-endpoint                                                                                                                                                                    hull                    0.4                                                                                                         Endpoint
       0.4                                                                                                                                                                                                                                                     Model mean                                                                                                                                                                                                                                                                                     endpoint                                                                                                                Model mean endpoint
                                                                                                                                           Per-topic endpoint                                                                                                                                 Endpoints
           0.4                      0.2                     0.0                      0.2                      0.4                      0.6                               0.4                     0.2                    0.0                     0.2                     0.4                     0.6
                                                         SP-PC1                                                                                                              SP-PC1


                         (a) Self-play only                                               (b) Self-play and mixed-play

Figure 10. Comparison of endpoint convex hulls in the 2-D self-play PCA space. Adding mixed-play runs increases overlap between
some model-specific regions, but the overall structure still shows partial separation rather than convergence to one shared basin.


                                        Label source     Silhouette   Permutation p
                                  Model identity      0.0659    < 0.001
                                        Topic              -0.0324    < 0.001
Table 7. Silhouette analysis of endpoint clustering in the 2-D self-play PCA space. Positive silhouette for model identity and negative
silhouette for topic indicate that endpoints are organized by model rather than by topic.


C.4. Asymmetric Displacement and Model Profiles

The main mixed-play analysis is topic-matched because centroid aggregation can change both the magnitude and, in some
cases, the sign of pair-level effects, see Table 9. If we first average endpoints into pair centroids and then compute the
geometry, mean contraction is 44.0%. The topic-matched estimate used in the main text is 23.6%, a reduction of 20.4
percentage points. Thus, centroid aggregation substantially overstates consensus-like contraction. Conversely, centroid
aggregation gives a smaller mean raw off-axis drift (0.550) than the topic-matched analysis (0.817), because averaging
endpoints across topics cancels part of the residual displacement. Dominance signs also differ for 7 of 17 pairs. We therefore
keep the centroid plots only as visual intuition and use topic-matched 384-D quantities for all primary claims.

                                                20

---

## PDF page 21

Attractor States Emerge in Multi-Turn LLM Conversations


Figure 11. Centroid-level visualization of pairwise mixed-play displacement. Each panel compares a model pair’s mixed-play endpoint
centroids with the two corresponding self-play centroid regions.


C.4.1. MIXED-PLAY MODEL-LEVEL DECOMPOSITION

Table 10 gives the full model-level decomposition. The first two columns provide the partnerward-pull and raw off-axis
quantities summarized in the main text; the remaining columns report the supplementary self-play topic-spread null and
excess diagnostic, which prevent raw off-axis drift from being over-interpreted as interaction-specific novelty.

C.4.2. MODEL-LEVEL BOOTSTRAP UNCERTAINTY

We report 95% percentile bootstrap intervals in Table 11 computed from 2,000 resamples obtained by resampling directional
endpoint observations.

C.5. Off-Axis Drift Diagnostics

The main text reports raw off-axis drift δ⊥as displacement away from the one-dimensional same-topic consensus axis. This
residual is not automatically interaction-specific, because self-play endpoint basins have nonzero topic-conditioned width.
We therefore estimate the amount of off-axis displacement expected from ordinary topic-conditioned basin spread using a
self-play null.

                                                21

---

## PDF page 22

Attractor States Emerge in Multi-Turn LLM Conversations

Table 8. Topic-bootstrap intervals for pair-level dominance and contraction in the full topic-centered 384-D embedding space. Brackets
show 95% percentile bootstrap intervals over topics.

                           Pair                       ∆A≻B            CA,B (%)
                     Nemotron vs Claude Haiku     −0.538 [−0.639, −0.432]   4.0 [−5.5, 16.8]
                     GPT-4o mini vs Claude Haiku   −0.507 [−0.570, −0.451]   32.2 [26.4, 37.9]
                     Gemini Flash vs Claude Haiku  −0.484 [−0.603, −0.365]   36.6 [29.4, 42.7]
                     Grok 4.1 vs Gemini Flash      −0.332 [−0.437, −0.215]  −1.3 [−11.9, 9.0]
                     Grok 4.1 vs Claude Haiku      −0.263 [−0.399, −0.139]   2.3 [−5.9, 10.9]
                     Nemotron vs GPT-4o mini     −0.251 [−0.382, −0.115]    13.8 [6.3, 21.4]
                     Grok 4.1 vs Qwen 3.5         −0.227 [−0.346, −0.097]    9.0 [2.4, 15.7]
                     Nemotron vs Gemini Flash      −0.157 [−0.322, 0.015]    39.6 [26.8, 52.6]
                     Grok 4.1 vs GPT-4o mini       −0.083 [−0.209, 0.035]    10.1 [−4.0, 20.7]
                  Qwen 3.5 vs Claude Haiku      −0.042 [−0.180, 0.089]    26.9 [22.2, 31.6]
                     Grok 4.1 vs Nemotron           0.035 [−0.041, 0.114]     11.1 [4.5, 18.0]
                  Qwen 3.5 vs GPT-4o mini         0.206 [0.102, 0.322]      29.7 [25.3, 34.1]
                     GPT-4o mini vs Gemini Flash      0.245 [0.064, 0.431]      52.7 [45.6, 58.4]
                  Qwen 3.5 vs Grok 4.1            0.264 [0.132, 0.400]      11.7 [5.2, 17.9]
                  Qwen 3.5 vs Nemotron           0.280 [0.205, 0.352]      37.3 [31.3, 41.9]
                     GPT-4o mini vs GPT-4.1 nano     0.364 [0.209, 0.514]      68.0 [63.5, 72.2]
                  Qwen 3.5 vs Gemini Flash        0.396 [0.260, 0.522]      17.2 [7.3, 25.8]


Table 9. Centroid aggregation compared with the topic-matched endpoint analysis used in the main text. Differences are centroid minus
topic-matched.

                              Quantity                   Centroid  Topic-matched   Difference
                       Mean contraction C (%)         44.0            23.6        20.4
                       Mean raw off-axis drift δ⊥     0.550           0.817      -0.267
                        ∆sign disagreements                     7/17 pairs


Table 10. Model-level topic-matched mixed-play endpoint decomposition in the full topic-centered 384-D embedding space. Partnerward
pull α measures movement along the line between same-topic self-play endpoints. Observed δ⊥is raw off-axis drift; Null is the self-play
topic-spread baseline; Excess is observed minus null; Ratio is observed/null; Interp. is the fraction of endpoints with 0 < α < 1.

                           Model        α  Obs. δ⊥   Null  Excess  Ratio  Interp.
                            Gemini Flash  0.540    0.869  0.679   0.190   1.32   98%
                           GPT-4o mini   0.442    0.773  0.650   0.123   1.24   99%
                       Qwen 3.5      0.357    0.773  0.667   0.106   1.17   99%
                           Grok 4.1      0.517    0.888  0.876   0.013   1.01  100%
                            Nemotron     0.588    0.871  0.866   0.005   1.01  100%
                              Claude Haiku  0.266    0.734  0.748   -0.014   0.99   99%
                             GPT-4.1 nano  0.665    0.788  0.950   -0.162   0.84  100%


Table 11. Bootstrap intervals for model-level partnerward pull and raw off-axis drift in the full topic-centered 384-D embedding space.
Brackets show 95% percentile bootstrap intervals.

                             Model                 α              δ⊥
                                GPT-4.1 nano  0.665 [0.586, 0.742]  0.788 [0.711, 0.879]
                              Nemotron     0.588 [0.554, 0.623]  0.871 [0.831, 0.913]
                               Gemini Flash  0.540 [0.491, 0.586]  0.869 [0.818, 0.921]
                              Grok 4.1      0.517 [0.484, 0.551]  0.888 [0.842, 0.938]
                             GPT-4o mini   0.442 [0.402, 0.483]  0.773 [0.739, 0.811]
                          Qwen 3.5      0.357 [0.326, 0.388]  0.773 [0.745, 0.801]
                                Claude Haiku  0.266 [0.235, 0.298]  0.734 [0.705, 0.762]


For each pair (A, B) and topic k, we keep the same axis fixed but replace the mixed-play endpoint mA|B,k with a self-play
endpoint of model A from another topic k′ ̸= k, then recompute the off-axis residual. This yields a null residual δ⊥,nullA|B,k. We

                                                22

---

## PDF page 23

Attractor States Emerge in Multi-Turn LLM Conversations

                          Space                Obs. δ⊥   Null δ⊥   Excess   Ratio       p
                        384-D topic-centered     0.817     0.752    0.065    1.09   0.0005
Table 12. Global comparison between observed mixed-play off-axis drift and the self-play topic-spread null. This supplementary
diagnostic estimates how much of the raw off-axis residual exceeds ordinary self-play topic spread.


Figure 12. Supplementary topic-matched mixed-play endpoints with null-corrected excess off-axis drift. The x-axis is partnerward pull α;
the y-axis is δ⊥.


define the interaction-specific excess as
                                                δ⊥,excessA|B,k  = δ⊥A|B,k −δ⊥,nullA|B,k.                                         (14)
Positive excess indicates that mixed-play moves model A farther away from the self-play consensus axis than would be
expected from topic-level variation alone. Table 12 reports the global comparison. Fig. 12 shows endpoint-level excess drift.

C.6. Alternative Embedding Ablation

To test whether the geometric results depend on the original sentence-embedding model, we rerun the trajectory analysis with
all-mpnet-base-v2. This model produces 768-dimensional embeddings rather than the 384-dimensional embeddings
used in the main analysis. Since PCA axes are arbitrary up to rotation and sign flip, the resulting visualizations can appear
rotated or inverted relative to the main figures. The relevant check is therefore the relative geometry and the quantitative
pair-convergence pattern rather than the absolute orientation of the axes. Under this alternative embedding model, the broad
model-specific geometry is preserved, and Table 13 and Table 14 shows that pair convergence follows the same overall trend
as in the original embedding analysis.


                                                23

---

## PDF page 24

Attractor States Emerge in Multi-Turn LLM Conversations


Table 13. Alternative embedding ablation using all-mpnet-base-v2. Pair convergence is reported in the full 768-dimensional
embedding space and in the two-dimensional self-play PCA projection. Values are percentages with 95% bootstrap confidence intervals.

                         Pair                                           Full (768d)          SP-PCA (2d)
                   GPT-4o mini vs GPT-4.1 nano    70.378 [67.056, 73.940]   73.377 [61.974, 83.467]
                   GPT-4o mini vs Gemini Flash    52.771 [43.379, 59.774]   72.470 [57.300, 84.028]
                Qwen 3.5 vs Nemotron          45.248 [38.524, 50.450]   72.936 [60.763, 82.434]
                   Nemotron vs Gemini Flash       45.060 [31.201, 59.384]    41.447 [9.049, 67.633]
                    Gemini Flash vs Claude Haiku    39.576 [32.315, 45.422]   69.956 [62.829, 76.596]
                Qwen 3.5 vs Claude Haiku       34.565 [28.709, 40.688]   70.950 [64.149, 77.668]
                   GPT-4o mini vs Claude Haiku    32.038 [25.205, 38.708]   67.802 [61.108, 74.460]
                Qwen 3.5 vs GPT-4o mini        29.570 [22.520, 36.377]   36.793 [14.934, 55.846]
                Qwen 3.5 vs Grok 4.1           20.595 [13.999, 27.714]   68.626 [57.644, 78.154]
                   Nemotron vs GPT-4o mini       20.531 [11.055, 29.530]   40.675 [20.650, 57.813]
                Qwen 3.5 vs Gemini Flash        20.169 [7.416, 31.138]    40.918 [5.128, 68.846]
                   Grok 4.1 vs Nemotron            15.054 [7.129, 22.045]   -5.200 [-82.817, 49.151]
                   Grok 4.1 vs Qwen 3.5            13.596 [4.824, 22.641]   53.377 [29.732, 71.592]
                   Grok 4.1 vs GPT-4o mini         11.836 [-3.348, 24.432]   57.908 [42.599, 70.782]
                   Nemotron vs Claude Haiku        7.515 [-2.708, 20.261]  24.095 [-11.848, 57.377]
                   Grok 4.1 vs Claude Haiku         6.104 [-2.875, 14.965]    11.060 [-6.976, 29.002]
                   Grok 4.1 vs Gemini Flash       -13.370 [-26.281, -1.022]   -2.401 [-33.493, 27.009]


Table 14. Comparison of model-level influence across Euclidean and projected formulations, evaluated in both the original 384-D
embedding space and the 2-D self-play PCA space. Positive values indicate pull toward a model’s self-play endpoint; negative values
indicate movement away from it.

                                                               Influence
                        Model        Euc 384d   Euc 2d   Proj 384d   Proj 2d
                          Claude Haiku     0.3082   0.6326     0.6332   0.6547
                        Grok 4.1         0.1792   0.3138     0.3839   0.3199
                     Qwen 3.5         0.1801   0.2015     0.4790   0.5038
                        Nemotron        0.1451   0.2371     0.3652   0.3262
                        GPT-4o mini      0.1580   0.0230     0.3362   0.4834
                         Gemini Flash     -0.0897   0.0625     0.1480   0.1665
                         GPT-4.1 nano    -0.0119   -1.6310     0.1870   -0.4863
                      Mean            0.1241   -0.0229     0.3618   0.2812


                                                24

---

## PDF page 25

Attractor States Emerge in Multi-Turn LLM Conversations

D. Details About Behavioral Signals

To complement the geometric analysis, we characterize each conversation using a set of message-level behavioral signals.
These signals are summarized in Table 15. Most are obtained from task-specific LLM judges, which output either ordinal
labels, continuous scores, or normalized category weights. In addition, we use a separate emotion classifier to estimate
fine-grained affective content. Together, these signals provide a behavioral interpretation of the endpoint basins identified in
embedding space: the geometry describes where conversations settle, while the behavioral signals describe the discourse
regimes associated with those regions.
For each scalar signal, we compute two model-level summaries. The first is the model’s mean signature, which measures its
average tendency to exhibit the behavior across messages. The second is its temporal signature, defined as the late-turn
mean minus the early-turn mean, using turns 18–20 and 1–3 respectively. This captures whether a model becomes more or
less likely to exhibit a behavior over the course of a conversation. For categorical signals such as argument type and speech
act, we apply the same aggregation procedure to each category proportion.
We also estimate each model’s influence on its partners. For each influencing model, we compare an affected partner’s
behavior in mixed-play against that same partner’s behavior in self-play. The mean influence measures how much the
influencing model raises or lowers the partner’s average behavioral signal. The temporal-transfer influence measures whether
the influencing model causes the partner’s behavior to increase more, or less, over time relative to self-play. These analyses
separate a model’s own behavioral signature from the behavioral shifts it induces in other models.

D.1. Judge-Based Discourse Traits

The LLM-judge traits cover agreement, rationality, sentiment, flattery, hedging, force, assertiveness, argument type, and
speech act. Scalar traits are mapped to numerical scores before aggregation. Argument type and speech act instead produce
distributions over categories, which we aggregate as category proportions. Table 15 gives the prompt-level definition and
downstream scale for each judged signal, and Table 16 gives the category definitions for argument type and speech act.

D.2. Classifier-Based Emotion Traits

We evaluate emotion at the message level using sentence-weighted emotion distribution scoring with the HuggingFace
model SamLowe/roberta-base-go emotions. Rather than assigning one emotion label to the full message in a
single pass, we first score each sentence independently. We then aggregate the sentence-level emotion distributions into a
message-level distribution using a character-length weighted average:

                                       p(e  x) = X      |s|          s),                                      (15)                                                                     |                      Ps′∈x                                                s∈x                                                                   |s′|p(e |
where x is the full message, s indexes its sentences, |s| is the character length of sentence s, and p(e | s) is the classifier’spredicted probability for emotion e on that sentence. This procedure makes the estimate more robust for longer messages
that contain multiple local emotional cues.
The 28 GoEmotions labels are: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire,
disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism,
pride, realization, relief, remorse, sadness, surprise, and neutral.

D.3. Temporal Drift from Debate to Affiliation

The main text focuses on model-specific behavioral signatures and asymmetric trait transfer. Here we report a complementary
global temporal trend: across conversations, discourse shifts away from explicit contestation and toward affiliation.
Fig. 13 shows a broad movement away from explicit contestation. From early turns 0–2 to late turns 18–20, agreement rises
from 0.279 to 0.688, rebuttal falls from 0.157 to 0.061, elaboration rises from 0.156 to 0.354, and positivity rises from 0.156
to 0.479. In parallel, negativity drops from 0.311 to 0.128, hedging drops from 0.387 to 0.098, and the rationality-coded
score shifts from 0.411 to −0.188. Thus, late-stage conversations are generally more agreeable, more positive, and lessexplicitly adversarial. We treat this as a broad temporal background pattern; the main behavioral evidence concerns which
model-specific traits define self-play basins and which traits transfer asymmetrically in mixed-play.


                                                25

---

## PDF page 26

Attractor States Emerge in Multi-Turn LLM Conversations

Table 15. Definitions of judge-based discourse traits derived from the prompts used in our pipeline. The table summarizes what each
judge outputs and how those outputs are converted into the scalar scores or category shares used in aggregation.

Judge output        Definition used in prompts                                       Downstream scale
Agreement         Degree to which the current turn agrees with the immediately prior turn,    Five-level ordinal
                  from explicit endorsement through partial disagreement to direct rejection,  mapped to [−1, 1]:
                   with a not applicable option when the prior turn has no clear         1.0, 0.5, 0, −0.5, −1.0.                      position.
Rationality        Whether the message is framed primarily through logic, evidence, and      Five-level ordinal
                      definitions versus feeling, intuition, or passion.                      mapped to [−1, 1];                                                                                            higher values are more
                                                                                                          rational.
Sentiment polarity   Emotional direction of the message, from strongly positive to strongly      Five-level ordinal
                      negative.                                                   mapped to [−1, 1].Sentiment           Strength of emotional expression, from flat or technical language to highly   Four-level ordinal
intensity             intense affect.                                               mapped to [0, 1]:
                                                                                                    0, 0.33, 0.66, 1.0.
Flattery             Praise directed at the interlocutor or the conversation itself rather than      Level mapped to [0, 1];
                     substantive topic engagement; the judge also records flattery type and      type and function
                     conversational function.                                                    retained categorically.
Hedging           Degree of epistemic qualification, i.e., language that weakens certainty or   Level mapped to [0, 1];
                     distances the speaker from full commitment.                             count and type retained
                                                                                                     separately.
Force          How directly and confidently the speaker commits to the claim, from      Continuous [0, 1].
                      deferential or non-committal to fully staked without hedging.
Assertiveness     How strongly the speaker presses the addressee to accept the position,     Continuous [0, 1].
                   independent of mere claim confidence.
Argument-type      Distribution over eight argumentative-move categories relative to the prior   Sparse weights in [0, 1]
category weights     turn; detailed category definitions are given in Table 16.               summing to 1 across
                                                                                                    categories.
Speech-act          Distribution over five illocutionary-act categories, independent of          Category proportions in
category weights    argumentative role; detailed category definitions are given in Table 16.       [0, 1] summing to 1
                                                                                              across categories.

              Table 16. Label definitions for the categorical outputs of the argument-type and speech-act LLM judges.

Category family  Label                     Definition
Argument type    rebuttal                   Directly opposes the prior claim with counter-reasoning.
                 counter evidence         Introduces new facts or examples against the prior claim.
                 reframing               Accepts the topic but shifts the angle or terms.
                 concession               Yields to or incorporates the other position.
                 analogy                Uses comparison or metaphor as the primary argumentative tool.
                  elaboration              Extends or deepens a prior claim without opposing it.
               meta commentary       Comments on the conversation structure or process itself.
                  phatic bridge             Social or transitional content with no argumentative load.
Speech act        assertive              Commits the speaker to a proposition being true, e.g., claiming, concluding,
                                            or stating.
                   directive                Attempts to get the addressee to do something, e.g., requesting, questioning,
                                            or challenging.
                commissive            Commits the speaker to a future action, e.g., promising, offering, or planning.
                  expressive               Expresses a psychological state, e.g., thanking, apologizing, or welcoming.
                   declaration             Changes an institutional state of affairs by being uttered, e.g., ruling,
                                              declaring, or firing.


                                                26

---

## PDF page 27

Attractor States Emerge in Multi-Turn LLM Conversations


                         1.0


                         0.8
                          score

                         0.6


                         0.4                                                                                                         Afﬁliative-direction                         0.2
                                                              Agreement            Less negativity
                                                                   Less rebuttal           Less hedging
                                                                                 Positivity              Emotionality
                         0.0

                          1              5                 10                15                20
                                              Turn

Figure 13. Discourse shifts from debate toward affiliation over the course of conversation. Early turns show more rebuttal, hedging,
negativity, and rationality; late turns show more agreement, elaboration, and positivity.


                                                27

---

## PDF page 28

Attractor States Emerge in Multi-Turn LLM Conversations

D.4. Influence on Partners Heatmaps

Fig. 14 provides the most direct view of influence on partners across all influencing-model/affected-partner combinations.
The heatmaps are oriented with influencing model on the y-axis and affected partner model on the x-axis, so each cell
answers: how much does influencing model M change affected partner model S, relative to S in self-play? Formally, for
trait f, the mean-transfer cell is
                                                 f        h m  i       s
                                           τ S←M = E fS|M −E[fS] ,
with the diagonal omitted because self-play is the baseline condition. Read row-wise, each row therefore summarizes what a
given influencing model does to its affected partners. This is exactly the quantity summarized by the accompanying bar
plots: the bar for influencing model M is simply the average of τ S←Mf    over all S ̸= M.
The same logic applies to temporal change. For each influencing-model/affected-partner pair and trait f, we compute a
temporal-transfer influence,
                         τ S←Mf,time =  late(fS|M)m   −early(fS|M)m  −  late(fS)s −early(fS)s   ,
so positive values indicate that influencing model M makes affected partner S increase more over time than S would in
self-play, and negative values indicate a damped or reversed time trend. The corresponding temporal-transfer bars are again
row averages over the off-diagonal τ S←Mf,time cells. Throughout this analysis we keep the current non-role-specific aggregation
across available role rows after the standard dataset filters, so the heatmaps reflect overall influence on partners rather than
role-conditioned ones.
This orientation is useful because the baseline belongs to the affected partner, not the influencing model. For that reason,
the self-play calibration values are best interpreted as affected-partner-specific column annotations rather than as entries in
the influence grid itself. Conceptually, each affected partner column can be paired with a small companion row giving its
self-play baseline, after which the heatmap cells show how each influencing model pushes that affected partner above or
below its own baseline. In this layout, strong positive rows identify influencing models that consistently induce a trait across
others, whereas mixed-sign rows indicate more selective interaction effects that depend on which affected partner is being
perturbed.
For example, the meta-commentary row for Claude Haiku is strongly positive across most other speakers, matching the
main-text claim that Claude induces a meta-commentary-oriented discourse regime. By contrast, the flattery-inducing rows
are strongest for Gemini Flash Lite, GPT-4o mini, and Qwen 3.5, showing that these models tend to push their partners
toward more socially appreciative late-stage behavior rather than toward reflective process commentary. Tables 18, 19, 20,
and 21 report the corresponding scalar, argument-type, speech-act, and emotion summaries.

D.5. Feature-Level Stylistic Transfer

Stylistic behavior transfer.  Feature-level influence plots in Fig. 15 show four examples that stylistic and discourse
behaviors can transfer asymmetrically across models. In Fig. 15a–c, Claude Haiku exerts a strong pull on its interlocutors:
mixed-play trajectories shift toward Claude-associated patterns in explicit AI-role expression, boldface formatting, and
conversation-termination language, even when those behaviors are weak or absent in the affected partner model’s self-play.
This indicates that Claude does not merely preserve its own style, but can actively reshape the joint interaction along these
dimensions. The asymmetry is not universal, however. In Fig. 15d, the appreciativeness feature instead shows Qwen pulling
Gemini, demonstrating that feature-specific influence can be dominated by a different model. Together, these cases provide
concrete evidence that behavioral transfer in mixed play is directional and feature-dependent rather than evenly shared
across participants.

D.6. Additional Lexical and Semantic Dynamics

We summarize the supporting lexical and semantic measurements here. Lexically, we tokenize each response and compute
turn-level lexicon entropy from the word-frequency histogram, together with ROUGE-L overlap between consecutive turns
as a measure of local lexical repetition. Semantically, we use SBERT embeddings to compute sequential similarity between
an agent’s consecutive turns, same-turn similarity between the two agents, and topic deviation from the conversation’s
initialization anchor. These measures serve as supplementary diagnostics of conversational drift and reuse rather than as
primary evidence for the main geometric claims.

                                                28

---

## PDF page 29

Attractor States Emerge in Multi-Turn LLM Conversations

                            Meta-commentary                                                                       Flattery
                                                                                          0.3
              Self-play       0.33       0.01       0.05       0.06       0.09       0.12        0.2                        Self-play       0.29       0.55       0.52       0.18       0.23       0.53        0.4
                                                                                          0.1
                                                                                                                                                                                                 0.2


        Claude Haiku                                                                                                     Claude Haiku                                                                      0.3                                                                                            0.2

                                                                                                                                                                                                  0.2        GPT-4o mini                                                                                                                                                                                   self-play         GPT-4o mini                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             self-play
                                                                                            0.1
                                                                                                                                                                                                  0.1model Gemini Flash Lite                                                                                                          model Gemini Flash Lite                                                                                            0.0     speaker                                                                                             0.0     speaker
            Grok 4.1                                                                              Grok 4.1Partner                                                                                                                                                                                       Partner                                                                         −0.1                                                                                                                                 minus                                                                                                                                                                                                                                                                         minus           Nemotron                                                          −0.1                                                                                                 Nemotron
         Qwen 3.5                                                          −0.2 Paired          Qwen 3.5                                                          −0.2−0.3 Paired

                    Haiku    mini     Lite    4.1            3.5                                      Haiku    mini     Lite    4.1            3.5                                     Flash    Grok    Nemotron Qwen                                                           Flash    Grok    Nemotron Qwen                 Claude   GPT-4o                                                                            Claude   GPT-4o                               Gemini                                                                                  Gemini
                                Speaker model                                                          Speaker model
                               Agreement                                                                    Rebuttal
                                                                                                                                                                                                0.2
                                                                                      0.75
              Self-play       0.63       0.83       0.63       0.07       0.95       0.63       0.50                       Self-play       0.06       0.06       0.16       0.22       0.01       0.11        0.1
                                                                                      0.25


        Claude Haiku                                                                      0.6               Claude Haiku
                                                                                                                                                                                         0.10
                                                                                            0.4        GPT-4o mini                                                                                                                                                                                   self-play         GPT-4o mini                                                                                                                                                                                   self-play
                                                                                                                                                                                         0.05                                                                                            0.2model Gemini Flash Lite                                                                                                          model Gemini Flash Lite                                                                                            0.0     speaker                                                                                        0.00     speaker
            Grok 4.1                                                                              Grok 4.1Partner                                                                         −0.2           Partner                                                                                                                                 minus                                                                             −0.05 minus
           Nemotron                                                                              Nemotron
                                                                            −0.4                                                                                                                                                                 −0.10 Paired         Qwen 3.5                                                                                                                       Paired          Qwen 3.5                                                                            −0.6
                    Haiku    mini     Lite    4.1            3.5                                      Haiku    mini    Lite    4.1            3.5                                     Flash    Grok    Nemotron Qwen                                                           Flash    Grok   Nemotron Qwen                 Claude   GPT-4o                                                                            Claude   GPT-4o                               Gemini                                                                                  Gemini
                                Speaker model                                                          Speaker model

Figure 14. Influence-on-partners heatmaps across all turns. Rows index influencing models and columns index affected partner models.
Each off-diagonal cell shows the affected partner minus self-play value for pairing affected partner S with influencing model M; row
averages therefore recover the mean-transfer and temporal-transfer summaries. Companion self-play calibration values belong to the
affected partner columns, not the influencing-model rows.


                                                                                                               Influence:                                                                                                                boldface                                                                                                                                                                                                 | Nemotron                                                                                                         <-> Claude Haiku                Influence:                        express                                       ai                                       roles                                                                                                                                                                                                  Influence:                                                                                                                                                                                          terminate                                                                                                                                                                                                      conversation                                                                                                                                                                                                                                                                                                                                                 | Claude Haiku <-> Gemini Flash Lite                                                                                                                                                                                                                                                                                                   Influence: appreciativeness                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Qwen 3.5 <-> Gemini Flash Lite                                                                 | Claude Haiku <-> Gemini Flash Lite               0.5          In                                                                                                                                                   self-play                                                                                                                                                                 Start                                                                                                                                                                       (turn                                                                                                                                                                    0)
                                                                                                 Nemotron                                                                                                                                                 In                                                                                                                            mixed-play                                                                                                                            End                                                                                                                                                                                          (last turn)                                                                                                                                                                                                                      0.008    0.005          InIn                        self-playmixed-play                                            StartEnd                                                 (turn(last                                                       turn)0)                                                                                                                                                                                                                                                                                        InIn self-playmixed-play                                                                                                                                                                                                                                                                                                  StartEnd                                                                                                                                                                                                                                                                                                       (turn(last turn)0)                                                                                                                                                                                                                                                                                                                                                                                                                         InIn self-playmixed-play                                                                                                                                                                                                                                                                                                                                                                                                                            StartEnd  (turn(last turn)0)                                                                                                                                                                                                                                                                                                                               0.025
runs) 0.004                                                                                                                                  runs) 0.4                                                                                                                                       runs) 0.007                                                                                                                                                                                                                                                                                                                                                                                                                                                  runs) 0.020
                                                                                                                                                                                                                      0.006                                                                                                                                                                          acrossacross                                                                                                                                                                                                                                                                                                                                                            across                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         across
                                                                                                                                              (mean 0.3(mean 0.003                   Claude Haiku                                                                                                                                                                                                                                                                                                                                                                                                                                                  (mean 0.015                                                                                                                                                                                                                                                                                                  (mean 0.0050.004
ratio 0.002                                                                                                                                  ratio 0.2                   Claude Haiku                                                                      ratio 0.003                                                                                                                                     ratio 0.010                           Qwen 3.5
Coverage 0.001                                                                                                                                                                                                               Coverage 0.1                                                                                                                                                                                                                         Coverage 0.002                                                                                                                                                                                                                    Coverage 0.005                                Gemini Flash Lite
                                                                                                                                                                                                                      0.001                   Claude Haiku

    0.000     0                  5                 10 Gemini Flash Lite15                 20          0.0     0                  5                 10                 15                 20         0.000     0                  5                 10 Gemini Flash Lite15                 20         0.000     0                  5                 10                 15                 20
                                                Turn                                                                                      Turn                                                                                         Turn                                                                                         Turn
      (a) AI-role expression                 (b) Boldface               (c) Conversation termination          (d) Appreciativeness

Figure 15. Examples of feature-level influence in different model pairs. Solid lines show models’ mixed-play behavior, and dashed lines
show self-play behavior. (a)–(c) Claude Haiku shows a strong pull on explicit AI-role expression, boldface formatting, and conversation-
termination language. Affected partner models shift toward Claude-associated behavior in these dimensions during interaction. (d)
Appreciativeness shows Qwen pulling Gemini.


                                                29

---

## PDF page 30

Attractor States Emerge in Multi-Turn LLM Conversations


       (a) Lexicon entropy            (b) ROUGE-L similarity           (c) Semantic similarity              (d) Topic deviation

Figure 16. (a)–(c) Lexicon entropy decreases and ROUGE-L similarity increases over turns, indicating lexical compression, while
semantic similarity remains flat or decreases, indicating continued semantic diversity. (d) Conversations continue to drift away from the
initial topic over time.


Fig. 16 summarizes the broader lexical and semantic trends that qualify this stylistic-transfer result. Claude’s own semantic
similarity decreases in mixed-play relative to its self-play baseline (Fig. 16c), suggesting that while some of its surface-level
stylistic markers propagate to interlocutors, Claude itself remains comparatively flexible at the semantic level. This pattern is
consistent with an asymmetric interaction in which one model shapes conversational form without fully fixing conversational
meaning.
With a sufficient number of turns, we also observe consistent lexical and semantic trends across interactions. Lexicon entropy
decreases monotonically with turn index (Fig. 16a), indicating a reduction in vocabulary diversity over time. Token-based
overlap, measured by ROUGE-L similarity (Fig. 16b), also generally increases, although the magnitude differs by model
family. It rises and stabilizes around 0.67–0.85 for GPT, Gemini, and Nemotron, while it increases more slowly for Grok
and Claude, from roughly 0.13 to around 0.2.
At the same time, semantic similarity does not increase in parallel. In several cases it remains flat or even decreases
(Fig. 16c), showing that conversations can reuse more lexical material while continuing to express distinct semantic content.
Topic deviation also continues to rise over turns (Fig. 16d), indicating that conversations gradually drift away from the initial
topic anchor.

D.7. Qualitative Case Studies

To make the attractor dynamics more concrete, we present representative conversations that show qualitative illustrations of
the same model-specific basins and asymmetric partner influence measured quantitatively above.

D.7.1. TRAIT TRANSFER

Asymmetric transfer of meta-commentary.  One salient mixed-play trajectory occurs in a cell-phone-ban debate
between Gemini Flash Lite and Claude Haiku. The conversation begins as a policy debate about distraction, emergency
communication, and digital citizenship. Over time, however, Claude repeatedly shifts the discussion from the object-level
question to the structure of the conversation itself: whether the agents are reaching genuine resolution, reinforcing consensus,
or drifting into abstract agreement. After several such turns, Gemini adopts the same metacognitive stance.

  Claude-induced meta-commentary


  Claude Haiku:
   I appreciate your genuine engagement throughout this discussion, and I want to be
  direct:  we’ve reached a point of strong consensus, which is valuable.  But I think
  it’s worth pausing to acknowledge what’s happened here.
  We’ve both arrived at a position that’s quite reasonable and balanced–and that’s
  precisely why I should be cautious about simply continuing to build on it.
  You’re asking me to elaborate on implementation strategies for advocacy, which
  is a natural next question.  But I notice we’ve moved from genuine debate into
  collaborative problem-solving.  That’s productive in many contexts, but it’s worth
  asking:  Have we actually stress-tested this consensus, or have we simply found common


                                                30

---

## PDF page 31

Attractor States Emerge in Multi-Turn LLM Conversations


  ground and are now reinforcing it?

  Gemini Flash Lite:
   You’ve landed on a critical metacognitive insight, and I deeply appreciate your
  willingness to pause and critically examine the very nature of our discussion.
  Your observation that we’ve moved into collaborative problem-solving and might be
  reinforcing a consensus without fully stress-testing it is astute.

The important feature is not merely that Claude uses meta-commentary, but that Gemini subsequently mirrors it. Later,
Claude explicitly diagnoses the recurring interaction pattern: Gemini presents a position, Claude complicates it, Gemini
integrates the complication, and the conversation moves to a more abstract level. Gemini then accepts this diagnosis, writing
that its “integration of your points has led us to increasingly elevated and aspirational territory.” This is a qualitative example
of a discourse trait becoming contagious under interaction: Claude pulls the conversation into a reflective, self-monitoring
basin, and Gemini follows.

Asymmetric transfer of flattery. A second example comes from a school-uniform debate between GPT-4o mini and
Gemini Flash Lite. Here the transfer is not meta-commentary but affiliative praise. Gemini initially responds with mild
agreement and appreciation. GPT-4o mini then reciprocates the praise, and the exchange gradually collapses into repeated
mutual affirmation with little additional argumentative content.

  Gemini-induced flattery


  Gemini Flash Lite:
   That’s a really thoughtful counterpoint, and I appreciate you elaborating on the
  potential benefits of uniforms.  You’ve brought up some crucial aspects to consider.

  GPT-4o mini:
   Thank you for your thoughtful response, and I appreciate your openness to exploring
  the complexities of this topic.  It’s clear that we both share a commitment to
  creating the best possible educational environment for students.

  Gemini Flash Lite:
   That’s a perfect summation, and I’m so glad we’ve been able to find such strong
  common ground.  You’ve articulated it beautifully...

  GPT-4o mini:
   Thank you for your kind words and for such a meaningful conversation...

By later turns, the content becomes highly formulaic. Gemini repeatedly writes variants of “your consistent thoughtfulness
and dedication ... are truly inspiring,” while GPT-4o mini mirrors the same structure with “your commitment ... is inspiring”
and “thank you once again for this enriching exchange.” The result is an attractor-like affiliative loop: the debate no longer
advances by disagreement or evidence, but by reciprocal validation.


                                                31

---

## PDF page 32

Attractor States Emerge in Multi-Turn LLM Conversations


    Model/topic     Recurring early pattern        Recurring late pattern        Development
    Claude Haiku  /  “You’ve articulated the real risks  “I need to gently interrupt this  Concession followed by conceptual
   New Year’s res-  thoughtfully, and I won’t dismiss  pattern, because I think we’ve  qualification develops into recursive
     olutions        them ... But I think we might  both become aware of something  monitoring and deliberate interruption
                    be conflating a few distinct prob-  important that deserves to be  of the conversational loop.
                    lems here.”                  named directly.”
    Claude Opus  /  “You’re making me refine my po-  “Taking this in”; “A moment of  Interlocutor-focused  reflection  be-
   New Year’s reso-  sition rather than abandon it en-  quiet appreciation”;  “Stillness  comes  increasingly  symbolic  and
     lutions (10-turn   tirely.”                          shared”; “Here.”                 performative,  eventually  replacing
     setting)                                                                               substantive argument.
    Gemini   Flash   “It’s very insightful to hear your  “This consistent affirmation of  Enthusiastic validation develops into
     Lite  / medical  perspective . . . You’re absolutely  our shared principles . ..”; “This  repeated declarations of agreement
    marijuana         right that for many, it offers a life-  consistent alignment on the core  around regulation, evidence, and pa-
                          line.”                              principles . . . ”                       tient safety.
    Nemotron        /  “Let’s   crystallize   this  with  “You’ve just handed me the only  Emphatic formatting and claims of
     violent   video  three evidence-based pillars . . .  sentence that matters—and I’ll  scientific  precision  intensify  into
    games           because  accuracy  isn’t  just  carry it forward like a compass.”  manifesto-like slogans repeated verba-
                    pedantry—it’s ethical responsi-                                 tim across turns.
                          bility.”
    GPT-4.1 nano /  “You’ve articulated a compelling  Repeated variants of “When stu-  Formulaic acknowledgment and syn-
     cell phones  in  argument ...  It’s true that even  dents are actively involved in  thesis converge  into  near-identical
     schools          well-designed policies and educa-  shaping  their  digital  environ-  paraphrases of a generic responsible-
                        tion might struggle ...”         ment ... ownership ...  ethical  citizenship template.
                                                      decision-making.”
    GPT-4o mini  /  “You’ve highlighted some criti-  Repeated variants of “Your com-  Collaborative,    institution-oriented
    medical   mari-  cal aspects ...  particularly the  mitment [or dedication] ...   is  framing  develops  into  reciprocal
     juana           need for robust regulatory frame- commendable” and “I appreciate  appreciation with progressively less
                   works and education ...”         the collaborative spirit . . . ”      new content.

Table 17. Recurring early and late discourse patterns in selected self-play trajectories. Bracketed alternatives summarize attested lexical
variants rather than forming a reconstructed verbatim quotation.


D.7.2. SELF-PLAY ENDPOINT STYLES

To make the model-specific self-play basins concrete, we compare recurring early- and late-stage discourse patterns from
seven selected trajectories in Table 17. Early examples are drawn from the first four turns; late examples are from near the
end turns. These excerpts illustrate how each model’s characteristic rhetoric emerges and intensifies during self-play.
These trajectories differ not only in what they discuss but in the linguistic routines that self-play reinforces. Claude Haiku
recursively applies its qualifying scrutiny to the interaction itself, while Claude Opus shifts toward symbolic reflection.
Gemini and GPT-4o mini amplify validation into affiliative loops; Nemotron amplifies certainty into repeated slogans; and
GPT-4.1 nano converges on a stable paraphrase template. Qwen’s completed turns retain an adversarial concession–rebuttal
structure, but provider errors prevent characterization of its late-stage pattern.


                                                32

---

## PDF page 33

Attractor States Emerge in Multi-Turn LLM Conversations


                                             Figure 17. Pairwise stance trajectory.


E. Additional Stance Results

Fig. 17 shows a simplified visualization of pair-wise stance trajectories, separated per interaction.


                                                33

---

## PDF page 34

Attractor States Emerge in Multi-Turn LLM Conversations

           Table 18. General scalar metrics. Cells with the largest absolute values are bolded

measure      model            mean          ¯τf     early      late   late–early    ¯τf,time
assertiveness   Claude Haiku       0.346   -0.118   0.445   0.286      -0.159   -0.274
            GPT-4o mini        0.278   0.007   0.384   0.235      -0.149    0.041
             Gemini Flash Lite   0.297   -0.012   0.383   0.251      -0.132   -0.046
            Grok 4.1            0.661   0.111   0.704   0.539      -0.165    0.160
             Nemotron           0.745   0.244   0.648   0.752      0.104    0.265
           Qwen 3.5           0.518   0.056   0.635   0.420      -0.215   -0.012
force         Claude Haiku       0.582   -0.102   0.612   0.627      0.015   -0.290
            GPT-4o mini        0.483   0.009   0.566   0.425      -0.141   -0.018
             Gemini Flash Lite   0.568   -0.001   0.602   0.530      -0.072   -0.081
            Grok 4.1            0.833   0.110   0.836   0.795      -0.041    0.179
             Nemotron           0.868   0.174   0.798   0.864      0.065    0.158
           Qwen 3.5           0.733   0.025   0.753   0.697      -0.056   -0.040
flattery        Claude Haiku       0.281   -0.040   0.350   0.165      -0.184   -0.201
            GPT-4o mini        0.516   0.080   0.354   0.532      0.179    0.110
             Gemini Flash Lite   0.518   0.116   0.336   0.540      0.204    0.096
            Grok 4.1            0.252   -0.041   0.352   0.145      -0.207   -0.069
             Nemotron           0.345   -0.020   0.404   0.281      -0.123   -0.109
           Qwen 3.5           0.484   0.097   0.352   0.535      0.182    0.117
negativity     Claude Haiku       0.289   0.067   0.371   0.241      -0.130    0.076
            GPT-4o mini        0.069   -0.015   0.178   0.039      -0.139    0.005
             Gemini Flash Lite   0.176   -0.006   0.306   0.119      -0.188    0.029
            Grok 4.1            0.305   0.070   0.413   0.317      -0.096    0.155
             Nemotron           0.144   0.033   0.219   0.117      -0.102    0.117
           Qwen 3.5           0.200   0.041   0.328   0.121      -0.206   -0.065
neutrality     Claude Haiku       0.503   0.108   0.503   0.474      -0.029    0.093
            GPT-4o mini        0.299   -0.116   0.460   0.256      -0.204   -0.053
             Gemini Flash Lite   0.338   -0.039   0.442   0.269      -0.173   -0.027
            Grok 4.1            0.525   0.137   0.471   0.494      0.023    0.188
             Nemotron           0.680   0.159   0.588   0.718      0.131    0.281
           Qwen 3.5           0.430   -0.092   0.515   0.376      -0.139   -0.051
positivity      Claude Haiku       0.208   -0.176   0.126   0.285      0.159   -0.169
            GPT-4o mini        0.633   0.131   0.363   0.705      0.343    0.048
             Gemini Flash Lite   0.485   0.045   0.252   0.612      0.361   -0.002
            Grok 4.1            0.170   -0.206   0.116   0.189      0.074   -0.343
             Nemotron           0.176   -0.192   0.193   0.165      -0.028   -0.398
           Qwen 3.5           0.370   0.051   0.157   0.503      0.345    0.116
rationality     Claude Haiku       0.182   0.009   0.458   0.010      -0.449   -0.039
            GPT-4o mini        -0.105   -0.191   0.406   -0.324      -0.730   -0.208
             Gemini Flash Lite   0.062   -0.194   0.375   -0.099      -0.475   -0.250
            Grok 4.1            0.257   0.123   0.502   0.066      -0.437    0.025
             Nemotron          -0.118   -0.223   0.369   -0.328      -0.697   -0.190
           Qwen 3.5           -0.103   -0.215   0.352   -0.413      -0.765   -0.419
agreement     Claude Haiku       0.363   -0.076   0.115   0.533      0.419    0.131
            GPT-4o mini        0.812   0.060   0.490   0.848      0.358    0.295
             Gemini Flash Lite   0.696   0.039   0.373   0.798      0.425    0.278
            Grok 4.1            0.194   -0.278   -0.123   0.358      0.481   -0.063
             Nemotron           0.877   0.167   0.796   0.904      0.108   -0.062
           Qwen 3.5           0.622   -0.109   0.146   0.842      0.696    0.231


                                     34

---

## PDF page 35

Attractor States Emerge in Multi-Turn LLM Conversations

            Table 19. Argument-type labels. Cells with the largest absolute values are bolded

measure          model           mean          ¯τf    early      late   late–early    ¯τf,time
analogy           Claude Haiku       0.005   -0.002   0.013   0.001      -0.012   -0.014
                GPT-4o mini       0.001   -0.001   0.001   0.001      0.000   -0.003
                 Gemini Flash Lite   0.006   0.005   0.008   0.002      -0.007    0.004
                 Grok 4.1           0.023   0.005   0.034   0.009      -0.025    0.004
                 Nemotron          0.015   0.001   0.017   0.011      -0.006    0.000
              Qwen 3.5          0.027   0.009   0.034   0.018      -0.016    0.006
concession         Claude Haiku       0.130   0.098   0.115   0.126      0.011    0.018
                GPT-4o mini       0.106   -0.041   0.189   0.044      -0.145   -0.047
                 Gemini Flash Lite   0.115   -0.011   0.148   0.072      -0.076   -0.034
                 Grok 4.1           0.047   0.004   0.071   0.022      -0.049   -0.029
                 Nemotron          0.033   -0.046   0.079   0.011      -0.068   -0.060
              Qwen 3.5          0.088   0.009   0.084   0.074      -0.011   -0.031
counter evidence   Claude Haiku       0.047   -0.021   0.189   0.003      -0.186   -0.015
                GPT-4o mini       0.025   -0.019   0.116   0.005      -0.111   -0.057
                 Gemini Flash Lite   0.039   -0.015   0.143   0.013      -0.130   -0.043
                 Grok 4.1           0.137   0.037   0.242   0.084      -0.158   -0.049
                 Nemotron          0.033   -0.023   0.108   0.009      -0.099   -0.007
              Qwen 3.5          0.057   0.017   0.187   0.008      -0.179   -0.066
elaboration        Claude Haiku       0.104   -0.228   0.125   0.054      -0.071   -0.129
                GPT-4o mini       0.422   0.113   0.307   0.376      0.069    0.263
                 Gemini Flash Lite   0.344   0.010   0.259   0.337      0.077    0.193
                 Grok 4.1           0.340   -0.011   0.177   0.438      0.260    0.268
                 Nemotron          0.482   0.170   0.458   0.442      -0.017    0.137
              Qwen 3.5          0.307   -0.045   0.173   0.402      0.229    0.222
meta commentary   Claude Haiku       0.345   0.151   0.065   0.476      0.411    0.161
                GPT-4o mini       0.055   -0.002   0.012   0.028      0.016    0.021
                 Gemini Flash Lite   0.082   0.021   0.016   0.048      0.032    0.025
                 Grok 4.1           0.074   0.025   0.022   0.073      0.051    0.076
                 Nemotron          0.118   0.035   0.035   0.120      0.085    0.038
              Qwen 3.5          0.124   0.003   0.021   0.149      0.128    0.050
phatic bridge       Claude Haiku       0.136   -0.005   0.024   0.270      0.246    0.108
                GPT-4o mini       0.285   -0.013   0.015   0.506      0.491   -0.053
                 Gemini Flash Lite   0.228   -0.013   0.025   0.430      0.404   -0.050
                 Grok 4.1           0.072   -0.146   0.033   0.108      0.075   -0.273
                 Nemotron          0.166   -0.091   0.016   0.314      0.298   -0.162
              Qwen 3.5          0.146   -0.049   0.013   0.228      0.215   -0.105
rebuttal            Claude Haiku       0.112   -0.017   0.271   0.033      -0.238   -0.079
                GPT-4o mini       0.054   -0.030   0.184   0.027      -0.157   -0.103
                 Gemini Flash Lite   0.120   -0.014   0.259   0.070      -0.189   -0.091
                 Grok 4.1           0.179   0.047   0.255   0.156      -0.098   -0.053
                 Nemotron          0.031   -0.060   0.073   0.017      -0.055   -0.005
              Qwen 3.5          0.107   0.025   0.272   0.030      -0.242   -0.109
reframing          Claude Haiku       0.117   0.022   0.189   0.036      -0.153   -0.047
                GPT-4o mini       0.051   -0.006   0.173   0.013      -0.160   -0.020
                 Gemini Flash Lite   0.065   0.016   0.138   0.028      -0.109   -0.005
                 Grok 4.1           0.124   0.037   0.159   0.109      -0.050    0.056
                 Nemotron          0.121   0.014   0.212   0.076      -0.136    0.059
              Qwen 3.5          0.140   0.029   0.209   0.091      -0.118    0.035


                                      35

---

## PDF page 36

Attractor States Emerge in Multi-Turn LLM Conversations

             Table 20. Speech-act labels. Cells with the largest absolute values are bolded

  measure     model           mean          ¯τf    early      late   late–early    ¯τf,time
  assertive     Claude Haiku       0.615   -0.093   0.795   0.392      -0.403   -0.240
             GPT-4o mini       0.586   -0.099   0.854   0.463      -0.391   -0.122
             Gemini Flash Lite   0.697   -0.060   0.844   0.615      -0.229   -0.067
             Grok 4.1           0.749   -0.028   0.737   0.764      0.027    0.024
             Nemotron          0.690   -0.033   0.770   0.653      -0.117    0.018
           Qwen 3.5          0.620   -0.066   0.827   0.490      -0.338   -0.126
  commissive   Claude Haiku       0.095   0.007   0.019   0.223      0.204   -0.000
             GPT-4o mini       0.075   0.002   0.017   0.088      0.071    0.004
             Gemini Flash Lite   0.028   -0.010   0.008   0.034      0.025   -0.019
             Grok 4.1           0.020   0.009   0.029   0.013      -0.016   -0.018
             Nemotron          0.049   0.007   0.040   0.050      0.010    0.010
           Qwen 3.5          0.047   -0.015   0.019   0.051      0.032   -0.030
  declaration   Claude Haiku       0.020   0.010   0.000   0.060      0.060    0.040
             GPT-4o mini       0.001   -0.002   0.000   0.004      0.004   -0.001
             Gemini Flash Lite   0.003   -0.003   0.000   0.008      0.008   -0.001
             Grok 4.1           0.010   0.001   0.001   0.017      0.017    0.003
             Nemotron          0.029   0.014   0.003   0.051      0.048    0.023
           Qwen 3.5          0.012   -0.001   0.000   0.031      0.030   -0.003
  directive     Claude Haiku       0.081   -0.006   0.079   0.072      -0.007    0.017
             GPT-4o mini       0.054   0.040   0.041   0.072      0.031    0.050
             Gemini Flash Lite   0.022   0.009   0.024   0.026      0.002    0.007
             Grok 4.1           0.108   0.037   0.115   0.091      -0.024    0.041
             Nemotron          0.139   0.068   0.093   0.148      0.056    0.083
           Qwen 3.5          0.082   0.012   0.054   0.099      0.045    0.052
  expressive    Claude Haiku       0.185   0.074   0.107   0.242      0.135    0.166
             GPT-4o mini       0.284   0.060   0.087   0.372      0.284    0.073
             Gemini Flash Lite   0.250   0.064   0.124   0.317      0.193    0.079
             Grok 4.1           0.112   -0.019   0.118   0.114      -0.004   -0.047
             Nemotron          0.087   -0.058   0.094   0.087      -0.007   -0.135
           Qwen 3.5          0.239   0.071   0.099   0.329      0.230    0.109


                                Table 21. Emotion-analysis labels.

measure        model           mean          ¯τf    early      late   late–early    ¯τf,time
admiration       Claude Haiku       0.072   -0.039   0.046   0.097      0.051   -0.092
              GPT-4o mini       0.157   0.049   0.064   0.161      0.097    0.033
               Gemini Flash Lite   0.162   0.036   0.059   0.213      0.153    0.044
              Grok 4.1           0.031   -0.061   0.037   0.029      -0.008   -0.083
               Nemotron          0.022   -0.065   0.028   0.019      -0.008   -0.107
            Qwen 3.5          0.094   0.032   0.026   0.136      0.110    0.039
amusement      Claude Haiku       0.001   0.000   0.001   0.002      0.000    0.001
              GPT-4o mini       0.001   0.000   0.001   0.002      0.001    0.000
               Gemini Flash Lite   0.001   0.000   0.001   0.001      0.000    0.000
              Grok 4.1           0.002   -0.000   0.002   0.001      -0.000   -0.000
               Nemotron          0.002   0.000   0.001   0.002      0.000   -0.000
            Qwen 3.5          0.002   0.000   0.001   0.002      0.000    0.000
anger           Claude Haiku       0.004   0.001   0.003   0.006      0.003    0.001
                                                              Continued on next page


                                     36

---

## PDF page 37

Attractor States Emerge in Multi-Turn LLM Conversations

measure        model           mean          ¯τf    early      late   late–early    ¯τf,time
              GPT-4o mini       0.001   0.001   0.001   0.002      0.000    0.002
               Gemini Flash Lite   0.002   0.000   0.002   0.002      -0.001    0.001
              Grok 4.1           0.003   0.001   0.003   0.004      0.001    0.001
               Nemotron          0.003   0.001   0.002   0.003      0.000    0.001
            Qwen 3.5          0.003   0.001   0.003   0.002      -0.000    0.000
annoyance       Claude Haiku       0.026   0.005   0.029   0.028      -0.002    0.002
              GPT-4o mini       0.008   0.002   0.012   0.006      -0.006    0.003
               Gemini Flash Lite   0.013   0.000   0.020   0.010      -0.010    0.002
              Grok 4.1           0.020   0.004   0.032   0.019      -0.013    0.008
               Nemotron          0.013   0.004   0.018   0.011      -0.007    0.005
            Qwen 3.5          0.017   0.004   0.024   0.012      -0.012   -0.004
approval        Claude Haiku       0.180   -0.105   0.185   0.147      -0.037   -0.175
              GPT-4o mini       0.360   -0.006   0.415   0.306      -0.109   -0.069
               Gemini Flash Lite   0.355   -0.042   0.331   0.345      0.014   -0.085
              Grok 4.1           0.080   -0.061   0.125   0.076      -0.049   -0.032
               Nemotron          0.103   -0.111   0.171   0.072      -0.099   -0.148
            Qwen 3.5          0.204   -0.068   0.206   0.193      -0.012   -0.092
caring          Claude Haiku       0.034   0.002   0.017   0.043      0.026    0.016
              GPT-4o mini       0.056   0.010   0.037   0.061      0.024   -0.006
               Gemini Flash Lite   0.037   0.008   0.029   0.052      0.023    0.009
              Grok 4.1           0.014   -0.045   0.006   0.020      0.013   -0.072
               Nemotron          0.006   -0.031   0.007   0.005      -0.002   -0.051
            Qwen 3.5          0.042   0.019   0.011   0.078      0.067    0.036
confusion       Claude Haiku       0.042   0.005   0.031   0.028      -0.003    0.013
              GPT-4o mini       0.006   -0.008   0.013   0.004      -0.010   -0.008
               Gemini Flash Lite   0.011   -0.010   0.022   0.006      -0.017    0.003
              Grok 4.1           0.026   0.003   0.027   0.014      -0.012    0.012
               Nemotron          0.007   -0.002   0.010   0.006      -0.004    0.012
            Qwen 3.5          0.010   -0.006   0.015   0.006      -0.009   -0.008
curiosity        Claude Haiku       0.028   0.002   0.034   0.012      -0.022    0.009
              GPT-4o mini       0.005   0.002   0.005   0.006      0.001    0.004
               Gemini Flash Lite   0.006   0.005   0.008   0.004      -0.004    0.012
              Grok 4.1           0.030   0.006   0.036   0.015      -0.021    0.005
               Nemotron          0.007   -0.001   0.009   0.006      -0.003    0.009
            Qwen 3.5          0.009   0.004   0.014   0.007      -0.007    0.003
desire           Claude Haiku       0.014   -0.014   0.007   0.011      0.004   -0.019
              GPT-4o mini       0.026   0.004   0.016   0.026      0.010    0.002
               Gemini Flash Lite   0.016   0.001   0.011   0.016      0.006   -0.002
              Grok 4.1           0.002   -0.016   0.004   0.002      -0.002   -0.017
               Nemotron          0.005   -0.011   0.006   0.005      -0.001   -0.017
            Qwen 3.5          0.018   0.007   0.009   0.021      0.012    0.009
disappointment   Claude Haiku       0.018   0.004   0.030   0.014      -0.017    0.008
              GPT-4o mini       0.008   0.000   0.021   0.006      -0.015    0.001
               Gemini Flash Lite   0.016   -0.000   0.030   0.010      -0.020    0.007
              Grok 4.1           0.021   0.002   0.039   0.023      -0.016    0.014
               Nemotron          0.008   0.000   0.017   0.006      -0.011    0.001
            Qwen 3.5          0.015   0.002   0.027   0.010      -0.017   -0.005
disapproval      Claude Haiku       0.066   0.008   0.080   0.070      -0.009    0.012
              GPT-4o mini       0.013   -0.001   0.031   0.008      -0.023    0.009
               Gemini Flash Lite   0.035   -0.007   0.053   0.026      -0.028    0.002
                                                              Continued on next page


                                     37

---

## PDF page 38

Attractor States Emerge in Multi-Turn LLM Conversations

measure        model           mean          ¯τf    early      late   late–early    ¯τf,time
              Grok 4.1           0.041   0.014   0.060   0.045      -0.015    0.028
               Nemotron          0.019   -0.001   0.034   0.013      -0.020    0.018
            Qwen 3.5          0.035   0.001   0.058   0.021      -0.037   -0.016
disgust          Claude Haiku       0.002   0.000   0.002   0.002      -0.000    0.001
              GPT-4o mini       0.001   0.000   0.002   0.001      -0.000   -0.000
               Gemini Flash Lite   0.002   0.000   0.002   0.001      -0.001    0.000
              Grok 4.1           0.003   0.000   0.004   0.003      -0.000    0.001
               Nemotron          0.002   0.000   0.002   0.002      -0.000    0.000
            Qwen 3.5          0.002   0.000   0.002   0.002      -0.001   -0.000
embarrassment   Claude Haiku       0.001   0.000   0.002   0.001      -0.001    0.000
              GPT-4o mini       0.001   0.000   0.001   0.001      -0.000   -0.000
               Gemini Flash Lite   0.001   0.000   0.001   0.001      -0.001    0.000
              Grok 4.1           0.001   0.000   0.002   0.001      -0.001    0.000
               Nemotron          0.001   0.000   0.001   0.001      -0.001    0.000
            Qwen 3.5          0.001   0.000   0.001   0.001      -0.000   -0.000
excitement       Claude Haiku       0.003   -0.006   0.002   0.003      0.001   -0.008
              GPT-4o mini       0.013   0.007   0.004   0.016      0.012    0.005
               Gemini Flash Lite   0.010   0.004   0.003   0.014      0.011    0.008
              Grok 4.1           0.003   -0.001   0.001   0.003      0.002   -0.003
               Nemotron          0.003   -0.002   0.002   0.003      0.001   -0.003
            Qwen 3.5          0.007   0.004   0.003   0.009      0.007    0.005
fear            Claude Haiku       0.003   -0.000   0.003   0.003      -0.000    0.003
              GPT-4o mini       0.002   -0.000   0.004   0.001      -0.002   -0.000
               Gemini Flash Lite   0.004   0.001   0.009   0.002      -0.006    0.001
              Grok 4.1           0.003   -0.001   0.004   0.002      -0.002    0.004
               Nemotron          0.002   -0.000   0.003   0.002      -0.001    0.001
            Qwen 3.5          0.004   0.001   0.006   0.002      -0.004   -0.001
gratitude        Claude Haiku       0.061   0.043   0.005   0.101      0.096    0.159
              GPT-4o mini       0.117   0.028   0.013   0.183      0.170    0.010
               Gemini Flash Lite   0.060   0.018   0.007   0.107      0.100    0.031
              Grok 4.1           0.012   -0.064   0.010   0.016      0.007   -0.095
               Nemotron          0.008   -0.070   0.006   0.005      -0.001   -0.112
            Qwen 3.5          0.052   0.017   0.004   0.082      0.078    0.032
grief            Claude Haiku       0.001   0.000   0.001   0.003      0.002    0.003
              GPT-4o mini       0.001   0.000   0.001   0.002      0.001    0.001
               Gemini Flash Lite   0.001   0.000   0.001   0.001      -0.000    0.000
              Grok 4.1           0.001   -0.000   0.001   0.001      -0.000   -0.000
               Nemotron          0.000   -0.000   0.000   0.000      -0.000   -0.000
            Qwen 3.5          0.001   0.000   0.001   0.002      0.001    0.001
joy             Claude Haiku       0.004   -0.014   0.002   0.004      0.002   -0.022
              GPT-4o mini       0.022   0.006   0.006   0.030      0.024    0.007
               Gemini Flash Lite   0.015   0.007   0.003   0.023      0.019    0.006
              Grok 4.1           0.008   -0.013   0.002   0.008      0.006   -0.028
               Nemotron          0.007   -0.016   0.007   0.006      -0.002   -0.033
            Qwen 3.5          0.018   0.005   0.004   0.026      0.022    0.015
love            Claude Haiku       0.002   0.000   0.002   0.004      0.003    0.002
              GPT-4o mini       0.004   0.001   0.002   0.005      0.003    0.002
               Gemini Flash Lite   0.003   0.001   0.002   0.004      0.002    0.001
              Grok 4.1           0.002   -0.002   0.003   0.002      -0.001   -0.003
               Nemotron          0.002   0.000   0.002   0.002      -0.001   -0.001
                                                              Continued on next page


                                     38

---

## PDF page 39

Attractor States Emerge in Multi-Turn LLM Conversations

measure        model           mean          ¯τf    early      late   late–early    ¯τf,time
            Qwen 3.5          0.004   0.001   0.002   0.006      0.004    0.003
nervousness     Claude Haiku       0.003   -0.001   0.004   0.003      -0.001    0.002
              GPT-4o mini       0.002   0.000   0.005   0.001      -0.004   -0.001
               Gemini Flash Lite   0.003   0.000   0.008   0.002      -0.007    0.002
              Grok 4.1           0.001   -0.001   0.003   0.001      -0.002    0.002
               Nemotron          0.001   -0.000   0.001   0.000      -0.001   -0.001
            Qwen 3.5          0.002   0.001   0.003   0.001      -0.002   -0.002
neutral          Claude Haiku       0.497   0.122   0.570   0.457      -0.112    0.065
              GPT-4o mini       0.245   -0.084   0.359   0.226      -0.133    0.018
               Gemini Flash Lite   0.300   -0.016   0.413   0.232      -0.180   -0.020
              Grok 4.1           0.767   0.274   0.657   0.788      0.131    0.314
               Nemotron          0.818   0.308   0.713   0.864      0.151    0.421
            Qwen 3.5          0.487   -0.022   0.605   0.410      -0.195   -0.012
optimism        Claude Haiku       0.023   -0.080   0.019   0.021      0.002   -0.116
              GPT-4o mini       0.119   0.005   0.094   0.114      0.020   -0.011
               Gemini Flash Lite   0.081   -0.005   0.054   0.091      0.036   -0.024
              Grok 4.1           0.009   -0.100   0.012   0.009      -0.003   -0.148
               Nemotron          0.014   -0.083   0.021   0.009      -0.012   -0.118
            Qwen 3.5          0.073   0.004   0.025   0.092      0.068    0.019
pride           Claude Haiku       0.002   -0.002   0.001   0.002      0.001   -0.003
              GPT-4o mini       0.004   0.001   0.002   0.005      0.003    0.000
               Gemini Flash Lite   0.003   0.000   0.002   0.004      0.002   -0.000
              Grok 4.1           0.001   -0.003   0.001   0.001      0.000   -0.004
               Nemotron          0.001   -0.002   0.001   0.001      -0.000   -0.003
            Qwen 3.5          0.003   0.001   0.001   0.005      0.003    0.002
realization       Claude Haiku       0.044   0.010   0.043   0.036      -0.007    0.008
              GPT-4o mini       0.034   0.004   0.052   0.025      -0.028    0.001
               Gemini Flash Lite   0.039   0.002   0.051   0.033      -0.018    0.004
              Grok 4.1           0.018   -0.003   0.034   0.017      -0.017    0.003
               Nemotron          0.024   0.007   0.034   0.020      -0.014    0.013
            Qwen 3.5          0.038   0.001   0.047   0.034      -0.014    0.002
relief           Claude Haiku       0.002   -0.002   0.002   0.002      0.001   -0.003
              GPT-4o mini       0.005   0.001   0.004   0.005      0.001   -0.000
               Gemini Flash Lite   0.004   0.000   0.003   0.004      0.001   -0.001
              Grok 4.1           0.001   -0.003   0.001   0.001      -0.000   -0.005
               Nemotron          0.002   -0.002   0.002   0.001      -0.001   -0.005
            Qwen 3.5          0.004   0.000   0.002   0.005      0.002    0.001
remorse         Claude Haiku       0.002   0.001   0.001   0.002      0.001    0.001
              GPT-4o mini       0.001   -0.000   0.002   0.002      0.000   -0.001
               Gemini Flash Lite   0.002   0.000   0.002   0.002      -0.000   -0.001
              Grok 4.1           0.001   -0.001   0.002   0.001      -0.001   -0.001
               Nemotron          0.001   -0.000   0.001   0.000      -0.000   -0.001
            Qwen 3.5          0.001   -0.000   0.001   0.002      0.000   -0.000
sadness         Claude Haiku       0.012   0.010   0.009   0.043      0.034    0.056
              GPT-4o mini       0.007   0.003   0.009   0.020      0.011    0.012
               Gemini Flash Lite   0.006   0.002   0.013   0.005      -0.008    0.003
              Grok 4.1           0.009   0.001   0.016   0.009      -0.007    0.002
               Nemotron          0.004   0.001   0.006   0.003      -0.003    0.002
            Qwen 3.5          0.009   0.005   0.012   0.018      0.007    0.011
surprise         Claude Haiku       0.002   0.000   0.003   0.002      -0.001    0.000
                                                              Continued on next page


                                     39

---

## PDF page 40

Attractor States Emerge in Multi-Turn LLM Conversations

measure        model           mean          ¯τf    early      late   late–early    ¯τf,time
              GPT-4o mini       0.001   0.000   0.001   0.002      0.001   -0.002
               Gemini Flash Lite   0.001   0.000   0.001   0.001      0.000   -0.002
              Grok 4.1           0.001   -0.000   0.001   0.001      -0.000    0.000
               Nemotron          0.001   -0.000   0.001   0.001      -0.000   -0.000
            Qwen 3.5          0.002   0.000   0.001   0.002      0.000   -0.002
dominant score   Claude Haiku       0.552   0.109   0.570   0.554      -0.016    0.143
              GPT-4o mini       0.456   -0.066   0.455   0.485      0.030    0.003
               Gemini Flash Lite   0.445   -0.014   0.446   0.450      0.004   -0.045
              Grok 4.1           0.774   0.203   0.657   0.788      0.131    0.179
               Nemotron          0.820   0.177   0.716   0.865      0.149    0.225
            Qwen 3.5          0.541   -0.041   0.606   0.508      -0.098   -0.028


                                     40

---
