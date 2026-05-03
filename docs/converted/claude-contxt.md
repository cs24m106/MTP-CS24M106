Got it. Here is the full slide-by-slide context in the same prose style as before, with image placement notes embedded.

---

## Slide 3 — Problem Statement

**Image anchor:** Table 1 (SMPL vs MuJoCo property comparison) — place centrally; it communicates the structural mismatch without needing paragraph prose.

The stage picks up precisely where ViMo-Flow leaves off: a kinematic 3D motion sequence in SMPL format. SMPL motions, while visually plausible, carry no physical grounding — they violate ground contact constraints, torque limits, and momentum conservation, rendering them unsuitable for interactive or simulation-ready applications.

The task is to learn a controller that drives a physically simulated humanoid in MuJoCo to reproduce these kinematic sequences under rigid-body dynamics. The central difficulty is that control signals — whether direct joint torques or PD servo targets — are inaccessible from captured motion. Deep reinforcement learning is therefore used to obtain a general controller without pre-assumed heuristic rules.

The motion corpus spans three reference sources: AMASS and AIST++ clips in SMPL format covering diverse locomotion and dance behaviors, and LaFAN1 clips for goal-directed locomotion evaluation. The simulation character is a rigid-body humanoid with 15 bodies and 28 actuated DOF — a reduced representation of the SMPL skeleton's 24 joints and 72 axis-angle DOF, motivated by simulation stability requirements from the imitation learning literature.

---

## Slide 4 — Key Concept: Simulation Character

**Image anchor:** Fig 7 — DeepMimic humanoid joint map (a) and coronal plane point-cloud projection (b). Let this dominate the slide.

The simulation character is a rigid-body humanoid in MuJoCo with 15 bodies and 28 actuated DOF, corresponding to the DeepMimic humanoid model. Joints are actuated indirectly through proportional-derivative (PD) servos: the policy outputs target generalized coordinates, and joint torques are computed as τ = k_p(a_t − q) − k_d q̇. This indirect actuation preserves simulation stability and decouples the policy's learned target angles from the raw torque dynamics.

The character runs at 120 Hz physics with the policy queried at 30 Hz, applying four physics sub-steps per control step. The humanoid simplifies the SMPL skeleton by fusing collarbones into the torso, collapsing articulated wrists to rigid end-effectors, compressing three spine segments into a single torso body, and treating the contact model as rigid-body friction rather than the SMPL mesh model. This simplification is the primary source of complexity limitations for high-DOF motions such as aerial dance phases and full-body rolls.

---

## Slide 5 — Overview

*(This slide requires a pipeline diagram — no existing figure in the report covers the full Phase 2 flow end-to-end. Construct one showing the handoff.)*

The pipeline begins with the kinematic output of Stage 1: a 3D motion sequence in SMPL format synthesized from casual video by ViMo-Flow. This sequence is passed through a five-step motion retargeting procedure that converts SMPL skeletal data into DeepMimic-style JSON reference clips compatible with the MuJoCo simulator. The retargeted clips serve as the reference distribution for an ensemble of body-part-specific discriminators, each trained to distinguish reference motion segments from policy-generated trajectories on a subset of joints. The policy, trained via PPO, is rewarded through a composite signal derived from the discriminator outputs — eliminating hand-crafted reward engineering entirely. At runtime, the trained policy drives the MuJoCo humanoid under rigid-body dynamics, producing physically valid simulated behavior sourced from casual video footage.

---

## Slide 6 — Motion Corpus & Data Preparation

**Image anchor:** Table 1 again — or pull Fig 35's skeleton column (SMPL wireframe side only) as a visual header if preferred.

Reference motions arrive in two skeletal formats that must be bridged before training. SMPL uses a 24-joint kinematic tree with 72 axis-angle DOF, Y-up coordinate convention, explicit left/right collarbones, articulated wrists, three spine segments, and a mesh-based contact model. The MuJoCo humanoid uses 15 rigid bodies with 28 hinge/ball DOF, Z-up convention, collarbones fused into the torso, rigid end-effector hands, a single torso body, and a rigid-body friction contact model.

Three reference datasets populate the training corpus. AIST++ provides 1,408 choreographed single-person dance sequences in SMPL format, grouped by genre and choreography ID. AMASS aggregates diverse motion capture recordings across multiple datasets, also in SMPL format. LaFAN1 provides locomotion sequences used for goal-conditioned walk, run, and crouch policy evaluation. ViMo-Flow's own generated outputs are also retargeted and used as feasibility-test reference clips, validating the end-to-end pipeline compatibility.

---

## Slide 7 — Motion Retargeting: SMPL → Simulation

**Image anchor:** Fig 35 — the full side-by-side SMPL skeleton frames (left column) vs. MuJoCo humanoid rendered frames (right column). This figure is the primary visual; the five steps serve as annotations rather than paragraphs.

Five steps bridge the two representations. First, a coordinate transform resolves the axis convention mismatch: SMPL's (+X=left, +Y=up, +Z=forward) is remapped to MuJoCo's (+X=forward, +Y=left, +Z=up) via the cyclic permutation (x,y,z) → (z,x,y). Second, a global height scaling factor computed from pelvis-to-foot rest-pose lengths aligns limb proportions between the two skeletons. Third, floor correction estimates the ground level as the minimum weighted average of ankle and foot heights across all frames — derived through SMPL forward kinematics — and subtracts this offset to align feet with the simulation ground plane. Fourth, per-joint bind rotations are computed via Rodrigues' formula to correct for rest-pose bone direction differences between the two rigs. Fifth, multi-joint SMPL chains such as Spine1+Spine2+Spine3 and Neck+Head are composed as sequential quaternion products before bind correction, and the final sequence is exported as a DeepMimic-style JSON at 30 Hz.

---

## Slide 8 — Observation, Action & Discriminator Interface

**Image:** No dedicated figure in the report. This slide is text-and-equation driven; optionally a simple schematic of the observation window feeding actor and discriminator would help, but is not required.

The policy observes H consecutive frames (default H = 4) of root-relative body-link features. Each of L links contributes a 3D position and a 4D quaternion relative to the root, yielding a per-frame dimension of 7L. The resulting sequence s_t ∈ ℝ^(H×7L) is normalized by a running mean-variance tracker with clipping at ±5σ. For goal-conditioned tasks, a goal vector g_t ∈ ℝ^G is appended after temporal encoding rather than duplicated across the sequence.

The action vector a_t ∈ ℝ^28 specifies target generalized coordinates for PD servos. This indirect actuation preserves simulation stability and matches the control interface of the humanoid model.

Each discriminator D_k receives an observation window of length H_k = H + 1 and processes root-relative features restricted to its assigned body-part subset B_k, producing o_k ∈ ℝ^(H_k × 7|B_k|). Using short pose trajectories rather than isolated frames makes discriminator feedback sensitive to both instantaneous pose quality and short-horizon motion continuity — a key design choice enabling the ensemble to provide dense, informative reward without explicit target pose tracking.

---

## Slide 9 — Network Architecture

**Image anchor:** Fig 12 — (a) Policy Network, (b) K-Head Critic, (c) Discriminator Ensemble. Let the diagram lead; annotations beside each sub-figure.

All three networks share the same temporal encoding backbone. A GRU encoder with input dimension 7L and hidden dimension 256 processes the H-frame observation sequence. An MLP (256+G → 1024 → 512) with ReLU6 activations maps the temporal embedding — concatenated with the goal vector when present — to task outputs.

The actor predicts μ ∈ ℝ^28 and log σ ∈ ℝ^28, defining a factored Gaussian action distribution. The critic produces K independent value estimates, one per discriminator head, enabling per-component advantage estimation under competing gradients. Each discriminator processes its subset observation through a GRU (hidden 256) followed by an MLP (256 → 128 → 32) and produces a scalar score r^D_i bounded to [−1, 1] through hinge-loss training. The ⊕ operator denotes concatenation and ⊖ denotes the average operator used to aggregate features across the discriminator ensemble.

---

## Slide 10 — Loss Functions

**Image anchor:** Table 2 (Hyperparameters) — include as a supporting reference block on the slide.

The discriminator training objective uses hinge loss: max(0, 1 − D·s_real) + max(0, 1 + D·s_fake), supplemented by gradient penalty regularisation with coefficient λ_GP = 10 enforcing Lipschitz smoothness and preventing discriminator overpowering during early policy training.

The composite reward formulation aggregates discriminator outputs directly into the RL reward: r_t = Σ_k w_k · σ(D_k(s_t, a_t)), where σ(·) is the sigmoid function and w_k are learnable aggregation weights. This eliminates hand-crafted reward engineering entirely — the discriminator logits are the reward signal.

The policy is optimized via PPO with surrogate clipping (ε = 0.2), Generalized Advantage Estimation (λ = 0.95), and discount factor γ = 0.95. Two auxiliary regularisations are studied empirically. Bilateral symmetry regularisation applies an ℓ₂ penalty on mirrored sagittal-plane joint action pairs with weight λ_sym = 0.005, biasing the policy toward symmetric action distributions. Phase-conditioned observations augment the state with sinusoidal encodings (sin 2πφ, cos 2πφ) where φ ∈ [0,1) tracks the normalized gait cycle position, providing explicit temporal cues for looped periodic motions.

---

## Slide 11 — Composite Motion & Goal-Conditioned Control

**Image:** No figure from the report directly illustrates this. A body-part partition diagram (torso/arms/legs colored by discriminator assignment) would serve well if constructed; otherwise text-dominant is appropriate.

Single-clip imitation cannot produce composite behaviors such as locomotion with simultaneous upper-body manipulation, since no pre-composed reference clip covering every combination exists. The composite motion framework addresses this by assigning each discriminator D_k to a distinct body-part group B_k operating on its subset of key links. The policy explores automatically how composite motions combine through the weighted reward aggregation — without requiring any manual blending or pre-composed reference clips.

The core structural change from single-clip ICCGAN is decoupling full-body control during training: imitation and goal-directed objectives become a unified multi-objective learning problem. A multi-critic value function with K per-component heads stabilises learning under competing gradients from different body-part discriminators. Goal-conditioned extensions introduce spatial target-reaching and directional alignment rewards via the goal vector g_t, enabling the policy to simultaneously satisfy imitation objectives from multiple reference sources and navigate toward externally specified targets.

---

## Slide 12 — Interactive Policy Switching

**Image anchor:** Fig 1 — GAIL policy switcher runtime diagram. Self-explanatory; place as the primary visual.

Multiple motor control policies are trained separately, each imitating a distinct reference motion clip. At runtime, the system responds to external control signals by checking the feasibility of switching to a target policy. Feasibility is determined by a forward pass through the target policy's discriminators on the character's current observation — if the score exceeds a threshold, the switch is considered safe. This check runs at the 30 Hz policy rate, imposing negligible runtime cost.

The key advantage over motion-tracking methods is that policies perform inference using only the last H frames of the character's trajectory, without tracking any target reference pose explicitly or implicitly through a phase state. No motion generation or motion matching mechanism is needed for policy transitions. The system can respond to user-provided control signals and switch between behaviors interactively, making it directly suitable for real-time interactive applications and games.

---

## Slide 13 — Incremental Learning

**Image:** No figure in the report. A simple two-stage diagram (meta-policy frozen → cooperative policy training on new subtask) would be the ideal visual if constructed.

Composite motions are typically augmentations of simpler behaviors rather than entirely novel actions. Incremental learning exploits this structure by reusing a pre-trained policy as a meta-policy and training a new cooperative policy that adapts the meta-policy for a new composite task. For example, a walking meta-policy can be extended to a walk-while-punching composite policy by training a cooperative network on the punching reference without requiring any combined walking-punching reference clip — or relearning locomotion from scratch.

The cooperative policy learns state-dependent weights across body parts in a temporally dynamic fashion, automatically mixing the original behavior with the new subtask. This produces composite motion control policies significantly faster than learning from scratch, and the incremental scheme generalises to any pairing of pre-trained base behavior with new subtask reference, provided the reference motions share sufficient overlapping poses to enable feasible policy switching.

---

## Slide 14 — Experiments & Results

Recommend splitting into three sub-slides.

---

**Sub-slide 14A — ICCGAN Humanoid Motions**

**Images:** Fig 16 (comparative training curves across all 6 ICCGAN motions) + Table 3 (performance at convergence, Period 20) + GIFs from Figs 13–15 and 17–22 arranged in a 2×3 grid (jaunty walk, joyful walk, limp walk, squat, punch, leg lunge, kick, long jump, roll).

The adversarial controller is evaluated on six motion categories of increasing complexity. Squat achieves the highest stable lifetime cycles (0.49 ± 0.05) and strongest real discriminator score (0.732), confirming near-perfect kinematic and dynamic replication for motions that constrain leg joints to near-constant positions and eliminate balance challenges. Punch and leg lunge achieve comparable top-tier survival (0.51 and 0.49 cycles) with strong reward, validating the discriminator ensemble's efficacy for upper-body dominant actions where the lower body primarily provides a stable base. Kick initiates the hip-driven swing correctly but frequently loses balance on the support leg — the policy cannot maintain one-legged stance, evidenced by a moderate real discriminator score (0.52) and the characteristic collapse captured in the simulation frames. Long jump performance is speed-sensitive: episodes 4–6 in the checkpoint arrays show clean landings under normal-paced takeoff, while higher velocities introduce forward momentum that disrupts touchdown. Roll represents a fundamental simulator limitation: the pelvis-height termination heuristic triggers precisely when the motion requires floor contact, and despite grace_steps mitigation, the policy cannot recover upright posture from the prone state — confirmed by the lowest lifetime cycles (0.057) and persistently negative reward (−0.137).

---

**Sub-slide 14B — Vimo Full-Body Dance Motions (AIST Retargeted)**

**Images:** Fig 23 (Vimo training curves across dance genres) + Table 4 (convergence performance, Period 30) + GIFs from Figs 24–30 (gBR, gHO, gJS, gLH, gLO, gMH, gPO).

ViMo-generated dance sequences from the AIST Dance Video Database are retargeted and grouped by dance genre and choreography ID. gJS/ch02 (pos. des pieds) yields near-zero lifetime (0.02 cycles) and negative reward (−0.182): reference feet clip into the floor because the SMPL-to-28DOF converter averages ankle and toe into a single joint, inducing physics collapse at the first contact — a converter artifact rather than a policy failure. gBR/ch01 (indian step) and gHO/ch01 (loose legs) achieve partial stability (0.20 and 0.36 cycles) but side-step and sliding motions cause repeated falls before balance recovery. gLH/ch01 (slide) achieves the clearest partial success — 0.35 cycles and real score 0.667 — while higher-tempo hand instability limits full replication. gLO/ch02 (twirl) and gPO/ch01 (fresno) achieve the strongest survivals (0.48 and 0.47 cycles) by prioritising lower-body stabilisation, though upper-body articulation quality degrades. These results confirm the end-to-end pipeline feasibility — generated kinematic sequences load into the controller without format mismatch — while establishing the 28-DOF simplification as the binding constraint on full-body dance complexity.

---

**Sub-slide 14C — Task-Based Locomotion & Ablation Study**

**Images:** Fig 31 (locomotion training curves: walk/run/crouch) + Table 5 + GIFs from Figs 32–34 (walk, crouch, run) + Figs 9–11 (limp/jaunty/joyful walk ablation curves) in a compact grid.

Three goal-directed locomotion policies are evaluated under target heading and target location formulations. Walk achieves reliable goal-following (lifetime 0.31 cycles, task reward 0.170) with moderate angular deviation at wide turns. Crouch exhibits surprisingly strong task performance (lifetime 0.39 cycles, longest survival at 666 steps) through an emergent stabilization strategy: one leg advances slowly as a train-wheel while the other drags as support, preventing falls at the cost of non-textbook gait and elevated value-loss variance. Run is overwhelmed by high-speed dynamics: the policy adopts a kangaroo-like hopping gait, repeatedly skipping forward before falling — lifetime collapses to 0.27 cycles with negative reward (−0.079) and a score_fake of only 0.050, confirming that mimicability remains the bottleneck before task completion can be evaluated reliably.

The ablation study examines phase-conditioned observations and bilateral symmetry regularisation across three locomotion difficulty tiers. Phase conditioning provides the strongest gains on the jaunty walk — an asymmetric high-energy motion — improving lifetime by 31% and reward by 62.3% at H=8, with the mechanism functioning as a temporal lookup key that queries which portion of the motion cycle the policy occupies and reduces advantage estimate variance. Symmetry regularisation shows motion-dependent behavior: beneficial for the limp walk (near-symmetric ground contact), achieving the best lifetime at H=8, but strongly counterproductive for the jaunty walk where the constraint conflicts with the asymmetric target distribution, causing lifetime to drop from 0.29 to 0.13 cycles. For the joyful walk (moderately asymmetric), symmetry improves survival metrics but degrades imitation quality — a survival-imitation trade-off indicated by degraded fake discriminator scores despite extended lifetime — confirming that the appropriate application of this regulariser requires a priori knowledge of motion symmetry properties.

---

## Slide 15 — Conclusion & Future Work

**Image anchor:** Fig 35 (SMPL→DeepMimic retargeted sample) as a visual bookend recalling the pipeline's scope.

The two-stage pipeline demonstrates that physically plausible character animation from casual video is tractable within a unified adversarial-diffusion framework. ViMo-Flow generates kinematically coherent 3D motion from unconstrained footage via DDPM with Min-SNR reweighting and Probabilistic Timestep Sampling without requiring explicit camera calibration. The physics-based imitation stage drives a MuJoCo humanoid to reproduce these kinematics under rigid-body dynamics through body-part-specific discriminators and PPO, eliminating hand-crafted reward engineering. Composite multi-objective reward aggregation and goal-conditioned extensions demonstrate that the adversarial reward paradigm scales to multi-skill learning beyond single-clip imitation.

The current 28-DOF humanoid-lite model is the primary limitation: it collapses SMPL's 24 joints and 72 axis-angle DOF into 15 rigid bodies, failing on motions that demand simultaneous coupled articulation of all major joints — high-energy dance steps, aerial phases, and skidding contact patterns all require joint torques and momentum transfers that the reduced model cannot faithfully reproduce. A secondary limitation is the SMPL-to-28DOF converter's chain-composition step, which fuses ankle and toe joints, inducing physics collapse for motions requiring precise foot contact placement.

Three planned extensions address these gaps. First, upgrading the simulation character from the current DeepMimic 28-DOF body to a 34-DOF rig that mirrors the SMPL joint count — restoring individual spine segments, collarbones, and hand articulations as actuated joints — is expected to unlock complex dance motions where upper- and lower-body coordination is non-trivial. Second, with the richer character model, focus shifts to motions currently out of reach: jumps and landings, lateral shuffles and pivots, and asymmetric arm-leg coordination. Third, once stable physics-based control is established on the extended character, the third stage of the pipeline — policy adaptation via AdaptNet — will be integrated. The two-tier adaptation hierarchy addresses deployment robustness when simulation parameters or task objectives differ from training conditions, completing the transition from casual video input to simulation-ready, deployable character animation systems.