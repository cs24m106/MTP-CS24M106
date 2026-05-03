Now I have a complete visual catalog of every figure and table. Here is the full asset reference guide followed by slide-by-slide content with image placements and flashcard-style text.

---

## Complete Figure & Table Catalog (Phase 2 Slides)

| Asset | Description | Use in slide |
|---|---|---|
| **Fig 1** | GAIL runtime system diagram (policy switcher flow) | Slide: Interactive Policy Switching |
| **Fig 7** | DeepMimic humanoid joint map + coronal plane (a)(b) | Slide: Key Concept — Simulation Character |
| **Fig 8** | Duplicate of Fig 7 (same diagram, repeated in report) | — |
| **Table 1** | SMPL vs MuJoCo property comparison table | Slide: Motion Corpus & Data Prep |
| **Fig 35** | SMPL (34-joint) → DeepMimic (28-joint) retargeted side-by-side frames | Slide: Motion Retargeting |
| **Fig 12** | Network architecture: (a) Policy, (b) K-Head Critic, (c) Discriminator Ensemble | Slide: Network Architecture |
| **Table 2** | Hyperparameters table | Slide: Network Architecture / Loss Functions |
| **Fig 9** | Limp walk training comparison (baseline/phase/sym/phase+sym, H=8 & H=16) | Slide: Experiments — Ablation |
| **Fig 10** | Jaunty walk training comparison across variants | Slide: Experiments — Ablation |
| **Fig 11** | Joyful walk training comparison | Slide: Experiments — Ablation |
| **Fig 13–15** | Jaunty / Joyful / Limp walk overlapped simulation frames → **use GIFs** | Slide: Results — ICCGAN |
| **Table 3** | ICCGAN motion performance at convergence | Slide: Results — ICCGAN |
| **Fig 16** | Comparative training curves across ICCGAN motions (6 motions) | Slide: Results — ICCGAN |
| **Fig 17** | Squat — stable frames → **use GIF** | Slide: Results — ICCGAN |
| **Fig 18** | Punch — upper-body frames → **use GIF** | Slide: Results — ICCGAN |
| **Fig 19** | Leg lunge — forward extension frames → **use GIF** | Slide: Results — ICCGAN |
| **Fig 20** | Kick — hip-swing frames → **use GIF** | Slide: Results — ICCGAN |
| **Fig 21** | Long jump — takeoff/landing frames → **use GIF** | Slide: Results — ICCGAN |
| **Fig 22** | Roll — dive initiation frames → **use GIF** | Slide: Results — ICCGAN |
| **Fig 23** | Vimo training curves across dance genres | Slide: Results — Vimo Dance |
| **Table 4** | Vimo dance motion performance at convergence | Slide: Results — Vimo Dance |
| **Fig 24–30** | Dance genre simulation frames (gBR, gHO, gJS, gLH, gLO, gMH, gPO) → **use GIFs** | Slide: Results — Vimo Dance |
| **Fig 31** | Task-based locomotion training curves (walk/run/crouch) | Slide: Results — Task Locomotion |
| **Table 5** | Task-based locomotion performance at convergence | Slide: Results — Task Locomotion |
| **Fig 32** | locomotion_walk frames → **use GIF** | Slide: Results — Task Locomotion |
| **Fig 33** | locomotion_crouch frames → **use GIF** | Slide: Results — Task Locomotion |
| **Fig 34** | locomotion_run frames → **use GIF** | Slide: Results — Task Locomotion |

---

## Slide-by-Slide Content (Slides 3–15)

---

### Slide 3 — Problem Statement

**Image:** Table 1 (SMPL vs MuJoCo) — anchor the structural mismatch visually.

**Flashcard text:**

> **Input:** Kinematic 3D motion sequences from Stage 1 (ViMo-Flow) — SMPL format, 24 joints, axis-angle, Y-up
>
> **Goal:** Drive a physically simulated humanoid to reproduce these motions under rigid-body dynamics
>
> **Core challenge:** SMPL skeletons carry no physics — no torques, no contact model, no ground truth control signals
>
> **Approach:** Learn a control policy via deep RL, using adversarial imitation to eliminate manual reward engineering

---

### Slide 4 — Key Concept: Simulation Character

**Image:** Fig 7 — DeepMimic humanoid joint map (a) + coronal plane (b). Dominant visual, minimal text alongside.

**Flashcard text:**

> **Simulator:** MuJoCo — rigid-body physics, friction contacts, torque-actuated joints
>
> **Character:** DeepMimic humanoid — 15 rigid bodies, 28 actuated DOF
>
> **Actuation:** PD servos — policy outputs target joint angles, not raw torques
>
> **Control rate:** 30 Hz policy / 120 Hz physics (4 sub-steps per policy step)

---

### Slide 5 — Overview

*(A clean pipeline diagram — you may draw this or generate it; no existing figure covers the full Phase 2 flow.)*

**Flashcard text:**

> Stage 1 output (SMPL kinematic sequence) → **Retargeting** (SMPL→MuJoCo) → **Reference clips** → **Discriminator ensemble** (body-part-specific) → **PPO policy** ← **MuJoCo physics** → physically valid humanoid control

---

### Slide 6 — Motion Corpus & Data Preparation

**Image:** Table 1 (already seen in Slide 3 — reuse or replace with a side-by-side skeleton diagram if available). Also candidate: Fig 35 (SMPL→DeepMimic retargeted sample column — just the skeleton view side).

**Flashcard text:**

> **Reference datasets:** AMASS / AIST++ (dance, SMPL format), LaFAN1 (locomotion)
>
> **SMPL format:** 24 joints · 72 axis-angle DOF · Y-up · no contact model
>
> **MuJoCo humanoid:** 15 bodies · 28 DOF · Z-up · quaternion · rigid-body + friction
>
> **Key mismatches:** coordinate axes · joint count · rotation format · hand/collar fusion

---

### Slide 7 — Motion Retargeting: SMPL → Simulation

**Image:** Fig 35 — full side-by-side SMPL skeleton frames (left) vs. MuJoCo humanoid frames (right). This image IS the explanation — let it lead, keep text minimal.

**Flashcard text (5 steps):**

> 1. **Coord. transform** — (x,y,z) → (z,x,y) cyclic permutation (Y-up → Z-up)
> 2. **Height scaling** — pelvis-to-foot ratio aligns limb proportions
> 3. **Floor correction** — ankle/foot weighted min height → ground plane alignment
> 4. **Bind rotations** — per-joint Rodrigues correction for rest-pose differences
> 5. **Chain composition** — multi-joint SMPL chains (Spine1+2+3, Neck+Head) → single MuJoCo body; export as DeepMimic JSON @ 30 Hz

---

### Slide 8 — Observation, Action & Discriminator Interface

**Image:** None available — consider a simple diagram (optional). Text as flashcards suffices here.

**Flashcard text:**

> **Policy observes:** H=4 consecutive frames of root-relative body-link features (3D pos + 4D quat per link) → s_t ∈ ℝ^(H×7L), normalized ±5σ
>
> **Action space:** a_t ∈ ℝ^28 — target joint angles for PD servos; τ = k_p(a_t − q) − k_d q̇
>
> **Discriminator observes:** window H+1 frames, restricted to its body-part subset B_k → o_k ∈ ℝ^(H_k × 7|B_k|)
>
> **Why short trajectories?** Sensitive to both instantaneous pose quality AND short-horizon motion continuity

---

### Slide 9 — Network Architecture

**Image:** Fig 12 — (a) Policy Network, (b) K-Head Critic, (c) Discriminator Ensemble. Full image, labels point to key components.

**Flashcard text (beside or below image):**

> **Actor:** GRU (hidden 256) + MLP [1024→512] → outputs μ, log σ for stochastic actions
>
> **Critic:** same backbone → K independent value heads (one per discriminator)
>
> **Discriminator:** GRU + MLP [256→128→32] → scalar score r^D_i ∈ [−1, 1] (hinge-bounded)
>
> **Key design:** shared GRU across all three; goal vector g_t appended post-encoding

---

### Slide 10 — Loss Functions

**Image:** Table 2 (Hyperparameters) — include as a compact reference block.

**Flashcard text:**

> **Discriminator loss** (hinge): max(0, 1 − D·s_real) + max(0, 1 + D·s_fake) + λ_GP · gradient penalty
>
> **Composite reward:** r_t = Σ_k w_k · σ(D_k(s_t, a_t)) — learnable aggregation weights w_k
>
> **PPO objective:** surrogate clip ε=0.2, GAE λ=0.95, γ=0.95
>
> **Symmetry loss:** ℒ_sym = ‖(a_left − ā) − (a_right − ā)‖² / N_pairs, weight λ_sym = 0.005
>
> **Phase conditioning:** observations augmented with (sin 2πφ, cos 2πφ) for cycle-tracking

---

### Slide 11 — Composite Motion & Goal-Conditioned Control

**Image:** No figure in report — consider a simple body-part partition diagram or a comp.md figure reference. Text-dominant slide is acceptable here.

**Flashcard text:**

> **Problem:** Single-clip imitation cannot produce composite behaviors (e.g., walk + gesture)
>
> **Solution:** Each discriminator D_k assigned to a distinct body-part group B_k
>
> **Automatic mixing:** policy learns state-dependent weights — no manual clip blending needed
>
> **Goal control:** goal vector g_t ∈ ℝ^G appended to actor/critic — supports target heading and target location rewards
>
> **No pre-composed reference clips required**

---

### Slide 12 — Interactive Policy Switching

**Image:** Fig 1 — GAIL policy switcher runtime diagram. Clean, self-explanatory.

**Flashcard text (beside image):**

> **How it works:** discriminator feasibility check — forward pass only, real-time (~30 Hz)
>
> **No phase state or target pose needed** — causal inference on last H frames only
>
> **Switch condition:** if D_target(s_t) passes threshold → activate target policy
>
> **Low runtime cost** — easily integrated into interactive applications

---

### Slide 13 — Incremental Learning

**Image:** None in report. Text-dominant or a simple diagram of meta-policy → cooperative policy flow.

**Flashcard text:**

> **Motivation:** Composite motions are typically augmentations of simpler behaviors
>
> **Meta-policy:** pre-trained locomotion policy frozen as base
>
> **Cooperative policy:** new policy trained to cooperate with meta-policy for composite task
>
> **Result:** significantly faster acquisition than learning from scratch
>
> **Example:** walking meta-policy + punching cooperative policy → walk-while-punch, no combined reference clip needed

---

### Slide 14 — Experiments & Results

Split into **three sub-slides** (or one slide with tabbed layout):

**Sub-slide A — ICCGAN Humanoid Motions**

> **Images:** Fig 16 (training curves) + GIFs: Figs 17–22 (squat, punch, lunge, kick, long jump, roll) arranged in a grid + Table 3
>
> **Flashcard:** Squat/Punch/Lunge → near-perfect imitation · Kick/Roll → single-leg instability & termination heuristic limits · Body-part discriminator scales to 6 motion categories without reward tuning

**Sub-slide B — Vimo Full-Body Dance (AIST Retargeted)**

> **Images:** Fig 23 (training curves) + GIFs: Figs 24–30 (gBR, gHO, gJS, gLH, gLO, gMH, gPO) + Table 4
>
> **Flashcard:** gJS collapse → SMPL→28DOF converter fuses ankle+toe (physics collapse) · gLH partial success · 28-DOF simplification is the ceiling for complex dance · End-to-end feasibility confirmed: generated kinematics load without format mismatch

**Sub-slide C — Task-Based Locomotion (LaFAN1) + Ablation**

> **Images:** Fig 31 (training curves) + GIFs: Figs 32–34 (walk, crouch, run) + Table 5 + Figs 9–11 (ablation curves for limp/jaunty/joyful walk)
>
> **Flashcard:** Walk/Crouch → effective goal-following · Run → kangaroo hopping; high-velocity dynamics exceed 28-DOF capacity · Phase conditioning: +62% reward on jaunty walk · Symmetry regularisation: motion-dependent — beneficial for limp, harmful for jaunty

---

### Slide 15 — Conclusion & Future Work

**Image:** Fig 35 again (SMPL→MuJoCo retargeted sample) as a reminder of the pipeline's scope — visual bookend.

**Flashcard text:**

> ✓ **Achieved:** physics-valid imitation from casual video via unified adversarial-diffusion pipeline
>
> ✓ **Achieved:** composite multi-objective control from decoupled motion sources, no pre-composed clips
>
> ✓ **Achieved:** end-to-end feasibility — ViMo-Flow kinematics structurally compatible with MuJoCo controller
>
> ⚠ **Limit:** 28-DOF humanoid-lite collapses SMPL's 72 DOF — complex dance, aerial phases, rolls out of reach
>
> → **Next:** 34-DOF extended rig · complex motion support · policy adaptation via AdaptNet (two-tier: latent-embedding + deep-layer modification)