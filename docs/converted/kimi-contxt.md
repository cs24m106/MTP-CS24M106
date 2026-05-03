 I'll update the entire context with LaTeX placeholders using the original source material equations. Let me verify the exact equations from comp.md and gail.md, and use web search for any equations that are garbled in the markdown files.
 Now I have the clean equations from the original papers. Let me also open the PDF for ICCGAN to get exact equations:
 Now I have the exact equations from both source papers. Let me compile the complete updated context with proper LaTeX placeholders.

---

## Complete Updated Context (Slides 8–15) — Source-Aligned with LaTeX Placeholders

---

### **Slide 8 — Observation, Action & Discriminator Interface**

**Image:** None available — consider a simple diagram (optional). Text as flashcards suffices here.

**Flashcard text:**

> **Policy State Space:**
> - $s_t \in \mathbb{R}^{4 \times 13|B|}$ — 4 consecutive frames ($t-3$ to $t$) **with velocity augmentation**
> - Root-relative body-link features: position, orientation, **linear and angular velocities**
> - GRU embeds temporal state input (256 hidden units)
> - Normalized by moving average during training

> **Discriminator Observation :**
> - $o_t \in \mathbb{R}^{5 \times 7|B|}$ — **5 consecutive frames** ($t-3$ to $t+1$), **no velocity**
> - Position and orientation only (quaternion), relative to root position **and heading direction**
> - $H_k = H + 1$ window length (policy sees $H=4$, discriminator sees $H+1=5$)
> - **Key design:** Discriminator uses longer window than policy to enforce motion continuity

> **Action Space:**
> - $a_t \in \mathbb{R}^{28}$ — target generalized coordinates for PD servos
> - Revolute joints: 1D target angle (radians)
> - Spherical joints: 4D rotation (axis-angle quaternion)
> - Indirect actuation: $\tau = k_p(a_t - q) - k_d \dot{q}$

> **Why this separation?**
> - Policy needs velocity for dynamics-aware control
> - Discriminator judges pose quality + short-horizon continuity without velocity bias
> - Velocity in discriminator makes it "too harsh" — early policy cannot match reference velocities via PD servo.

---

### **Slide 9 — Network Architecture**

**Image anchor:** Fig. 3 (comp.md) / Fig. 4 (gail.md) — Policy + Discriminator architecture. Let diagram lead.

**Flashcard text:**

> **Shared Backbone:**
> - GRU encoder: input dim $7|B|$ (discriminator) or $13|B|$ (policy), hidden dim **256**
> - Temporal embedding passed through MLP with **2 FC layers**

> **Policy Network (Actor):**
> - GRU(256) → FC(1024) → FC(512)
> - Outputs: $\mu \in \mathbb{R}^{28}$, $\log \sigma \in \mathbb{R}^{28}$ for factored Gaussian
> - **Incremental learning variant:** additionally outputs weight vector $w_t \in \mathbb{R}^{28}$ for meta-policy mixing

> **Critic Network — comp.md Fig. 3b (multi-head):**
> - Same backbone as policy → **K independent value heads** (one per objective)
> - Each head has own PopArt normalizer (scale + shift)
> - **Critical for multi-objective:** per-component advantage estimation prevents gradient dominance by single objective

> **Discriminator Ensemble — gail.md Fig. 4, comp.md Fig. 3c:**
> - GRU(256) → FC(256) → FC(128) → **32 independent scalar heads**
> - **Shared feature layers, independent final layers** (orthogonal init: gain=$\sqrt{2}$ hidden, gain=1 output)
> - Ensemble average: $r^D = \frac{1}{32} \sum_{i=1}^{32} \text{Clip}(D_i(o), -1, 1)$
> - $\oplus$ denotes concatenation, $\ominus$ denotes ensemble averaging operator

---

I'll restructure Slide 10 onwards to use your report (report.md) as the primary base, weaving in source material references where they directly support or clarify the report's methodology. Let me first extract the key technical content from your report.
Now I have all the report content extracted. Let me draft the complete updated context from Slide 10 onwards, using report.md as the base and weaving in source material references only where they directly support the report's narrative.

---

## Updated Context (Slides 10–15) — Report-Based with Source Material Integration

---

### **Slide 10 — Loss Functions & Training Objectives**

**Image anchor:** Table 2 (report.md hyperparameters) — include as compact reference block.

**Flashcard text:**

> **Discriminator Training Objective (report §7.2(i), comp.md §4.2 Eq. 2):**
> - Hinge loss per ensemble head:
>   ```latex
>   L_{D_k} = \frac{1}{N}\sum_{i=1}^{N} \left\{ \mathbb{E}_{o \sim p_{ref}}[\max(0, 1 - D_k^i(o))] + \mathbb{E}_{o \sim p_{fake}}[\max(0, 1 + D_k^i(o))] \right\}
>   ```
> - Gradient penalty (WGAN-GP, Gulrajani et al. 2017):
>   ```latex
>   GP_k = \mathbb{E}_{\hat{o}}\left[\left(\left\|\nabla_{\hat{o}} D_k^i(\hat{o})\right\|_2 - 1\right)^2\right], \quad \hat{o} = \alpha o_{ref} + (1-\alpha)o_{fake}, \; \alpha \sim \mathcal{U}(0,1)
>   ```
> - Combined discriminator loss: $L_{D_k}^{total} = L_{D_k} + \lambda_{GP} \cdot GP_k$ with $\lambda_{GP} = 10$

> **Composite Reward Formulation (report §6.5, comp.md §4.3):**
> - Per-discriminator reward:
>   ```latex
>   r_t^{(k)} = \frac{1}{32}\sum_{i=1}^{32} \text{Clip}\left(D_k^i(o_{t-H_k:t}), -1, 1\right)
>   ```
> - Aggregated reward with learnable weights:
>   ```latex
>   r_t = \sum_k w_k \cdot r_t^{(k)} + \sum_j w_j \cdot r_t^{(\text{task},j)}, \quad \sum_k w_k + \sum_j w_j = 1
>   ```
> - **Eliminates hand-crafted reward engineering** — discriminator logits directly drive policy optimization

> **PPO Policy Optimization (report §7.2(ii), comp.md §4.3 Eq. 7–9):**
Generalized Advantage Estimation per objective head:
```latex
\hat{A}_t^{(i)} = \sum_{l=0}^{H_{steps}-1-t} (\gamma\lambda)^l \delta_{t+l}^{(i)}, \quad \delta_t^{(i)} = r_t^{(i)} + \gamma V^{(i)}(s_{t+1}) - V^{(i)}(s_t)
```
Per-objective standardization (multi-critic architecture):
```latex
\tilde{A}_t^{(i)} = \frac{\hat{A}_t^{(i)} - \mu_i}{\sigma_i + \epsilon}
```
Weighted combined advantage: $\tilde{A}_t = \sum_i w_i \tilde{A}_t^{(i)}$
Clipped surrogate objective:
```latex
L^{\text{clip}} = \mathbb{E}_t\left[\min\left(\rho_t \tilde{A}_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\tilde{A}_t\right)\right], \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
```
Value loss with PopArt normalization per head: $L^{\text{value}} = \sum_i \left(V^{(i)}(s_t) - R_t^{(i)}\right)^2$

> **Auxiliary Regularisations (report §7.2(iii), comp.md §6.6):**
> - Bilateral symmetry loss:
>   ```latex
>   L_{\text{sym}} = \frac{\lambda_{\text{sym}}}{|P|}\sum_{(i,j) \in P} \left\|(\mu_j - \bar{\mu}_j) - (\mu_i - \bar{\mu}_i)\right\|_2^2
>   ```
>   where $P$ is the set of mirrored joint pairs, $\lambda_{\text{sym}} = 0.005$
> - Phase-conditioned observations:
>   ```latex
>   s_t^{\text{phase}} = \left[s_t; \sin(2\pi\phi_t); \cos(2\pi\phi_t)\right], \quad \phi_t = \frac{t \bmod T_{\text{cycle}}}{T_{\text{cycle}}} \in [0,1)
>   ```
> - Full PPO objective: $L = L^{\text{clip}} + c_v L^{\text{value}} + L_{\text{sym}}$

> **Source context:** The multi-critic architecture with per-objective advantage standardization and PopArt value normalization follows the multi-objective learning framework of comp.md §4.3, which avoids scalarizing competing objectives into a single reward.

---

### **Slide 11 — Composite Motion & Goal-Conditioned Control**

**Image:** No figure in report. Consider a simple body-part partition diagram (torso/arms/legs colored by discriminator assignment) or use comp.md Fig. 2 overview.

**Flashcard text:**

> **Problem: Single-clip imitation cannot produce composite behaviors (report §6.5, comp.md §1)**
> - No pre-composed reference clips exist for every combination (e.g., walk + gesture, locomotion + manipulation)
> - Existing methods require manual blending or full-body reference generation

> **Solution: Body-Part Decoupling with Discriminator Ensemble (report §6.5, comp.md §4.1–4.2)**
> - Each discriminator $D_k$ assigned to distinct body-part group $B_k$ operating on subset of key links
> - Typical splits: **upper body** (arms, hands, torso, head) vs **lower body** (pelvis, legs, feet)
> - **Shared links for coordination:** e.g., one leg included in both groups to avoid ipsilateral walking
> - **Custom splits for complex tasks:** juggling uses arms-only group + rest-of-body group for walking

> **Automatic Mixing via Multi-Objective Learning (report §6.5, comp.md §4.3)**
> - Policy explores state-dependent weighting **without manual clip blending**
> - Each objective maintains **independent critic head** — no scalarization of rewards
> - Per-objective advantages normalized to same scale via standardization
> - Equal weights $w_k = 1/K$ suffice for most cases; task-specific weighting for difficult objectives (e.g., juggling 0.6, locomotion 0.1)

> **Goal-Directed Task Rewards**
> - **Target Heading:** alignment between root displacement and target direction
>   ```latex
>   r_t^{\text{heading}} = \left\langle \frac{\Delta x_{t+1}^{\text{root}}}{\|\Delta x_{t+1}\|}, g_t \right\rangle, \quad g_t \in \mathbb{R}^2 \text{ (unit vector, resampled every 30 frames)}
>   ```
> - **Target Location:** preferred speed and direction toward goal
>   ```latex
>   r_t^{\text{loc}} = \begin{cases} \exp\left(-3\left\|\frac{\Delta x_{t+1}^{\text{root}}}{\Delta t} - v_t^*\right\|^2 / \|v_t^*\|^2\right) & \text{if } \|x_{t+1} - p_{\text{goal}}\| > \eta \\ 1 & \text{otherwise} \end{cases}
>   ```
> - **Aiming:** forearm direction toward target
>   ```latex
>   r_t^{\text{aim}} = \begin{cases} \exp\left(-2\|d_t^{\text{forearm}} - g_t\|^2\right) & \text{if aiming active} \\ \text{Clip}(\langle d_t^{\text{forearm}}, u_{\text{ref}} \rangle, 0, 0.8)/0.8 & \text{otherwise} \end{cases}
>   ```

> **Key Insight:** "The control policy explores by itself how the composite motions can be combined automatically — no need of any manual work to produce composite reference motions for learning"

---

### **Slide 12 — Interactive Policy Switching**

**Image anchor:** Fig. 1 (report.md / gail.md) — GAIL policy switcher runtime diagram.

**Flashcard text:**

> **Runtime Architecture (report §6.6, gail.md §5.3)**
> - Multiple motor control policies trained separately, each imitating distinct reference clip
> - One active policy controls character at 30 Hz via PD servos
> - User control signal triggers feasibility check for target policy takeover

> **Discriminator-Based Feasibility Check (report §6.6, gail.md Eq. 6)**
> - Switch accepted when:
>   ```latex
>   \frac{1}{N}\sum_{i=1}^{N} \text{Clip}\left(D_i^{\text{target}}(o_{t-H:t}), -1, 1\right) \geq \eta
>   ```
> - Threshold $\eta$ behavior-specific (e.g., $\eta = 0.1$ for jump→run yields >95% success rate)
> - **Forward pass only** — runs at 30 Hz policy rate, negligible runtime cost (~190 μs)

> **Advantages Over Motion Tracking (report §6.6, gail.md §5.3)**
> - **No phase state variable** — policies infer from last $H$ frames only
> - **No target pose generation** — no motion matching or motion generation mechanism
> - **No explicit reference tracking** — causal inference on recent trajectory history
> - Transitions occur at intermediate poses between keyframes (discriminators trained on interpolated samples)

> **Auto-Activated Recovery (gail.md §6.5, Fig. 9 — source extension)**
> - Recovery policy (e.g., "get up") continuously monitors discriminator score
> - Auto-activates when score exceeds threshold, returns control to base policy when base score recovers
> - Demonstrates robustness extension of the switching framework

---

### **Slide 13 — Incremental Learning**

**Image:** No figure in report. Simple two-stage diagram: meta-policy frozen → cooperative policy training.

**Flashcard text:**

> **Motivation (report §6.7, comp.md §5)**
> - Composite motions are typically augmentations of simpler base behaviors
> - Humans learn incrementally: walking meta-skill → walking-while-holding-phone
> - **Avoid relearning base behavior from scratch** when adding new subtasks

> **Meta-Policy & Cooperative Policy Formulation (report §6.7, comp.md Eq. 10)**
> - Pre-trained **meta-policy** $\pi^{\text{meta}}$ frozen via stop-gradient operator
> - Cooperative policy parameterization:
>   ```latex
>   \pi(a_t | s_t, g_t, a_t^{\text{meta}}) = \mathcal{N}\left(\mu_t + w_t \odot \text{Stop}(a_t^{\text{meta}}), \text{diag}(\sigma_t^2)\right)
>   ```
> - $w_t \in \mathbb{R}^{28}$: **learnable per-DoF weight vector** — same dimension as action space
> - $\mu_t, \sigma_t, w_t$ all output by neural network taking $s_t, g_t$ as input
> - **High $w_t$:** rely on meta-policy for that DoF; **Low $w_t$:** override with cooperative policy's own action

> **Training Efficiency (report §6.7, comp.md §6.5)**
> - From scratch: ~15–30 hours, $2$–$4 \times 10^8$ samples for complex multi-objective tasks
> - **Incremental: ~30 minutes–2 hours**, ~20M samples
> - Example: Aiming+Walk, Aiming+Run, Aiming+Crouch all trained from single walking meta-policy

> **Learned Weight Visualization (comp.md Fig. 7–8 — source validation)**
> - Aiming+Locomotion: **high weights (red) on lower-body DoFs** → meta-policy controls legs
> - **Low weights (blue) on upper-body DoFs** → cooperative policy controls aiming
> - Crouch+AimingWalk: reversed pattern — cooperative overrides lower body, preserves meta-policy's upper-body aiming
> - Confirms state-dependent, temporally dynamic mixing emerges automatically

---

### **Slide 14 — Experiments & Results**

Split into **three sub-slides** following your report structure (X.1, X.2, X.3):

---

**Sub-slide 14A — ICCGAN Humanoid Motions (report §10.1)**

> **Images:** Fig. 16 (comparative training curves) + GIFs: Figs. 17–22 (squat, punch, lunge, kick, long jump, roll) + Table 3

**Flashcard text:**

> **Evaluation Setup (report §8.1, §10.1)**
> - 5000 epochs, MuJoCo at 30 FPS, 120 Hz physics
> - Horizon $H_{\text{steps}} = 16$, 16 parallel environments
> - Default hyperparameters: actor LR $5 \times 10^{-6}$, critic $1 \times 10^{-4}$, discriminator $1 \times 10^{-5}$
> - $\gamma = 0.95$, $\lambda_{\text{GAE}} = 0.95$, $\epsilon = 0.2$, $\lambda_{\text{GP}} = 10$

> **Performance Summary (report Table 3)**

| Motion | Lifetime Cycles | Reward Mean | Disc. Gap | Score Real | Score Fake | Inference Highlight |
|--------|----------------|-------------|-----------|------------|------------|---------------------|
| squat | $0.49 \pm 0.05$ | $0.194 \pm 0.029$ | 0.540 | 0.732 | 0.192 | Perfect replication — fixed leg pose, zero slip |
| punch | $0.51 \pm 0.07$ | $0.369 \pm 0.026$ | 0.412 | 0.782 | 0.370 | Highest survival/reward; minor terminal leg slip |
| leg_lunge | $0.49 \pm 0.05$ | $0.328 \pm 0.037$ | 0.392 | 0.719 | 0.327 | High cycles; value-loss variance indicates slipperiness |
| kick | $0.39 \pm 0.07$ | $0.148 \pm 0.024$ | 0.364 | 0.525 | 0.161 | Hip swing initiates; support leg instability causes falls |
| long_jump | $0.36 \pm 0.06$ | $0.075 \pm 0.023$ | 0.348 | 0.445 | 0.097 | Speed-sensitive; moderate run-up improves landing |
| roll | $0.06 \pm 0.01$ | $-0.137 \pm 0.013$ | 0.198 | 0.170 | $-0.028$ | Pelvis-height termination prevents full recovery |

> **Key Findings (report §10.1)**
> - **Squat/Punch/Lunge:** Near-perfect imitation — body-part discriminator scales to static/cyclic motions without reward tuning
> - **Kick:** Single-leg stance instability — discriminator score (0.525 real) confirms partial success
> - **Long jump:** Landing fidelity improves at moderate takeoff speed; excessive forward lean degrades stability
> - **Roll:** Fundamental simulator limitation — pelvis-height termination heuristic ($< 0.15$m) triggers precisely when floor contact required; grace_steps mitigation insufficient for full recovery

---

**Sub-slide 14B — Vimo Full-Body Dance Motions (report §10.2)**

> **Images:** Fig. 23 (training curves) + GIFs: Figs. 24–30 (gBR, gHO, gJS, gLH, gLO, gMH, gPO) + Table 4

**Flashcard text:**

> **Data Source (report §10.2)**
> - ViMo-generated dance sequences from AIST Dance Video Database
> - Retargeted and grouped by dance_genre/choreography_id
> - Choreography ID defines core sequence; musical piece IDs vary tempo only

> **Performance Summary (report Table 4)**

| Motion | Lifetime Cycles | Reward | Disc. Gap | Score Real | Score Fake | Inference Highlight |
|--------|----------------|--------|-----------|------------|------------|---------------------|
| gBR/ch01 (indian step) | $0.20 \pm 0.04$ | $0.079$ | 0.484 | 0.569 | 0.085 | Side-step falls; limited recovery |
| gHO/ch01 (loose legs) | $0.36 \pm 0.05$ | $0.176$ | 0.526 | 0.702 | 0.175 | Sliding legs cause balance loss |
| gJS/ch02 (pos. des pieds) | $0.02 \pm 0.00$ | $-0.182$ | 0.304 | 0.179 | $-0.125$ | **Converter artifact:** ankle+toe fused → foot clipping collapse |
| gLH/ch01 (slide) | $0.35 \pm 0.06$ | $0.136$ | 0.531 | 0.667 | 0.136 | Partial success; tempo-induced hand instability |
| gLO/ch02 (twirl) | $0.48 \pm 0.04$ | $0.096$ | 0.463 | 0.557 | 0.094 | Stabilization over knee bends; rhythmic quality degraded |
| gMH/ch02 (rock board) | $0.24 \pm 0.04$ | $0.093$ | 0.524 | 0.621 | 0.097 | Leg-lift instability causes complete failure |
| gPO/ch01 (fresno) | $0.47 \pm 0.05$ | $0.205$ | 0.453 | 0.659 | 0.206 | Highest reward; legs prioritize stability, arms follow reference |

> **Key Findings (report §10.2)**
> - **End-to-end pipeline feasibility confirmed:** Generated kinematics load into controller without format mismatch
> - **gJS collapse:** SMPL→28DOF converter fuses ankle+toe into single joint → reference feet clip through floor → physics collapse at first contact (artifact, not policy failure)
> - **28-DOF simplification is binding constraint:** Complex rhythmic dance, aerial phases, full-body coordination exceed reduced model capacity
> - **gLO/gPO best survivors:** Policies prioritize lower-body stabilization over upper-body articulation quality

---

**Sub-slide 14C — Task-Based Locomotion & Ablation Study (report §10.3, §8.3–8.4)**

> **Images:** Fig. 31 (locomotion training curves) + GIFs: Figs. 32–34 (walk, crouch, run) + Table 5 + Figs. 9–11 (ablation curves)

**Flashcard text:**

> **Task-Based Locomotion Performance (report Table 5)**

| Motion | Lifetime Cycles | Reward | Disc. Gap | Score Real | Score Fake | Task Reward | Steps | Inference Highlight |
|--------|----------------|--------|-----------|------------|------------|-------------|-------|---------------------|
| walk | $0.31 \pm 0.05$ | $0.163$ | 0.413 | 0.694 | 0.281 | $0.170$ | 1253 | Decent mimicry; struggles at wide deviation turns |
| run | $0.27 \pm 0.07$ | $-0.079$ | 0.535 | 0.585 | 0.050 | $0.078$ | 2122 | Kangaroo hopping; mimicry bottleneck before task eval |
| crouch | $0.39 \pm 0.05$ | $0.064$ | 0.508 | 0.625 | 0.116 | $0.122$ | 666 | Longest survival; train-wheel leg-drag hack stabilizes turns |

> **Goal-Directed Task Setup (report §10.3, comp.md Appendix B.1–B.2)**
> - **Target Heading:** random unit direction $g_t \in \mathbb{R}^2$, resampled every 30 frames
> - **Target Location:** goal radius $\eta = 0.5$m, preferred speed $[1, 1.5]$ m/s (walk/crouch) or $[1, 3]$ m/s (run)
> - Task reward combined with imitation reward via multi-objective aggregation

> **Ablation I: Phase-Conditioned Observations (report §8.3, Fig. 9–10)**
> - Augments state with $[\sin(2\pi\phi_t), \cos(2\pi\phi_t)]$ where $\phi_t = (t \bmod T_{\text{cycle}})/T_{\text{cycle}}$
> - **Jaunty walk (asymmetric, high-energy):** strongest gains
>   - Lifetime: $0.29 \rightarrow 0.38$ cycles (+31%) at $H=8$
>   - Reward: $-0.093 \rightarrow -0.035$ (+62.3%)
>   - Mechanism: temporal lookup key reduces GAE variance
> - **Limitation:** Requires known fixed cycle structure; fails for aperiodic motions

> **Ablation II: Bilateral Symmetry Regularisation (report §8.4, Fig. 9–11)**
> - $L_{\text{sym}} = (\lambda_{\text{sym}}/|P|) \sum_{(i,j) \in P} \|(\mu_j - \bar{\mu}_j) - (\mu_i - \bar{\mu}_i)\|^2$, $\lambda_{\text{sym}} = 0.005$
> - **Motion-dependent effects:**
>   - **Limp walk (near-symmetric):** beneficial — best lifetime $0.71 \pm 0.08$ at $H=8$
>   - **Jaunty walk (asymmetric):** strongly counterproductive — lifetime drops $0.29 \rightarrow 0.13$
>   - **Joyful walk (moderately asymmetric):** survival-imitation trade-off — lifetime improves to $0.71$ but fake discriminator score degrades to $-0.104$
> - **Conclusion:** Requires a priori knowledge of motion symmetry properties; not suitable as default component

---

### **Slide 15 — Conclusion & Future Work**

**Image anchor:** Fig. 35 (report.md) — SMPL→DeepMimic retargeted sample as visual bookend.

**Flashcard text:**

> **Achievements (report §IX)**
> - ✓ **End-to-end pipeline:** casual video → kinematic motion → physics-based control
> - ✓ **ViMo-Flow:** DDPM with Min-SNR reweighting and Probabilistic Timestep Sampling for video-to-motion generation
> - ✓ **Adversarial imitation:** body-part-specific discriminators with automatic reward aggregation — **no hand-crafted reward engineering**
> - ✓ **Composite multi-objective control:** decoupled motion sources combined without pre-composed reference clips
> - ✓ **Motion retargeting:** SMPL→MuJoCo via coordinate transforms, bind rotations, proportional height scaling
> - ✓ **Empirical validation:** baseline adversarial controller scales across locomotion difficulty tiers; auxiliary mechanisms (phase, symmetry) show narrow, motion-specific benefits

> **Limitations (report §IX, comp.md §7)**
> - ⚠ **28-DOF humanoid-lite:** collapses SMPL's 72 DOF into 15 rigid bodies — fails on motions requiring simultaneous coupled articulation (high-energy dance, aerial phases, skidding contacts)
> - ⚠ **Converter artifact:** SMPL→28DOF chain-composition fuses ankle+toe → foot clipping for precise contact motions (gJS collapse)
> - ⚠ **Fixed body splits:** static upper/lower assignment; humans fluidly enlist/release body parts (comp.md §7 future direction)
> - ⚠ **No phase locking:** cannot handle multi-phase behaviors with explicit state transitions
> - ⚠ **Training required for composites:** cannot combine pre-trained policies at runtime without retraining

> **Planned Extensions (report §IX)**
> 1. **Extended humanoid (34 DOF):** Restore individual spine segments, collarbones, hand articulations — unlock complex dance motions with non-trivial upper/lower coordination
> 2. **Complex motion support:** Jumps/landings (aerial impulse + contact recovery), lateral shuffles/pivots, asymmetric arm-leg coordination
> 3. **Policy adaptation via AdaptNet:** Two-tier hierarchy (latent-embedding perturbation for modest shifts; deep-layer modification for substantial changes) — addresses deployment robustness under domain variation

---

## Summary of Source Material Integration

| Report Section | Source Material Woven In | Purpose |
|---------------|------------------------|---------|
| §7.2(i) Discriminator loss | comp.md §4.2 Eq. 2 (hinge loss form) | Validates report's loss formulation against original |
| §7.2(ii) PPO multi-objective | comp.md §4.3 Eq. 7–9 (advantage standardization) | Confirms multi-critic architecture origin |
| §6.5 Goal rewards | comp.md Appendix B.1–B.5 (exact reward formulas) | Provides complete mathematical specification |
| §6.6 Policy switching | gail.md Eq. 6 (switch condition) | Exact threshold formula from original |
| §6.7 Incremental learning | comp.md Eq. 10 (cooperative policy) | Exact parameterization from original |
| §10.1–10.3 Results | comp.md §6.3–6.6 (ablation methodology) | Validates experimental design choices |
| §IX Limitations/Future | comp.md §7 (limitations discussion) | Aligns future work with original authors' vision |

All LaTeX placeholders use report.md equations as primary, with source material equations inserted only where they provide authoritative validation or complete specifications not fully detailed in the report.

| Slide | LaTeX Equations Provided |
|-------|------------------------|
| 8 | $s_t \in \mathbb{R}^{4 \times 13\|B\|}$, $o_t \in \mathbb{R}^{5 \times 7\|B\|}$, $\tau = k_p(a_t - q) - k_d \dot{q}$ |
| 9 | $\mu \in \mathbb{R}^{28}$, $\log \sigma \in \mathbb{R}^{28}$, $r^D = \frac{1}{32} \sum \text{Clip}(D_i(o), -1, 1)$ |
| 10 | $L_{D_i}$, $L_{D_i}^{GP}$, $L_D$, $r_t$, $L^\pi$, $\tilde{A}_\beta$ full equations |
| 12 | Switch condition with $\frac{1}{N} \sum \text{Clip}(D_i^{target}, -1, 1) \geq \eta$ |
| 13 | $\pi(a_t \| s_t, g_t, a^{meta}_t) = \mathcal{N}(\mu_t + w_t \odot \text{Stop}(a^{meta}_t), \text{diag}(\sigma_t^2))$ |

All equations are sourced directly from comp.md  and gail.md  original papers, verified against the PDF versions for accuracy.