From Pixels to Physics: Imitation and Adaptation

of Human Motions from Casual Videos

submitted in partial fulfillment of the requirements

for the degree of

MASTER OF TECHNOLOGY
in

COMPUTER SCIENCE AND ENGINEERING
by

LOGESH .V CS24M106

Supervisor(s)

Chalavadi Vishnu

DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING
INDIAN INSTITUTE OF TECHNOLOGY TIRUPATI

April 26, 2026

DECLARATION

I declare that this written submission represents my ideas in my own words and where others’

ideas or words have been included, I have adequately cited and referenced the original sources.

I also declare that I have adhered to all principles of academic honesty and integrity and have

not misrepresented or fabricated or falsified any idea/data/fact/source in my submission to the

best of my knowledge. I understand that any violation of the above will be cause for disciplinary

action by the Institute and can also evoke penal action from the sources which have thus not

been properly cited or from whom proper permission has not been taken when needed.

Place: Tirupati
Date: April 26, 2026

Signature
Logesh .V
CS24M106

BONA FIDE CERTIFICATE

This is to certify that the report titled From Pixels to Physics: Imitation and Adaptation
of Human Motions from Casual Videos, submitted by Logesh .V, to the Indian Institute of
Technology, Tirupati, for the award of the degree of Master of Technology, is a bona fide record

of the project work done by him under my supervision. The contents of this report, in full or in

parts, have not been submitted to any other Institute or University for the award of any degree or

diploma.

Place: Tirupati
Date: April 26, 2026

Chalavadi Vishnu
Guide
Assistant Professor
Department
Science & Engineering
IIT Tirupati - 517619

of Computer

ABSTRACT

Generating physically plausible three-dimensional human motion from casual video footage

represents a fundamental challenge at the intersection of computer vision, graphics, and machine

learning. While kinematic motion generation methods produce visually reasonable poses, they

inherently violate physical constraints such as ground contact, torque limits, and momentum

conservation, rendering them unsuitable for interactive applications. This research addressed

the problem of bridging the gap between video-driven kinematic motion synthesis and physics-

based character animation through a novel two-stage pipeline. The primary purpose was to

develop an end-to-end system capable of converting unconstrained casual video into physically

valid simulated humanoid behaviors without requiring hand-crafted reward engineering or pre-

composed reference datasets. The methodology combined denoising diffusion probabilistic

models for video-to-motion generation with adversarial reinforcement learning for physics-

based imitation. Specifically, a modified ViMo-Flow architecture extracted three-dimensional

skeletal motions from two-dimensional pose trajectories using Min-SNR weighted loss functions

and vectorized timestep sampling strategies. The kinematic outputs were then retargeted to a

simulated humanoid in the MuJoCo physics environment through coordinate transformation,

bind rotation estimation, and proportional height scaling. An ensemble of body-part-specific

discriminators provided reward signals to a Proximal Policy Optimization trained control policy,

eliminating the need for manual reward design through a composite reward formulation based

on discriminator logits with learnable aggregation weights. The investigation also examined

bilateral symmetry regularisation and phase-conditioned observations as auxiliary training

mechanisms. Results demonstrated successful imitation of diverse locomotion and gesture

behaviors from decoupled motion sources, with the multi-discriminator approach achieving

robust performance without explicit target pose tracking or phase-based synchronization. The

trained policies exhibited stable control under moderate perturbations and supported runtime

policy switching through discriminator feasibility checks. This work established the algorithmic

foundations for composite multi-objective motion control and identified policy adaptation under

domain variation as the critical next step toward deployable simulation-ready character animation

systems.

i

TABLE OF CONTENTS

ABSTRACT

List of Figures .

. .

. .

List of Tables .

. .

. .

.

.

ABBREVIATIONS

I

II

Introduction .

.

.

.

Related Work .

. .

.

.

.

.

.

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

2.1

2.2

2.3

2.4

2.5

2.6

2.7

Video-to-Motion Generation . . . . . . . . . . . . . . . . . . . . .

Diffusion Training Dynamics

. . . . . . . . . . . . . . . . . . . .

Physics-Based Character Animation . . . . . . . . . . . . . . . . .

Adversarial Imitation Learning . . . . . . . . . . . . . . . . . . . .

Composite Motion and Task Control . . . . . . . . . . . . . . . . .

Symmetry and Phase Priors

. . . . . . . . . . . . . . . . . . . . .

Policy Adaptation . . . . . . . . . . . . . . . . . . . . . . . . . . .

III Overview .

. .

. .

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

IV Problem Statement

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . .

4.1

4.2

4.3

Task Definition . . . . . . . . . . . . . . . . . . . . . . . . . . . .

Motion Corpus and Data Preparation . . . . . . . . . . . . . . . .

Simulation Character and Reference Motion Format

. . . . . . . .

V

Vimo-Flow .

.

.

.

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

5.1

5.2

5.3

5.4

Diffusion Formulation . . . . . . . . . . . . . . . . . . . . . . . .

Network Architecture . . . . . . . . . . . . . . . . . . . . . . . . .

FiLM Conditioning . . . . . . . . . . . . . . . . . . . . . . . . . .

Sampling and Training Strategy . . . . . . . . . . . . . . . . . . .

VI Composite-Motion . .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

6.1

6.2

6.3

Motion Retargeting: SMPL to Simulation . . . . . . . . . . . . . .

Observation, Action, and Discriminator Interface . . . . . . . . . .

Policy, Value, and Discriminator Networks

. . . . . . . . . . . . .

ii

i

vi

vi

vii

1

4

4

5

5

6

7

8

8

9

11

11

11

12

15

15

17

17

18

19

19

20

21

6.4

6.5

6.6

6.7

Controller Learning . . . . . . . . . . . . . . . . . . . . . . . . . .

Composite Motion and Goal Control . . . . . . . . . . . . . . . . .

Interactive Policy Switching . . . . . . . . . . . . . . . . . . . . .

Incremental Learning . . . . . . . . . . . . . . . . . . . . . . . . .

VII Loss Functions

.

.

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

7.1

Motion Generation Objective . . . . . . . . . . . . . . . . . . . . .

7.1 (i)

6D Rotation Representation . . . . . . . . . . . . . . . .

7.1 (ii)

Standard Regularizations

. . . . . . . . . . . . . . . . .

7.1 (iii) SNR Integration . . . . . . . . . . . . . . . . . . . . . .

7.2

Adversarial Control Objectives . . . . . . . . . . . . . . . . . . . .

7.2 (i)

Discriminator Training Objective . . . . . . . . . . . . .

7.2 (ii)

Policy Optimization via PPO . . . . . . . . . . . . . . .

7.2 (iii) Auxiliary Regularisations . . . . . . . . . . . . . . . . .

VIII Adversarial Control Experiments . . . . . . . . . . . . . . . . . . . . . . .

8.1

8.2

8.3

8.4

8.5

Implementation details . . . . . . . . . . . . . . . . . . . . . . . .

Evaluation metrics . . . . . . . . . . . . . . . . . . . . . . . . . .

Case study I: phase-conditioned observations . . . . . . . . . . . .

Case study II: bilateral symmetry regularisation . . . . . . . . . . .

Scope of the present study . . . . . . . . . . . . . . . . . . . . . .

IX Conclusion and Future Work . . . . . . . . . . . . . . . . . . . . . . . . .

X

Results . .

. .

. .

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . . .

10.1

ICCGAN Humanoid Motions

. . . . . . . . . . . . . . . . . . . .

10.2 Vimo Full-Body Dance Motions (AIST Retargeted) . . . . . . . . .

10.3

Task-Based Locomotion Simulations (LaFAN1) . . . . . . . . . . .

References .

.

.

.

.

.

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . .

A Motion Generation Experiments . . . . . . . . . . . . . . . . . . . . . . .

B

ViMo-Flow: Qualitative Results

. . . . . . . . . . . . . . . . . . . . . . .

Appendix .

.

.

.

.

.

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . . .

22

22

23

23

24

24

24

24

25

25

25

26

27

28

28

28

29

30

33

36

37

37

41

44

50

51

55

51

iii

LIST OF FIGURES

GAIL: The policy switcher uses the character’s observation to decide whether to
keep the currently activated policy or to switch to the target policy interactively
responding to the external control signal when such a transition is considered
feasible.

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. .

. .

. .

.

.

COCO 17-keypoint skeleton order used for 2D pose conditioning.

. . . . .

Representative pose estimator output with confidence-masked joints. . . . .

SMPL’s 3D Joint Indicies Format (Tool used for Forward Kinematics in imple-
mentation).

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

.

.

.

.

.

.

Motion denoising pipeline. Given a 2D pose sequence c, the model starts from
noise mT and iteratively denoises mt for t = T ? 0 to obtain a 3D motion
sequence m0.

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

.

.

.

.

.

Vectorized timestep modeling for video diffusion.
timesteps increase flexibility, while shared timesteps preserve efficiency.

Independent per-frame

. .

Deepmimic Humanoid Structure used by Mujoco Tool (a) Joint map between
the Kinect model and character model; (b) The coronal plane projected by the
point cloud of the human body. . . . . . . . . . . . . . . . . . . . . . . . .

Deepmimic Humanoid Structure used by Mujoco Tool (a) Joint map between
the Kinect model and character model; (b) The coronal plane projected by the
point cloud of the human body. . . . . . . . . . . . . . . . . . . . . . . . .

Limp walk training comparison across baseline (baseline), phase-input
(phase), symmetry-regularized (sym), and combined (phase+sym) variants
at H = 8 and H = 16. Metrics include lifetime cycles, discriminator scores, gap,
value loss, policy loss, reward mean, and normalized symmetry loss.

. . . .

Jaunty walk training comparison across experimental variants. The phase-
conditioned variant (green) shows superior lifetime and reward characteristics
for this asymmetric motion, while symmetry regularization (orange) degrades
performance substantially.

. . . . . . . . . . . . . . . . . . . . . . . . . .

Joyful walk training comparison. The symmetry variant (orange) achieves
high lifetime cycles but at the cost of discriminator quality (lower fake scores),
indicating a survival-imitation trade-off.

. . . . . . . . . . . . . . . . . . .

1

2

3

4

5

6

7

8

9

10

11

12 Network architectures. ? denotes the concatenation operator and ? denotes the

average operator.

Jaunty walk .

.

.

.

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

Joyful walk . .

. .

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

Limp walk . .

. .

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . . .

13

14

15

iv

6

12

13

14

17

18

20

21

31

32

34

35

37

38

38

17

18

Squat – stable pose with legs fixed, zero slip observed . . . . . . . . . . . .

Punch – upper-body dominant action with minor terminal leg slippage . . .

16 Comparative training curves across ICCGAN motions . . . . . . . . . . . .

19

Leg lunge – controlled forward extension with minor slip at terminal phase

20 Kick – hip-driven swing with support leg instability leading to falls . . . . .

21

Long jump – takeoff and landing phases with speed-dependent stability . .

22 Roll – diving initiation captured; episode terminates upon ground contact (pelvis

< 0.15m) preventing full recovery . . . . . . . . . . . . . . . . . . . . . . .

23 Vimo training curves across dance genres.

. . . . . . . . . . . . . . . . . .

24

25

26

27

28

29

30

32

33

31

34

35

gBR/ch01 side-step falls

. . . . . . . . . . . . . . . . . . . . . . . . . . .

gHO/ch01 sliding legs

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

gJS/ch02 foot clipping collapse . . . . . . . . . . . . . . . . . . . . . . . .

gLH/ch01 partial slide success . . . . . . . . . . . . . . . . . . . . . . . .

gLO/ch02 stabilization over knee bends

. . . . . . . . . . . . . . . . . . .

gMH/ch02 leg-lift failure . . . . . . . . . . . . . . . . . . . . . . . . . . .

gPO/ch01 legs focus on stability . . . . . . . . . . . . . . . . . . . . . . .

locomotion_walk – target-following with moderate turn capability . . . . .

locomotion_crouch – wide-deviation turns via stabilizing leg-drag gait . . .

Task-based locomotion training curves across walk, run, and crouch policies.

locomotion_run – high-speed kangaroo hopping with stability failure . . . .

SMPL Humanoid (34-joints) to Deepmimic Humanoid (28-joints) Retargeted
Sample

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. .

. .

. .

.

.

36 Ablation of SNR weighting strategies for a T = 1000 linear diffusion process
1 ? ¯?t,
(? ? [1e ? 4, 0.02]). Plotted are the signal term
benchmark decay curves (linear, square root, and their average), and proposed
SNR variants: simple ratio
, min-capped decays, ratio-split=0.1 blends,
normalized, and scaled weights. Area-under-curve (AUC) percentages (cap=1.0)
quantify integrated weight magnitude. Normalized SNR achieves the highest
AUC (99.843%), while simple SNR decays precipitously. . . . . . . . . . .

¯?t and noise term

¯?t
1? ¯?t

?

?

37 Ground-truth motion sequence (small model v1, no PTSS).

. . . . . . . . .

38 Deterministic predictions ˆx0 at timesteps t ? {100, 200, 300, 400, 500} (small

model v1, no PTSS).

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

39

Stochastic samples at timesteps t ? {100, 200, 300, 400, 500} (small model v1,
no PTSS).

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. .

. .

. .

.

40 Ground-truth motion sequence (small model v1, PTSS). . . . . . . . . . . .

v

38

38

39

40

40

40

41

42

43

43

43

43

44

44

44

45

45

46

47

52

53

55

55

55

56

45

46

47

48

49

50

51

1

2

3

4

5

41 Deterministic predictions ˆx0 at t ? {50, 100, 150, 200, 250, 300, 350, 400} (small

model v1, PTSS).

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

42

Stochastic samples at t ? {50, 100, 150, 200, 250, 300, 350, 400} (small model
v1, PTSS). .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. .

. .

. .

43 Ground-truth motion sequence (small model v2, PTSS). . . . . . . . . . . .

44 Deterministic predictions ˆx0 at t ? {50, 100, 150, 200, 250} (small model v2,

PTSS). .

.

.

. .

. .

. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

Stochastic samples at t ? {50, 100, 150, 200, 250} (small model v2, PTSS). .

Example 1: ground-truth motion sequence (big model v1, PTSS). . . . . . .

56

56

57

57

57

57

Example 1: deterministic predictions ˆx0 at t ? {50, 100, 150, 200, 250, 300, 350, 400, 450}
(big model v1, PTSS). .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

58

58

58

59

59

12

35

38

43

45

Example 1: stochastic samples at t ? {50, 100, 150, 200, 250, 300, 350, 400, 450}
(big model v1, PTSS). . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

Example 2: ground-truth motion sequence (big model v1, PTSS). . . . . . .

Example 2: deterministic predictions ˆx0 at t ? {50, 100, 150, 200, 250, 300, 350}
(big model v1, PTSS). . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

Example 2: stochastic samples at t ? {50, 100, 150, 200, 250, 300, 350} (big
model v1, PTSS).

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

.

.

.

LIST OF TABLES

SMPL kinematic model vs. MuJoCo simulation humanoid.

. . . . . . . . .

Hyperparameters

.

.

.

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

ICCGAN motion imitation performance at convergence (Period 20). Highlights
used to substantiate inferences on replication fidelity, stability, and termination
behavior. .

. . . . . . . . . . . . . . . . . . . . . . . . . . . .

. .

. .

. .

.

Vimo dance motion performance at convergence (Period 30). Numerical high-
lights from training traces directly substantiate inferences on stability, converter
artifacts, and partial success.

. . . . . . . . . . . . . . . . . . . . . . . . .

Task-based locomotion performance at convergence.

. . . . . . . . . . . .

vi

ABBREVIATIONS

The following abbreviations are used throughout this thesis. They are listed in alphabetical

order.

AIST

DDPM

DOF

GAE

GAIL

GAN

GRU

AIST Dance Video Database

Denoising Diffusion Probabilistic Model

Degrees of Freedom

Generalised Advantage Estimation

Generative Adversarial Imitation Learning

Generative Adversarial Network

Gated Recurrent Unit

ICCGAN

Imitation from Composite Critic Generative Adversarial Network

IITT

Indian Institute of Technology, Tirupati

LaFAN1

Locomotion Animation Dataset (Ubisoft, version 1)

Min-SNR Minimum Signal-to-Noise Ratio (loss weighting)
MLP

Multi-Layer Perceptron

MuJoCo

Multi-Joint dynamics with Contact (physics simulator)

PD

PPO

RL

SMPL

ViMo

Proportional-Derivative (servo)

Proximal Policy Optimization

Reinforcement Learning

Skinned Multi-Person Linear (body model)

Video-to-Motion (diffusion-based generator)

vii

I

Introduction

Realistic 3D human motion forms the foundation of compelling virtual character animation. As

portable devices and immersive applications accelerate demand for scalable content creation,

generating adaptive motions from casual video remains challenging. Recent video-to-motion

methods synthesize diverse 3D motions directly from unconstrained footage despite camera

motion and occlusion, leveraging abundant online content to produce contextually consistent

results [25].

Data scarcity and domain limitations. Data-driven generative approaches—such as text-

to-motion diffusion models [35] and music-conditioned dance generation [28]—offer scalable

alternatives but inherit biases from limited datasets like AMASS [18], Human3.6M [15], and

AIST++ [16]. Models trained on narrow domains struggle to generalize to styles such as martial

arts or classical dance, reflecting the ongoing scarcity and high cost of collecting diverse 3D

motion data.

Casual video offers an escape from this scarcity. Platforms host billions of clips span-

ning ballet, occupational tasks, and impromptu performance. Unlike curated motion-capture

datasets, these recordings feature dynamic camera work, shot transitions, occlusion, and var-

ied environments—precisely the failure modes of conventional 3D pose estimators. Methods

that regress 3D joints from 2D observations rely on static or accurately estimable camera pa-

rameters [37, 34]. Under casual video conditions, frame-wise predictions accumulate errors,

producing jittery and implausible trajectories.

ViMo-Flow: generative video-to-motion. ViMo [25] reframes video-to-motion as a gener-
ative task. Given 2D pose trajectories p ? RS×J×3 from video, a diffusion model synthesizes
plausible 3D motion sequences without explicit camera estimation. Multi-view projections of

generated motions are compared against input 2D sequences; the denoising network operates

over full temporal windows, enforcing coherence through transformer self-attention.

Our implementation enhances this framework through two training advances. Min-SNR
weighting [9] reweights each timestep by its signal-to-noise ratio, emphasizing high-SNR steps
that contribute meaningful signal. Vectorized timestep sampling [17] encodes timesteps as
dense vectors rather than scalar indices, enabling the transformer to exploit long-range temporal

correlations for improved rhythmic alignment.

Kinematic versus physics-based methods. Kinematic approaches generate motions with-

out leveraging physical equations of motion [13, 12]. While high-quality animations can be

produced depending on dataset size, such methods fail under complex perturbations or environ-

mental variations. Physics-based methods instead perform simulation through a physics engine,

1

guaranteeing physical realism and enabling sim-to-real transfer [14, 22].

A central challenge in physics-based motion generation is controller optimization. Control

signals—whether direct joint torques or proportional-derivative (PD) servos—are typically

inaccessible when capturing motion in the real world. Deep reinforcement learning has emerged

as the dominant paradigm for obtaining general controllers without pre-assumed heuristics [4,

21, 29].

Adversarial imitation learning. Reward engineering for diverse motion styles is notoriously
brittle. We formulate an adversarial imitation scheme that eliminates hand-crafted rewards
through generative adversarial imitation learning (GAIL) [10, 20]. An ensemble of discrimina-
tors {Dk}K
k=1 learns to distinguish reference motion segments from policy-generated trajectories,
with discriminator outputs directly driving policy optimization:

rt =

K
?
k=1

wk · ? (Dk(st, at)),

where ? (·) denotes the sigmoid function and wk are learnable aggregation weights.

This GAN-like approach [30, 21] achieves state-of-the-art imitation performance without

manual reward design. Unlike motion-tracking methods [29, 1] that require explicit target pose

tracking or phase-based synchronization, our policy operates using only recent trajectory history,

eliminating the need for motion generation or matching mechanisms during policy transitions.

Composite multi-objective control. Humans perform sophisticated composite behaviors—

walking while gesturing, locomoting while manipulating objects—without examples of every
possible combination. We extend the adversarial framework to composite motion learning [31],
where each discriminator is assigned to a distinct body-part group and operates on its subset of

key links. The policy explores automatically how composite motions combine through weighted

reward aggregation, without requiring pre-composed reference clips.

Our multi-objective framework supports task-directed goals such as navigation targets

alongside imitation objectives. Decoupling full-body control during training transforms imitation

and goal-directed objectives into a unified learning problem where the policy discovers state-

dependent weightings dynamically.

Training methodology. The policy ?? is trained via PPO [26] with three algorithmic compo-
nents:

1. Motion retargeting: SMPL skeletal data is transformed to the simulation character through
coordinate system alignment, per-joint bind rotation estimation, and proportional height

scaling.

2

2. Bilateral symmetry regularisation: An ?2 penalty on mirrored sagittal-plane joint actions

reduces exploration space for symmetric gaits [33].

3. Phase-conditioned observations: Sinusoidal encodings (sin 2?? , cos 2?? ), with ? ? [0, 1)
tracking normalized cycle position, provide explicit temporal cues for looped motions.

Contributions.

1. A diffusion-based video-to-motion framework (ViMo-Flow) with Min-SNR weighting

and vectorized timestep sampling for kinematic synthesis from casual video.

2. An adversarial physics-based imitation pipeline employing body-part-specific discrimina-

tors with automatic reward aggregation, eliminating hand-crafted reward engineering.

3. A composite multi-objective learning framework enabling policy training from decoupled

motion sources without pre-composed reference clips.

4. A motion retargeting procedure mapping SMPL skeletal data to simulation characters via

coordinate transforms, bind rotations, and height scaling.

5. Empirical analysis of bilateral symmetry regularisation and phase-conditioned observa-

tions for locomotion learning.

A third stage—policy adaptation for robust deployment [32]—is scoped as future work.

3

II

Related Work

2.1 Video-to-Motion Generation

The field of human motion generation has progressed from action-label conditioning and

music-conditioned synthesis toward video-conditioned approaches. Action-conditioned methods

generate motions from categorical labels or seed poses, operating entirely within the training

distribution. Degardin et al. propose a generative adversarial graph convolutional network that

synthesizes human actions conditioned on discrete class labels, yet the model cannot extrapolate

beyond the 10–15 action categories present during training [5]. Similarly, Action2Motion [8]

employs a VAE-transformer architecture to generate 3D motions from action categories, but the

generated repertoire remains bounded by the predefined set of 30 actions in the NTU RGB+D

dataset. These approaches share a fundamental limitation: the conditioning signal is a one-hot

or learned embedding of a seen category. The model learns a mapping from label to motion

distribution but acquires no mechanism to compose unseen behaviors. The vocabulary of actions

is closed at training time.

Data-driven generative approaches—such as text-to-motion diffusion models [35] and music-

conditioned dance generation [28]—offer scalable alternatives but inherit biases from limited

datasets like AMASS [18], Human3.6M [15], and AIST++ [16]. Models trained on narrow

domains struggle to generalize to styles such as classical dance or martial arts, reflecting the

ongoing scarcity and high cost of collecting diverse 3D motion data.

Pose estimation from videos. Reconstructing 3D pose from monocular video follows a two-

stage pipeline: detect 2D keypoints, then lift to 3D via convolutional or graph architectures. Cai

et al. exploit spatial-temporal relationships through graph convolutional networks, aggregating

joint correlations across frames to improve 3D accuracy [2]. Martinez et al. demonstrate

that a simple MLP operating on 2D detections can achieve competitive results when camera

parameters are known [19]. More recent work, SMPLer-X [3], scales training data to 4.5 million

frames and predicts expressive SMPL-X parameters, yet relies on accurate camera calibration

or approximates intrinsics from image metadata. These methods assume static cameras or

easily estimable camera motion; under casual video with rapid zoom, pan, and shot cuts, the

2D-to-3D lifting objective becomes ill-posed. Frame-wise predictions accumulate drift, yielding

jittery trajectories that violate physical plausibility. GLAMR [34] attempts global trajectory

optimization to smooth results, but computational cost is high and performance degrades under

aggressive montage editing.

Generative video-to-motion. ViMo [25] reframes video-to-motion as a generative task rather

than a reconstruction problem. Instead of estimating exact 3D coordinates and camera pose, a

4

diffusion model synthesizes plausible motion sequences conditioned on the 2D pose trajectory

extracted from video. Multi-view projections of the generated motion are compared against

the input 2D sequence, bypassing explicit camera modeling. The denoising network operates

over the full temporal sequence, enforcing coherence through transformer self-attention. This

approach tolerates missing frames, occlusions, and rapid viewpoint changes while producing

diverse motions that match the stylistic silhouette and rhythm of the source video. Our video-to-

motion framework sidesteps the restriction of closed vocabularies by conditioning on 2D pose

trajectories rather than categorical tokens. A video of a never-before-seen dance style provides

a kinematic blueprint that the diffusion model translates into plausible 3D motion, effectively

enabling zero-shot generation of unseen categories.

2.2 Diffusion Training Dynamics

Standard DDPM training [11] applies uniform loss weighting across noise levels, causing
gradient imbalance where low-SNR steps contribute minimal learning signal. Min-SNR-? [9]
reweights by wt = min(SNRt, ?), emphasising high-noise timesteps where coarse pose and
trajectory structure is learned. This reduces training epochs by approximately 30% compared to

constant weighting.

Standard diffusion models treat timestep as a scalar, ignoring frame-to-frame temporal

structure and discarding periodicity and phase relationships inherent in motion [11]. The

Vectorized Timestep Approach [17] redefines this by encoding timestep as a dense vector

function of sequence position, allowing the transformer to capture long-range correlations

and rhythmic patterns. We introduce a Probabilistic Timestep Sampling Strategy to manage

computational cost. In naive vectorized diffusion, sampling distinct timesteps per frame leads to
a combinatorial explosion, with 1000N combinations for N frames. We introduce a probability
p governing independent versus shared sampling: with probability p, each frame receives its
own timestep for independent evolution; with probability 1 ? p, a single timestep is broadcast
across all frames, preserving efficiency. This hybrid balances temporal expressivity and training
cost. In practice, p = 0.3 balances diversity and stability, preventing overfitting to arbitrary
frame-wise noise schedules while retaining fine-grained temporal control. These techniques

underpin the ViMo-Flow training regime.

2.3

Physics-Based Character Animation

Kinematic approaches generate motions without leveraging physical equations of motion [13, 12].

Though high-quality animations can be generated depending on the amount and quality of

the motion dataset, kinematics-based methods may suffer problems when facing complex,

unpredicted perturbations or environmental variations. Physics-based methods, on the other

hand, perform simulation through a physics engine, and thus guarantee the physical realism of

5

the generated motions [27]. Such methods can also enable sim-to-real transfer and apply motion

generation techniques on physical robots.

A challenge in physics-based methods for realistic motion generation is that a controller

is needed to be optimized either directly by applying joint torques or indirectly through, for

example, proportional-derivative (PD) servos to help the character reach a desired pose. However,

those control signals are typically inaccessible when we capture motion in the real world. In

recent years, reinforcement learning has been widely used to perform the optimization in order

to obtain a general controller without pre-assumed heuristic rules. Data-driven methods for

imitation learning under the framework of deep reinforcement learning have achieved state-of-

the-art performance and are able to generate high-quality motions [4, 21, 29].

DeepMimic [21] demonstrated RL-based motion imitation using hand-tuned per-joint re-

ward terms and an explicit phase variable synchronizing the policy with the reference clip.

ScaDiver [29] scales to larger motion datasets by clustering behaviours but retains manual re-

ward design. However, when facing an interactive control demand, methods that rely on motion

tracking usually need a motion generation or matching mechanism to ensure the transition

between two different behaviors or motor control policies [29, 1].

Adversarial Motion Priors (AMP) [23] replace explicit rewards with a learned discriminator

that distinguishes policy rollouts from reference data, yielding stylistic control from unstructured

datasets without per-task reward engineering.

2.4 Adversarial Imitation Learning

Generative Adversarial Imitation Learning (GAIL) [10] adapts the GAN framework to policy

optimization: a discriminator classifies state transitions as expert or agent-generated, and the

classification signal drives policy improvement. Direct application to physics-based characters

suffers from reward vanishing when the discriminator overpowers the early-stage policy.

Figure 1: GAIL: The policy switcher uses the character’s observation to decide whether to keep
the currently activated policy or to switch to the target policy interactively responding
to the external control signal when such a transition is considered feasible.

The ICCGAN formulation [30] stabilises training through three mechanisms: (i) an en-

semble of discriminators with shared feature layers but independent output heads, resisting
overfitting; (ii) hinge-loss optimization bounding scores to [?1, 1]; and (iii) gradient penalty

6

regularisation [7] enforcing Lipschitz smoothness. The discriminator ensemble score serves

directly as the RL reward for PPO [26], eliminating manual reward functions. The policy uses

a GRU-based recurrent encoder operating on recent root-relative body-part observations, per-

forming causal inference without phase counters or target poses—enabling interactive runtime

switching between behaviours.

This GAN-like approach achieves state-of-the-art imitation performance without manually

designing and fine-tuning a reward function. The policies directly control the character without

having to track any target reference pose explicitly or implicitly through a phase state. The system

can respond to external control signals provided by the user and interactively switch between

different policies without requiring any motion generation or motion matching mechanism.

2.5 Composite Motion and Task Control

Single-clip imitation cannot produce composite behaviours such as locomotion with simulta-

neous upper-body manipulation. Despite significant advancements in physics-based character

control, the majority of existing techniques rely on reference data consisting of motion cap-

ture recordings of an expert performing the behavior of interest. While such reference data is

paramount to train motor control policies that lead to natural and robust control, we are inter-

ested in synthesizing composite behaviors for physically simulated humanoids by combining

multiple motion capture reference clips into the training of a single policy. Further, we augment

these imitation controllers with task-specific rewards to train the policy to accomplish specific

functional tasks at the same time.

Assigning separate discriminators to body-part groups and aggregating their rewards via

multi-objective weighting [31] addresses this limitation. The core difference from existing

imitation learning approaches is decoupling full-body control during training, turning imitation

and goal-directed full-body training into a multi-objective learning framework. We extend GAN-

style reinforcement learning and introduce a multi-objective learning framework to support

multiple discriminators and automatic weighting of imitation and goal-driven subtask rewards.

We propose an incremental learning scheme that uses a meta-policy from an existing behavior

to augment the behavior with new subtasks, producing a composite motion control policy that

can be learned significantly faster than learning from scratch. Our scheme automatically learns

weights across the body that are state dependent in order to effectively mix the original behavior

with a new subtask in a temporally dynamic fashion.

A multi-critic value function with per-component heads stabilises learning under competing

gradients. Goal-conditioned extensions introduce spatial target-reaching and directional align-

ment rewards, while incremental training reuses pre-trained locomotion policies as meta-policies

for efficient composite skill acquisition. The framework supports composite multi-objective con-

trol policies trained from decoupled motion sources without requiring pre-composed reference

7

clips.

2.6

Symmetry and Phase Priors

Bilateral symmetry losses [33] penalise differences between mirrored left-right joint actions,

producing symmetric low-energy gaits without motion-capture data. Phase-functioned neural

networks [13] condition locomotion controllers on the gait cycle phase, encoding contact timing
as (sin 2?? , cos 2?? ). Both serve as auxiliary signals explored in our training pipeline, with
empirical analysis presented in the experiments.

2.7

Policy Adaptation

A trained imitation policy may degrade under environmental change—new terrains, altered

morphology, or modified task objectives. AdaptNet [32] proposes a two-tier adaptation hierarchy:

the first tier augments latent state embeddings for modest behavioural shifts, the second modifies

deeper network layers for substantial changes, achieving rapid adaptation while preserving

stylistic qualities. This constitutes the planned third stage of our pipeline.

8

III Overview

We present a novel two-stage approach to generating physically plausible 3D human motions

directly from casual videos. The pipeline overcomes challenges posed by complex camera

movements, occlusions, and the scarcity of labeled motion data, while ensuring that generated

motions satisfy physical constraints such as contact dynamics, torque limits, and momentum

conservation. At a high level, the system transforms multi-view 2D pose sequences extracted

from videos into temporally coherent kinematic motion sequences, then translates these into

physics-based control policies that drive a simulated humanoid under rigid-body dynamics.

Stage 1 — Video-to-Motion Generation. The core of the framework is a ViMo-style [25]

denoiser that takes time-series 2D pose observations as input and outputs per-frame 3D motion

data, including joint rotations, discrete foot-contact signals, and root trajectory. The denoiser

is trained within a DDPM [11] framework, enhanced with SNR-based loss weighting and a

probabilistic, vectorized timestep sampling strategy [9, 17]. This allows the model to synthesize

diverse and realistic motions from casual video without explicit camera calibration, learning

meaningful motion structure across all noise levels and generalizing well to unseen video

content.

Stage 2 — Physics-Based Imitation and Control. Kinematic outputs from Stage 1 undergo

a pre-processing phase where they are synthesized into reference motions and retargeted from

the SMPL skeleton representation to a MuJoCo [27] humanoid model through coordinate system

transformations, per-joint bind rotations, and height scaling. We create a library of physics-

based controllers by training corresponding action policies using a GAN-like approach under

the GAIL [10] framework. As such, we do not have to manually design a reward function for

imitation learning. The training framework operates as follows:

An ensemble of body-part-specific discriminators learns to distinguish reference motion

segments from policy-generated trajectories. The discriminator output serves directly as the

reinforcement learning reward signal, eliminating hand-crafted reward engineering. Formally,
the composite reward at timestep t is computed as:

rt =

1
K

K
?
k=1

clip(cid:0)Dk(ot?H:t), ?1, 1(cid:1),

where Dk denotes the k-th discriminator operating on observation window ot?H:t, and clip(·)
bounds scores to [?1, 1] per hinge-loss optimization [30, 7]. The policy ?? is trained via
PPO [26] to maximize cumulative discriminator scores while satisfying physical constraints.

For composite behaviours such as locomotion with simultaneous upper-body manipula-

tion, we decouple full-body control during training by assigning separate discriminators to

distinct body-part groups. Each discriminator operates on its subset of key links, and their

9

rewards are aggregated via multi-objective weighting [31]. A multi-critic value function with

per-component heads stabilises learning under competing gradients. This enables the policy

to explore automatically how composite motions combine without requiring pre-composed

reference clips.

Runtime Interactive Control. During execution, the user can select a target behavior via

input signals. The system responds by attempting to switch the current policy to the target

one. Policies trained with our approach perform inference based only on the short-term pose

trajectory of the character. Therefore, they can directly take over the character when given a pose

trajectory similar to that in the reference motions, without having to track any target reference

pose. We exploit the discriminators trained with the target policy as a policy switcher to measure

the similarity of the current pose trajectory to the reference motions, and decide if the transition

is feasible based on the discriminator score.

Incremental Learning. To accelerate training for composite behaviors, we employ an

optional incremental learning scheme that reuses a pre-trained policy as the meta policy and

trains a cooperative policy that adapts the meta one for new composite tasks. This significantly

reduces training time compared to learning from scratch, as the cooperative policy only needs to

learn the residual adaptation for the new subtask while preserving the base behavior.

10

IV

Problem Statement

4.1 Task Definition

Motion synthesis objective. Given a casual video of a single person, 2D pose sequences are
extracted, yielding a conditioning tensor p ? RS×J2d×3, where S = 150 frames (5 s at 30 FPS)
and J2d = 17 COCO-format joints with per-joint (x, y, conf). Missing or low-confidence joints
are masked. The objective is to generate a 3D motion sequence M ? RS×(J3d·C+4+3) comprising:

• Joint rotations R ? RJ3d×6: J3d = 24 joints in 6D rotation space [36],

• Foot contact labels f ? {0, 1}4 for left/right ankle and foot,

• Root position t ? R3 (global translation).

Control objective. The generated motion M, together with curated motion clips from AIST++,
serves as reference data for physical imitation. A control policy ?? : st (cid:55)? at drives a physically
simulated MuJoCo humanoid to reproduce these motions under rigid-body dynamics, formulated

as the MDP

J(? ) = E??

(cid:35)

?t rt

,

(cid:34) T
?
t=0

where state st encodes the character’s recent pose history, action at ? R28 specifies PD controller
targets, and reward rt is derived from adversarial discriminator scores rather than manually
engineered functions.

4.2 Motion Corpus and Data Preparation

AIST++ Dataset. The AIST++ benchmark [16] provides 1,408 sequences of 3D dance motions

across 10 genres with synchronized multi-view video and annotations. Each sequence stores
SMPL pose parameters (smpl_poses ? RN×24×3 axis-angle) and global translation (smpl_trans ?
RN×3). Nine calibrated camera views supply 2D keypoints at 1920×1080 px, 60 FPS, in COCO-
17 format with per-joint confidence.

2D Pose Extraction. Pose extractors such as OpenPose and AlphaPose provide per-joint

confidence scores used to mask unreliable detections; in our pipeline, the confidence field is used

directly for reliability-aware conditioning [6]. In multi-view scenarios the highest-confidence

detection per joint is selected, and optional temporal smoothing is applied conservatively to

preserve micro-motions [25].

Preprocessing. Raw sequences are resampled from 60 FPS to 30 FPS (every second frame).
Each sequence is partitioned into non-overlapping 150-frame clips. The conditioning tensor p

11

Figure 2: COCO 17-keypoint skeleton order used for 2D pose conditioning.

is normalized by centering relative to the root joint, removing absolute pixel coordinates and

enabling generalisation across cameras and video scales. Ground-truth 3D motion is encoded as
150 × 151 per clip: 144 dims for 6D rotations [36], 4 dims for foot contact labels (derived from
near-zero vertical foot/toe velocities), and 3 dims for root translation. Sequences in an ignore

list and SMPL axis-angle rotations are orthogonalized to 6D form [36].

4.3

Simulation Character and Reference Motion Format

MuJoCo Humanoid. The simulation character is a rigid-body humanoid in MuJoCo [27] with

15 bodies and 28 actuated DOF. Table 1 compares it to the SMPL kinematic model used by the

motion generator.

Table 1: SMPL kinematic model vs. MuJoCo simulation humanoid.

Property

SMPL

MuJoCo Humanoid

Total joints/bodies
Actuated DOF
Root representation
Coordinate system
Collarbones
Hands
Spine
Contact model

24
72 (axis-angle)
translation + axis-angle
Y-up (right-handed)
explicit (L/R_Collar)
articulated wrists
3 segments (Spine1–3)
none (mesh)

15
28 (hinge/ball)
3D position + quaternion
Z-up (right-handed)
fused into torso
rigid end-effectors
single torso body
rigid-body + friction

12

Figure 3: Representative pose estimator output with confidence-masked joints.

The DOF reduction from 72 to 28 collapses the three spine segments, collarbones, and

detailed hand joints into simplified rigid bodies while preserving the major bilateral chains

(arms, legs). This simplification is necessary for stable physics simulation: attempting to control

72 independent motors on a ragdoll-style body leads to instability [21].

Reference Motion Format. The Composite Motion framework expects reference data in
DeepMimic JSON format—a per-frame array of the character’s full qpos vector. Each frame
records:

• Pelvis world position p0 ? R3 and orientation q0 ? R4 (quaternion, [x, y, z, w]),

• Local quaternion orientations for the 14 non-root bodies.

This format directly encodes the MuJoCo generalized coordinates, ensuring compatibility with

the pretrained discriminator checkpoints and reference-motion samplers.

SMPL-to-Humanoid Data Bridge. ViMo-Flow outputs and AIST++ mocap clips are

both stored in SMPL representation. A retargeting converter maps these to the DeepMimic

format through coordinate transformation, height scaling, bind-rotation correction, and chain

composition. Key considerations motivating the conversion are:

• SMPL joint axes and coordinate conventions are incompatible with MuJoCo’s Z-up frame.

• Bone length proportions differ between the SMPL neutral body and the MuJoCo humanoid

XML.

• Multi-joint SMPL chains (e.g., three spine segments) must be composed into single

MuJoCo bodies with appropriate bind corrections.

13

Figure 4: SMPL’s 3D Joint Indicies Format (Tool used for Forward Kinematics in implementa-

tion).

14

V

Vimo-Flow

5.1 Diffusion Formulation

The denoising process follows a standard Markovian corruption of the clean motion m0. The
forward transitions are defined as

q(mt|mt?1) = N (

?tmt?1, (1 ? ?t)I),

?

producing mt at each step t. Unlike the original DDPM parameterization, which predicts the
noise term ?, the network D(mt,t, p) directly outputs an estimate of the clean motion ˆm0. This
formulation is better suited to the structured output space of 6D rotations [36], root translation,

and binary contact labels, where direct clean-motion prediction avoids additional decoding
constraints. The training objective minimizes the simple ?2 loss

Lsimple = E[?m0 ? D(mt,t, p)?2],

averaged over random timesteps and paired samples. At inference, the posterior mean ˜µt is
computed from ˆm0 using the closed-form expression

?

¯?t?1?t
1 ? ¯?t

˜µt =

ˆm0 +

?

?t(1 ? ¯?t?1)
1 ? ¯?t

mt,

enabling deterministic DDIM sampling or stochastic DDPM sampling.

Algorithm 1 Forward Diffusion Sampling
Require: Clean input x0, timestep vector t, noise schedule {?t}T
Ensure: Noisy sample xt

t=1

1: Initialize ?t = 1 ? ?t for t = 1, . . . , T
2: Compute ¯?t = ?t
i=1 ?i
3: Sample ? ? N (0, I) with dim(?) = dim(x0)
4:

¯?t ? gather( ¯?, t)

if dim(t) = 1 then

5:

6:

¯?t ? reshape( ¯?t, (B, 1, 1)) else

¯?t ? reshape( ¯?t, (B, S, 1))
?
?

¯?t · x0 +

1 ? ¯?t · ?

7:
8: xt ?
9: return xt

? Cumulative product

? Select ¯? values for given timesteps

? dim(t) = 2 PTSS case

15

Algorithm 2 Reverse Denoising Sampling
Require: Model M , initial state specification xT , total steps T , conditioning input p2d, stochas-

? Cumulative product

? Clone input tensor

? Set model to evaluation mode

? Batch timestep vector
? Model prediction

? Posterior mean

? Posterior variance

? Stochastic update else

? Deterministic update

? Final denoised estimate

? Restore model to training mode

ticity flag ? f lag, noise schedule {?t}T
Ensure: Generated motion sequence x0
1: Initialize ?t = 1 ? ?t for t = 1, . . . , T
2: Compute ¯?t = ?t

i=1 ?i
if xT is dimension tuple then

t=1

xt ? N (0, I) with shape xT else

3:

4:

xt ? xT

5:
6: Set batch size B ? dim(xt, 0)
7: M .eval()

for t = T ? 1 down to 0 do

8:

t ? vector of length B with value t

9: ˆx0 ? M (xt, t, p2d)
if t > 0 then

10:

¯?t ? ¯?[t], ¯?t?1 ? ¯?[t ? 1]
?

¯?t?1 · ?t
11: c1 ?
1? ¯?t
?t · 1? ¯?t?1
12: c2 ?
1? ¯?t
13: µ ? c1 · ˆx0 + c2 · xt

?

if ? f lag = True then

14:

? 2 ? ?t · 1? ¯?t?1
1? ¯?t

15: ? ? N (0, I)
?
16: xt ? µ +
17:

? 2 · ?

18: else
19:

xt ? µ

xt ? ˆx0

20:
21:
22:M .train()
23:return xt

16

Figure 5: Motion denoising pipeline. Given a 2D pose sequence c, the model starts from noise

mT and iteratively denoises mt for t = T ? 0 to obtain a 3D motion sequence m0.

5.2 Network Architecture

The denoiser stacks three transformer blocks, each containing self-attention, cross-attention, and

an MLP. The first self-attention block leverages temporal context from the noisy motion input.

Cross-attention conditions the model on 2D pose tokens, and FiLM blocks modulate intermediate

activations using scale and shift parameters derived from timestep and pose encoding. A final

linear layer projects the latent representation back to the motion space. The architecture is

designed to progressively recover the clean sequence while preserving long-range temporal

structure.

Two FiLM strategies were examined:

• Version 1 computes one pair of gamma/beta parameters per block and shares them across

the self-attention, cross-attention, and MLP paths.

• Version 2 uses separate gamma/beta predictors for each modulation point, allowing finer

control at the cost of higher parameter count.

Both versions broadcast the same modulation parameters across all frames in a sequence,

enforcing globally consistent conditioning.

5.3

FiLM Conditioning

Feature-wise Linear Modulation applies an affine transformation

y = ?(c) ? x + ? (c)

to intermediate activations x, where ?, ? are generated from a condition vector c [24]. In this
work, c is the concatenation of a timestep embedding embed(t) and a pose-sequence encoding

17

encode(p). The encoding network processes the full 2D trajectory through a lightweight
temporal transformer and produces a compact representation of pose geometry and rhythm.

FiLM then converts this representation into modulation parameters that steer denoising toward

motions consistent with the input video.

5.4

Sampling and Training Strategy

Classifier-free guidance [25] is employed during both training and generation. The condition p
is randomly dropped with probability 0.25, allowing the model to learn both conditioned and
unconditioned distributions. At sampling time, the final estimate combines both modes:

ˆm0 = (1 + w)D(mt,t, p) ? wD(mt,t, ?),

where w controls adherence to the input versus diversity.

Figure 6: Vectorized timestep modeling for video diffusion. Independent per-frame timesteps

increase flexibility, while shared timesteps preserve efficiency.

To prevent combinatorial explosion when using vectorized timesteps, PTSS [17] is intro-
duced. With probability p, each frame receives an independently sampled timestep; otherwise,
a single timestep is broadcast across all frames. This hybrid schedule bounds the number of

distinct noise patterns while preserving frame-wise temporal flexibility.

18

VI

Composite-Motion

6.1 Motion Retargeting: SMPL to Simulation

Generated motions and AIST++ clips are both represented in SMPL format (24 joints, axis-angle,

Y-up), whereas the MuJoCo humanoid requires Z-up quaternion orientations for 15 rigid bodies.

Five steps bridge the two representations.

Step 1 — Coordinate transform. SMPL uses (+X = left, +Y = up, +Z = forward), whereas
MuJoCo uses (+X = forward, +Y = left, +Z = up). The change of basis is the cyclic permuta-
tion (x, y, z) (cid:55)? (z, x, y):

G =

?

?
?

0 0 1

?

1 0 0

?
? ,

0 1 0

pmj = G psmpl,

Rmj = G Rsmpl G?.

Step 2 — Height scaling. Limb proportions differ between the two skeletons. A global scale

factor

s =

Lpelvis?foot
mj
Lpelvis?foot
smpl

is computed from rest-pose pelvis-to-foot lengths. All translated root positions are multiplied by
s before coordinate conversion.

Step 3 — Floor correction. The floor level is estimated as the minimum weighted average of
ankle (w = 1/4) and foot (w = 3/4) heights over all frames, computed through SMPL forward
kinematics. Subtracting this offset aligns the feet with the simulation ground plane.

Step 4 — Bind rotations. Because the MuJoCo rest-pose bone directions differ from SMPL,
a bind rotation R(b)
bind is computed for each body b using Rodrigues’ formula:

R = I + [v]× +

[v]2
×
1 + c

,

v = ˆa × ˆb,

c = ˆa · ˆb.

The corrected local quaternion for body b with parent p is

qb = q(p)

bind ? qsim

b ? q(b)
bind.

19

Step 5 — Chain composition and export. Multi-joint SMPL chains such as Spine1+Spine2+Spine3

or Neck+Head are composed as

qsim
b = q j1 ? · · · ? q jm

before bind correction. The final output is exported as a DeepMimic-style JSON sequence at 30

Hz.

Figure 7: Deepmimic Humanoid Structure used by Mujoco Tool (a) Joint map between the
Kinect model and character model; (b) The coronal plane projected by the point cloud
of the human body.

6.2 Observation, Action, and Discriminator Interface

Policy observation. The policy observes H consecutive frames (default H = 4) of root-relative
body-link features. Each link contributes a 3D position and a 4D quaternion relative to the root,
yielding per-frame dimension 7L for L links. The resulting sequence st ? RH×7L is normalized
by a running mean-variance tracker with clipping at ±5? . For goal-conditioned tasks, a goal
vector gt ? RG is appended after temporal encoding rather than duplicated across the sequence.

Action space. The action vector at ? R28 specifies target generalized coordinates for stable
PD servos:

? = kp(at ? q) ? kd ?q.

This indirect actuation preserves simulation stability and matches the control interface used by

the humanoid model.

Discriminator observation. Each discriminator Dk receives a slightly longer observation
window of length Hk = H + 1 and processes root-relative features restricted to its body-part
subset Bk, producing

ok ? RHk×7|Bk|.

20

Using short pose trajectories rather than isolated frames makes discriminator feedback sensitive

to both instantaneous pose quality and short-horizon motion continuity.

6.3

Policy, Value, and Discriminator Networks

Actor-critic model. Both actor and critic share the same temporal encoding backbone:

• A GRU encoder (input 7L, hidden dimension 256) processes the H-frame observation

sequence.

• An MLP (256+G ? 1024 ? 512) with ReLU6 activations maps the temporal embedding,

concatenated with the goal vector when present, to task outputs.

• The actor predicts µ ? R28 and log ? ? R28, defining

?? (a | s) = N (cid:0)µ, diag(? 2)(cid:1).

• The critic predicts one value head per reward component, and the targets are normalized

with DiagonalPopArt to stabilize learning under heterogeneous reward scales.

Discriminator ensemble. For each body group k, the discriminator uses a GRU encoder
(hidden dimension 256) followed by an MLP [256 ? 256 ? 128 ? 32] (Figure 12). The final
layer is interpreted as an ensemble of N = 32 scalar heads that share the feature extractor but
2 for hidden
retain independent output projections. Weights are initialized orthogonally (gain

?

layers and gain 1 for output heads), and observations are normalized online.

Figure 8: Deepmimic Humanoid Structure used by Mujoco Tool (a) Joint map between the
Kinect model and character model; (b) The coronal plane projected by the point cloud
of the human body.

21

6.4 Controller Learning

Controller learning follows a GAN-like adversarial loop under the GAIL framework [10, 30]. A

policy rollout produces state-action transitions and discriminator observation windows, while ref-

erence clips provide matched motion samples. Discriminators are updated to separate reference

windows from policy windows; PPO then updates the policy to maximize the discriminator-

derived reward. This removes the need for hand-designed imitation rewards and allows causal

control from short pose histories alone.

Algorithm 3 Adversarial Controller Learning
Require: Policy ?? , discriminators {Dk}K
Ensure: Trained policy and discriminator ensemble
1: Initialize policy replay buffer T and discriminator replay buffer B while training has not

k=1, reference motion set M

converged do

each environment step t

t?Hk:t

2: Sample action at ? ?? (st)
3: Step simulator and collect (st, at, st+1)
4: Construct discriminator windows o(k)
5: Compute imitation rewards from clipped discriminator scores
6: Sample matching reference windows from M
7: Append policy transitions to T and discriminator samples to B
8:
9: Update all discriminators on B with hinge loss and gradient penalty
10: Update ?? and value heads on T with PPO
11: Clear buffers
12:

6.5 Composite Motion and Goal Control

To learn composite behaviours from multiple references, full-body imitation is decoupled into
body-part objectives. Each discriminator Dk is assigned to a subset Bk of key links. These
groups are typically localized by function (for example, lower body, upper body, or object-

manipulating limbs) and may share links when coordination requires overlap. The rollout reward

can be written as

rt =

K
?
k=1

wk r(k)

t +?
j

w j r(task, j)
t

,

?
k

wk +?
j

w j = 1,

but policy optimization is not driven by this scalar alone. Instead, each objective maintains

its own critic and its own advantage stream, so balancing occurs at the level of standardized

per-objective updates [31].

For locomotion and navigation tasks, goal rewards are appended alongside imitation rewards.

22

A typical spatial objective is

with goal vector

t = exp(cid:0)???pt ? ptarget?2
rgoal

(cid:1),

gt = (?x, ?y, cos ? , sin ? ).

This formulation allows the controller to preserve motion style while satisfying task constraints.

6.6

Interactive Policy Switching

Interactive control is implemented through policy switching rather than target-pose tracking.

Each behavior is associated with its own policy-discriminator pair. At runtime, the discriminators

of a target policy evaluate whether the current short pose trajectory is compatible with the target

reference manifold. A switch is accepted when

1
N

N
?
i=1

clip(cid:0)D(i)

target(ot?Hk:t), ?1, 1(cid:1) ? ?,

where ? is a behavior-dependent threshold. Because the policy conditions only on recent pose
history, a feasible transition does not require explicit phase counters, motion matching, or

target-pose synthesis.

6.7

Incremental Learning

For harder composite tasks, training can be accelerated by reusing a pretrained controller as a

frozen meta policy and learning only the residual adaptation needed for the new objective. Let
ameta
t
as

denote the action sampled from the meta policy. The cooperative policy is parameterized

?(at | st, gt, ameta

t

) = N (cid:0)µt + wt ? stopgrad(ameta

t

), diag(? 2

t )(cid:1) ,

where wt is a learnable per-DOF mixing weight. The stop-gradient operator prevents the
cooperative update from modifying the frozen meta controller. This formulation preserves the

base behavior while allowing the new policy to inject localized task-specific corrections [31].

23

VII

Loss Functions

7.1 Motion Generation Objective

7.1 (i) 6D Rotation Representation

Joint rotations are encoded as 6D vectors capturing the first two columns of the orthonormal
rotation matrix. A 3D rotation R ? R3×3 is represented as [a1, a2] ? R6, where a1 and a2 are
orthonormal column vectors. The third column is recovered via cross product a3 = a1 × a2.
This representation is continuous, avoids the gimbal lock of Euler angles, and eliminates the

unit-norm constraint of quaternions, making it better suited for gradient-based optimization [36].

7.1 (ii) Standard Regularizations

Three auxiliary losses enforce physical plausibility.

3D Positions Loss penalizes discrepancy in global joint positions:

Ljoints =

1
S

S
?
i=1

?FK(Ri

0) ? FK( ˆRi

0)?2
2,

where FK denotes forward kinematics converting rotations to joint positions.

Velocity Loss enforces temporal smoothness via finite differences:

Lvel =

1
S ? 1

S?1
?
i=1

?(Ri+1

0 ? Ri

0) ? ( ˆRi+1

0 ? ˆRi

0)?2
2,

approximating angular velocity in 6D space.

Foot Contact Loss prevents foot sliding by masking joint velocities with predicted contact

labels:

Lfoot =

1
S ? 1

S?1
?
i=1

?(FK( ˆRi+1

0

) ? FK( ˆRi

0)) · ˆfi?2
2,

where ˆfi ? {0, 1}4 indicates left/right foot/toe ground contact. If ˆfi = 1, the velocity must be
near zero.

Traditional methods compute ˆfi via heuristic thresholds on foot velocity. ViMo-Flow learns
ˆfi as part of the model output without direct supervision; the only signal comes from Lfoot,
encouraging the model to predict contact labels that minimize sliding. This self-supervised

approach avoids arbitrary thresholds and adapts to the model’s own motion predictions.

24

7.1 (iii) SNR Integration

Training directly predicts clean motion ˆm0 rather than noise. The simple diffusion loss, together
with all auxiliary losses (Ljoints, Lvel, Lfoot), is reweighted by timestep-dependent SNR to
balance contributions across the noise schedule. The SNR at timestep t is SNRt = ¯?t/(1 ? ¯?t).
Pure SNR weighting wt = SNRt produces a steep cliff where most timesteps contribute near-zero
gradient, yielding poor sample efficiency.

Our implementation adopts a min-cap strategy that automatically selects the appropriate

weighting across the diffusion process:

(cid:32)

(cid:114)

wt = min

SNRt,

1 ?

(cid:33)

.

t
T

This formulation provides a smooth, monotonic ceiling: at early timesteps (high noise), the
square-root decay caps the potentially large SNRt to prevent gradient explosion; at late timesteps
(low noise), SNRt becomes small and dominates, avoiding vanishing gradients. The min
operation preserves the natural emphasis on coarse-structure recovery while keeping all timesteps

active during training.

The overall training objective becomes:

L = Et[wt?m0 ? ˆm0?2

2] + ?1Et[wtLjoints] + ?2Et[wtLvel] + ?3Et[wtLfoot].

Weights wt are normalized per batch to stabilize training. This square-root-capped Min-SNR
strategy preserves sample efficiency, prevents collapse to mean poses, and accelerates conver-

gence by emphasizing timesteps where the signal structure is most informative [9].

7.2 Adversarial Control Objectives

7.2 (i) Discriminator Training Objective

Each body-group discriminator Dk is trained with hinge loss to separate reference (real) from
policy-generated (fake) motion windows. Since each discriminator is implemented as an
ensemble of N = 32 scalar heads, the training objective is written head-wise as

LDk =

1
N

N
?
i=1

(cid:16)
Eo?ref
(cid:124)

(cid:2)max(0, 1 ? D(i)
(cid:123)(cid:122)
real loss

k (o))(cid:3)
(cid:125)

+ Eo??
(cid:124)

(cid:2)max(0, 1 + D(i)
(cid:123)(cid:122)
fake loss

(cid:17)
k (o))(cid:3)
(cid:125)

+ ?gp · GPk,

(1)

where the gradient penalty [7]

GPk = Eˆo

(cid:2)(cid:0)??ˆo

N
?
i=1

k (ˆo)?2 ? 1(cid:1)2(cid:3),
D(i)

ˆo = ? oref + (1??) o? , ? ? U(0, 1),

25

enforces Lipschitz smoothness with ?gp = 10. Hinge loss bounds scores to [?1, 1]: real samples
are pushed toward +1, fake samples toward ?1. Once a sample is correctly classified with
margin ? 1, the corresponding hinge term saturates, preventing discriminator over-training on
already-separated samples and preserving informative rewards [30].

The reward signal per discriminator is the mean over 32 ensemble heads:

r(k)
t =

1
32

32
?
i=1

clip(cid:0)D(i)

k (ot?Hk:t), ?1, 1(cid:1).

7.2 (ii) Policy Optimization via PPO

The policy is optimised with PPO [26]. Advantage estimates ˆAt are computed via GAE:

ˆA(i)
t =

Hsteps?1?t
?
l=0

(?? )l ? (i)
t+l,

t = r(i)
? (i)

t + ?V (i)(st+1) ?V (i)(st).

For multi-objective training, each advantage stream is standardized independently across the

rollout batch:

¯A(i)
t =

ˆA(i)
t ? µi
?i + ?

,

where (µi, ?i) are the batch mean and standard deviation of objective i. The weighted advantage
is then

The clipped surrogate objective is

¯At = ?

i

wi ¯A(i)
t

.

L clip = ?Et

(cid:2)min(cid:0)?t ¯At, clip(?t, 1??, 1+?) ¯At

(cid:1)(cid:3) ,

?t =

?? (at|st)
??old(at|st)

,

with ? = 0.2. The value loss is

L value = ?

(cid:0)V (i)

? (st) ? R(i)

t

(cid:1)2,

i

where targets R(i)
t
scales of discriminator and task rewards. This per-objective normalization followed by weighted

are normalized per component via DiagonalPopArt to handle the varying

combination matches the multi-critic implementation used in training.

26

7.2 (iii) Auxiliary Regularisations

Bilateral symmetry loss. An optional ?2 penalty between mirrored sagittal-plane joint pairs
P = {(i, j)} (11 pairs: shoulders, elbows, hips, knees, ankles):

Lsym =

?sym
|P| ?
(i, j)?P

?? j µi ? µ j?2,

where µi, µ j are policy mean outputs and ? j ? {?1, +1} accounts for lateral-axis sign flips [33].
The full PPO objective is:

L = L clip + cv L value + Lsym.

Phase-conditioned observations. When enabled, the normalised cycle position

?t =

t mod Tcycle
Tcycle

? [0, 1)

is encoded as (sin 2??t, cos 2??t) and appended to the policy observation, providing an explicit
temporal anchor for periodic motions.

Termination penalty. Episodes terminate when root body height drops below hmin for Ngrace
consecutive frames (fall detection). Terminated transitions receive a fixed negative reward rterm
to discourage unstable policies. Looped reference motions wrap cyclically, and the maximum

number of cycles per episode is configurable.

27

VIII

Adversarial Control Experiments

ViMo-Flow motion-generation experiments along with spml (34-joints) to deepmimic (24-joints)

retargeting convertion scope is convered here: Appendix A.

8.1

Implementation details

Controller training ran for 5,000 epochs in MuJoCo at 30 FPS. Two rollout settings were
evaluated: horizon Hsteps = 8 with 32 parallel environments, and horizon Hsteps = 16 with 16
parallel environments. PPO used 5 update epochs per rollout, clip ratio ? = 0.2, discount
? = 0.95, and GAE parameter ? = 0.95 [26]. Learning rates were fixed at 5 × 10?6 for the actor,
10?4 for the critic, and 10?5 for the discriminator. Gradient penalty used ?gp = 10 [7], and
symmetry experiments used ?sym = 0.005 [33]. The adversarial reward followed the ICCGAN-
style discriminator setup used in the control framework [30, 31].

Reference motions and difficulty tiers. The evaluation focused on three single-clip locomo-

tion motions with increasing control difficulty:

• limp_walk: slow, ground-contact dominated, no aerial phase;

• joyful_walk: moderate pace with mild vertical impulse;

• jaunty_walk: high-energy, fast, and visibly asymmetric.

The corresponding step counts per motion cycle were 57 for limp_walk, 35 for joyful_walk, and
41 for jaunty_walk, as parsed from the environment configuration. Maximum episode length
was set to two cycles for limp_walk and joyful_walk, and one cycle for jaunty_walk. In addition
to these controlled case studies, feasibility of the full pipeline was checked on retargeted motion

clips generated by the motion synthesis model and stored under the ‘vimo‘ motion set.

8.2 Evaluation metrics

For a motion with cycle length Scycle, the normalized survival metric is

lifetime_cycles =

lifetime
Scycle

.

Discriminator quality was tracked through mean real score, mean fake score, and their gap,

disc_gap = scorereal ? scorefake.

28

Additional diagnostics included mean reward, policy loss, value loss, and normalized symmetry

loss when symmetry regularization was active. Metrics were summarized over the final training

window (epochs 4500–5000), matching the procedure used in the case-study summaries.

Baseline performance. The baseline runs without phase input and without symmetry loss
established the reference operating point. At H = 8, final survival reached 0.68 ± 0.08 cycles
for limp_walk, 0.43 ± 0.05 cycles for joyful_walk, and 0.29 ± 0.04 cycles for jaunty_walk. At
H = 16, the corresponding baselines were 0.56 ± 0.09, 0.46 ± 0.07, and 0.23 ± 0.05 cycles.
the easier,
This ordering is consistent with the qualitative difficulty of the three motions:

contact-dominated gait survives longer, while faster and more asymmetric clips remain harder

to stabilize.

Horizon ablation (H = 8 versus H = 16).
poral support available to the critic and the discriminator, but it also reduces the number of

Increasing the rollout horizon improves the tem-

environments collected per update in the present setup. The effect was motion dependent. For
limp_walk, the phase-input variant improved from 0.69 ± 0.09 cycles at H = 8 to 0.73 ± 0.11
cycles at H = 16, indicating that longer temporal context is beneficial when the motion is
periodic and stable. For joyful_walk, the symmetry variant improved from 0.71 ± 0.08 cycles
at H = 8 to 0.77 ± 0.13 cycles at H = 16. For jaunty_walk, however, the best result remained
phase-based and only changed from 0.38 ± 0.05 to 0.33 ± 0.06, showing that horizon extension
alone does not resolve strongly asymmetric or impulsive control demands.

8.3 Case study I: phase-conditioned observations

Phase input augments the observation vector with sinusoidal encodings of the normalized cycle

position:

sphase
t

=

?

?
?

st
sin(2??t)
cos(2??t)

?

?
? ,

?t =

t mod Scycle
Scycle

,

where ?t ? [0, 1] denotes the phase within the motion cycle. This conditioning provides explicit
temporal alignment cues that reduce ambiguity in reference-following for periodic motions.

Limp walk results. At H = 8, phase conditioning improved:

• Mean reward: ?0.0901 ? ?0.0644 (28.5% improvement)

• Discriminator gap: 0.2718 ? 0.2658

• Lifetime cycles: 0.68 ? 0.69 (marginal gain)

At H = 16, the phase variant achieved the best configuration with 0.73 ± 0.11 cycles and reward

29

?0.0505, confirming that extended temporal context combined with phase information benefits
stable locomotion.

Jaunty walk results. For the asymmetric high-energy motion, phase conditioning produced

the strongest gains among all implemented variants:

• At H = 8: lifetime 0.29 ± 0.04 ? 0.38 ± 0.05 cycles (31% improvement)

• Reward: ?0.0925 ? ?0.0349 (62.3% improvement)

• Fake discriminator score: ?0.0646 ? ?0.0126 (closer to real)

• At H = 16: maintained advantage at 0.33 ± 0.06 vs. 0.23 ± 0.05 baseline

The mechanism operates as a temporal lookup key: the policy queries which portion of the
motion cycle it occupies and conditions its action distribution accordingly. This reduces variance

in the advantage estimates for GAE and stabilizes early training. However, the conditioning

assumes a known fixed cycle structure; it therefore generalizes poorly to aperiodic motions or
multi-behavior transitions where ?t becomes ill-defined.

8.4 Case study II: bilateral symmetry regularisation

The bilateral symmetry loss penalizes deviation from mirror symmetry across sagittal-plane
joint pairs. For each left-right pair ( jleft, jright) with corresponding actions (a jleft, a jright), the loss
computes:

Lsym =

1
Npairs

?
j?pairs

(cid:13)(a jleft ? ¯a j) ? (a jright ? ¯a j)(cid:13)
(cid:13)
(cid:13)2,

¯a j = 1

2 (a jleft + a jright).

The regularizer is applied with weight ?sym = 0.005 during PPO updates, biasing the policy
toward symmetric action distributions.

Motion-dependent effects. The regularizer’s effectiveness depends on the inherent symme-

try of the target motion:

1. Limp walk (near-symmetric): At H = 8, symmetry produced the best lifetime 0.71 ± 0.08
cycles; at H = 16, remained competitive at 0.64 ± 0.12 cycles. The prior aligns well with
the motion’s near-symmetric ground contact pattern.

2. Jaunty walk (asymmetric): The regularizer became strongly counterproductive:

• Lifetime dropped: 0.29 ± 0.04 ? 0.13 ± 0.01 at H = 8

• Normalized symmetry loss remained high: 0.4285 ± 0.0158
• The policy could not simultaneously minimize Lsym and match the asymmetric

reference

3. Joyful walk (moderately asymmetric): Symmetry improved survival metrics but degraded

imitation quality:

30

Figure 9: Limp walk training comparison across baseline (baseline), phase-input (phase),
symmetry-regularized (sym), and combined (phase+sym) variants at H = 8 and
H = 16. Metrics include lifetime cycles, discriminator scores, gap, value loss, policy
loss, reward mean, and normalized symmetry loss.

31

Figure 10: Jaunty walk training comparison across experimental variants. The phase-conditioned
variant (green) shows superior lifetime and reward characteristics for this asymmetric
motion, while symmetry regularization (orange) degrades performance substantially.

32

• Lifetime improved: 0.43 ± 0.05 ? 0.71 ± 0.08 at H = 8

• Fake discriminator score degraded: baseline ?0.03 ? ?0.1043 at H = 8

• Same trade-off observed at H = 16: lifetime 0.77 ± 0.13 but fake score ?0.0957

Interpretation. The symmetry prior functions as an exploration bias that constrains the ac-
tion space to a lower-dimensional symmetric manifold. For near-symmetric gaits, this constraint

excludes unstable asymmetric modes and accelerates convergence. For asymmetric motions,

the constraint conflicts with the target distribution, producing either mode collapse (jaunty) or

imitation-quality degradation (joyful). The appropriate application of this regularizer therefore
requires a priori knowledge of motion symmetry properties.

Pipeline use with generated motions. Beyond the controlled locomotion study, the pipeline

was exercised on retargeted motion clips derived from the motion generator and stored under the

‘vimo‘ motion assets and checkpoints. These runs were used primarily as feasibility tests: they

verified that the generated kinematic sequences could be converted into valid DeepMimic-style

references and loaded by the MuJoCo controller without format mismatch. This experiment was

important for the end-to-end objective of the project, because it established that the generated

motions are not only visually plausible in kinematic space but also structurally compatible with

the downstream control stack.

8.5

Scope of the present study

Only motion imitation and the two implemented ablations—phase-conditioned observations

and bilateral symmetry regularisation—were evaluated systematically in the current work.

Robustness experiments under perturbations, interactive policy switching, and fully goal-directed

control were not implemented in the present pipeline and are therefore excluded from quantitative

comparison here. Those components remain natural extensions once the imitation backbone is

stabilized on a broader set of motions.

Default hyperparameters used throughout (overridable per config for task-specific parameters

such as horizon, num_envs, and grace_steps):

The adversarial controller follows the ICCGAN framework [30] with a GRU-based policy

network, K-head critic, and ensemble of body-part-specific discriminators. The policy maps

state to action distribution via GRU + fully-connected layers; the critic estimates values per dis-

criminator head; the discriminator ensemble distinguishes reference from generated trajectories
on joint subsets. Average operator (?) aggregates features while concatenation (?) fuses state
and goal inputs. This architecture eliminates manual reward engineering by letting the policy

learn to fool the discriminators directly.

33

Figure 11: Joyful walk training comparison. The symmetry variant (orange) achieves high
lifetime cycles but at the cost of discriminator quality (lower fake scores), indicating
a survival-imitation trade-off.

34

Table 2: Hyperparameters

Parameter

policy network learning rate
value network learning rate
discriminator learning rate
reward discount factor (?)
GAE discount factor (? )
surrogate clip range (?)
gradient penalty coefficient (? GP)
PPO replay buffer size
PPO batch size
PPO optimization epochs
discriminator replay buffer size
discriminator batch size

Value
5 × 10?6
1 × 10?4
1 × 10?5
0.95
0.95
0.2
10
4096
256
5
8192
512

Figure 12: Network architectures. ? denotes the concatenation operator and ? denotes the

average operator.

35

IX

Conclusion and Future Work

The two-stage pipeline presented in this report demonstrates that physically plausible character

animation from casual video is tractable within a unified adversarial-diffusion framework. ViMo-

Flow generates kinematically coherent 3D motion from unconstrained footage via DDPM with

Min-SNR reweighting and Probabilistic Timestep Sampling, without requiring explicit camera

calibration. The physics-based imitation stage drives a MuJoCo humanoid to reproduce these

kinematics under rigid-body dynamics through body-part-specific discriminators and PPO,

eliminating hand-crafted reward engineering. Composite multi-objective reward aggregation

and goal-conditioned extensions demonstrate that the adversarial reward paradigm scales to

multi-skill learning beyond single-clip imitation.

Experiments across three locomotion difficulty tiers confirm that the baseline adversarial

controller scales with motion complexity and that both auxiliary mechanisms studied—phase-

conditioned observations and bilateral symmetry regularisation—offer narrow benefits on spe-

cific motion categories while introducing generalisation limitations that preclude their use as

default components.

Limitations. The current simulation character is a 28-DOF “humanoid-lite” model: it collapses

the SMPL skeleton’s 24 joints and 72 axis-angle DOF into 15 rigid bodies to prioritise simulation

stability. This simplification, motivated by the imitation-learning literature [21, 30], yields

controllable locomotion but fails on motions that demand simultaneous, coupled articulation of

all major joints—high-energy dance steps, aerial phases, and skidding contact patterns all require

joint torques and momentum transfers that the reduced model cannot faithfully reproduce.

A secondary limitation is dataset coverage. AIST++ provides choreographed single-person

dance; retargeted clips retain the gross body trajectory but lose subtle upper-body articulation

through chain-composition in the converter. ViMo-Flow outputs, while more diverse, carry

kinematic errors from the diffusion inference that the discriminator cannot always absorb.

Planned Extensions.

1. Extended humanoid (34 DOF). The simulation character will be upgraded from the

current DeepMimic 28-DOF body to a 34-DOF rig that mirrors the SMPL joint count

more closely—restoring individual spine segments, collarbones, and hand articulations as

actuated joints. This is expected to unlock complex dance motions where upper-body and

lower-body coordination is non-trivial, at the cost of a harder learning problem requiring

longer training and potentially curriculum initialisation.

2. Complex motion support. With the richer character model, focus will shift to motions

currently out of reach: jumps and landings (aerial impulse + contact recovery), lateral

36

shuffles and pivots, and asymmetric arm-leg coordination seen in jaunty walk and dance

genres. These require refining the retargeting pipeline to preserve aerial-phase root

trajectories and extending the discriminator window to capture contact transitions over

longer horizons.

3. Policy adaptation. Once stable physics-based control is established on the extended

character, the third stage of the pipeline—policy adaptation via AdaptNet [32]—will be

integrated. The two-tier adaptation hierarchy (latent-embedding perturbation for modest

shifts; deep-layer modification for substantial environment or morphology changes) ad-

dresses deployment robustness when the simulation parameters or task objectives differ

from training conditions. This stage was deferred to ensure the base imitation policy is

stable enough to serve as an initialisation point for adaptation.

X

Results

ViMo-Flow qualitative samples for different vimo configurations and thier outputs are covered

here: Appendix B as part of previous study.

10.1

ICCGAN Humanoid Motions

The adversarial imitation framework employs an ensemble of body-part-specific discriminators

trained in tandem with a PPO policy. The discriminators distinguish reference mocap trajectories

from policy-generated ones on subsets of joints (e.g., lower-body, upper-body, full-body),

providing dense reward signals without manual tuning. The policy maximizes the probability of

fooling these classifiers while satisfying dynamics constraints in the MuJoCo simulator. Default
parameters (horizon H = 16, N = 32 environments, simulation rate 120 Hz, learning rates
5 × 10?6 actor / 10?4 critic / 10?5 discriminator) are used across runs, with configs overriding
env-specific values such as grace_steps for ground-contact motions.

Performance is quantified via lifetime cycles (fraction of maximum episode length survived),
mean reward, discriminator gap (Dreal ? Dfake), and per-discriminator scores. Table 3 presents
final-period statistics (epochs 19 000–20 000, averaged over logged intervals) extracted from

training traces.

37

Figure 13: Jaunty walk

Figure 14: Joyful walk

Figure 15: Limp walk

Table 3: ICCGAN motion imitation performance at convergence (Period 20). Highlights used to

substantiate inferences on replication fidelity, stability, and termination behavior.

Motion

Lifetime Cycles

Reward Mean

Disc. Gap Score Real Score Fake Steps

Inference Highlight

squat
punch
leg_lunge
kick
long_jump
roll

0.4948 ± 0.0510
0.5051 ± 0.0650
0.4923 ± 0.0515
0.3942 ± 0.0672
0.3639 ± 0.0620
0.0575 ± 0.0095

0.1938 ± 0.0289
0.3694 ± 0.0258
0.3281 ± 0.0367
0.1481 ± 0.0235
0.0754 ± 0.0226
-0.1367 ± 0.0125

0.5399
0.4123
0.3919
0.3641
0.3483
0.1978

0.7321
0.7823
0.7185
0.5248
0.4451
0.1700

0.1922
0.3700
0.3266
0.1607
0.0968
-0.0278

84
39
109
56
53
99

perfect replication (fixed leg pose)
highest survival/reward; minor leg slip
high cycles but value-loss variance indicates slipperiness
moderate survival; hip-range instability causes falls
improved landing at moderate run speed
critically low cycles despite grace_steps; termination hurdle

Figure 17: Squat – stable pose with legs fixed, zero slip observed

Squat. Perfect kinematic and dynamic replication is attained. The motion constrains

leg joints to near-constant positions, eliminating balance challenges. This is numerically

corroborated by the highest stable lifetime cycles (0.49 of maximum), consistently positive

reward (0.1938), and strong real discriminator score (0.732). Simulation frames (Figure 17)

confirm exact pose matching without drift; the humanoid maintains the crouched configuration

throughout the episode without CoM excursions.

Figure 18: Punch – upper-body dominant action with minor terminal leg slippage

38

Figure 16: Comparative training curves across ICCGAN motions

39

Punch and leg lunge. Near-perfect imitation with only minor leg slipperiness in terminal

phases. Both achieve top-tier survival (0.51 and 0.49 cycles) and reward (0.37 and 0.33), with

score_fake approaching real (0.37 and 0.33). The slight slippage manifests as elevated value-loss

variance in lunge (0.31) but does not compromise overall stability, validating the discriminator

ensemble’s efficacy for upper-body dominant actions. Figure 18 illustrates the forward thrust

with stable base; lunge exhibits analogous behavior with minor foot repositioning at motion end.

Figure 19: Leg lunge – controlled forward extension with minor slip at terminal phase

Leg lunge. The policy achieves high-cycle performance (0.49) with strong reward (0.328),

though value-loss variance (0.31) indicates intermittent foot slip during the extension phase.

Figure 19 visualizes the controlled forward step; the slight repositioning at motion conclusion

accounts for the variance without destabilizing the overall trajectory.

Figure 20: Kick – hip-driven swing with support leg instability leading to falls

Kick. The policy initiates the hip-driven swing but frequently loses balance on the support

leg, leading to falls before completion. Lifetime cycles plateau at 0.39 with reward 0.148;

episodes terminate prematurely when center-of-mass projection exits the support polygon.

Occasional successful kicks occur but cannot maintain one-legged stance, as evidenced by the

moderate score_real (0.52) compared to squat/punch. Figure 20 captures the initiation phase;

subsequent frames show the characteristic collapse when weight shifts to the planted foot.

Figure 21: Long jump – takeoff and landing phases with speed-dependent stability

Long jump. Landing fidelity improves markedly in episodes where takeoff run velocity

remains moderate. Excessive forward lean during run-up (to counteract anticipated fall) produces

compensatory acceleration, degrading landing stability. The 0.36 cycles and 0.075 reward reflect

this sensitivity; renders in Figure 21 demonstrate successful cases under controlled speed.

40

Episodes 4–6 in the checkpoint RGB arrays show clean landings when run speed is normal-

paced; higher velocities introduce forward momentum that disrupts touchdown.

Figure 22: Roll – diving initiation captured; episode terminates upon ground contact (pelvis <

0.15m) preventing full recovery

Roll. Complete failure to complete the sequence despite targeted mitigation. The pelvis-

height termination (<0.15 m) triggers episode end precisely when the motion requires floor
contact. The grace_steps counter (extended in config) delays reset during prone states,
allowing partial diving imitation, yet the policy cannot recover upright posture from fallen initial

states. This is quantitatively confirmed by lifetime cycles collapsing to 0.0575, persistently

negative reward (-0.137), and lowest scores. Figure 22 captures the initial dive; the subsequent

ground contact triggers termination before the rollover and upright recovery can execute. The

implementation reveals a fundamental simulator limitation for motions violating the termination

heuristic, addressed partially but insufficiently for full rollout recovery.

These results demonstrate that the body-part discriminator approach scales effectively

to simple static and cyclic motions (squat, punch) but exposes limits on high-momentum

asymmetric or ground-interaction behaviors (kick, roll). Overlapping frame composites and per-

episode RGB arrays provide visual confirmation, with training curves showing rapid convergence

for high-fidelity cases within 5 000–10 000 epochs.

The locomotion baselines (jaunty, joyful, limp walks) were analyzed in the experimental

section; the above extends the evaluation to a broader mocap suite, confirming the pipeline’s

adaptability for mocap-driven physics-based animation while highlighting directions for ex-

tended DOF models and refined termination logic.

10.2 Vimo Full-Body Dance Motions (AIST Retargeted)

ViMo-generated full-body dance sequences from the AIST Dance Video Database were re-
targeted and grouped by dance_genre/choreography_id. Per the AIST naming con-
vention (gXX_sBM_cAll_dXX_mXX_chYY), the choreography ID defines the core sequence
while musical piece IDs vary tempo. Motions sharing the same choreography ID are structurally

similar with only execution speed differences. Grouping enables a single policy to learn from

a family of related references, improving generalization across tempo variations. Movement

descriptions derive from frame-wise comparison images in the preview assets.

41

Figure 23: Vimo training curves across dance genres.

42

Table 4: Vimo dance motion performance at convergence (Period 30). Numerical highlights
from training traces directly substantiate inferences on stability, converter artifacts, and
partial success.

Motion

Lifetime Cycles

Reward

Disc Gap Score Real Score Fake Steps

Inference Highlight

gBR/ch01 (indian step)
gHO/ch01 (loose legs)
gJS/ch02 (pos. des pieds)
gLH/ch01 (slide)
gLO/ch02 (twirl)
gMH/ch02 (rock board)
gPO/ch01 (fresno)

0.20 ±0.04
0.36 ±0.05
0.02 ±0.00
0.35 ±0.06
0.48 ±0.04
0.24 ±0.04
0.47 ±0.05

0.079 ±0.016
0.176 ±0.014
-0.182 ±0.007
0.136 ±0.017
0.096 ±0.010
0.093 ±0.018
0.205 ±0.057

0.484
0.526
0.304
0.531
0.463
0.524
0.453

0.569
0.702
0.179
0.667
0.557
0.621
0.659

0.085
0.175
-0.125
0.136
0.094
0.097
0.206

304
240
281
322
276
293
306

side-step falls; limited recovery
sliding legs cause balance loss
foot clipping from SMPL?28DOF converter causes collapse
partial success; tempo-induced hand instability
high survival but nullifies rhythmic knee bends
complete failure on leg-lift instability
legs prioritize stability; arms mimic reference

Figure 24: gBR/ch01 side-step falls

Figure 25: gHO/ch01 sliding legs

gBR/ch01. Side-step movements cause falls. Few balancing episodes fail to follow the

subsequent dance step (lifetime 0.20 cycles, reward 0.079; Figure 24).

gHO/ch01. Sliding leg motions lead to balance loss. Controller struggles with solid foot

plants (lifetime 0.36, reward 0.176; Figure 25).

Figure 26: gJS/ch02 foot clipping collapse

Figure 27: gLH/ch01 partial slide success

gJS/ch02. Easiest motion yet yields near-zero lifetime (0.02 cycles) and negative reward

(-0.182). Reference feet clip into floor due to SMPL-to-28DOF converter (ankle+toe averaged

into single joint). This induces physics collapse (Figure 26).

gLH/ch01. Simpler sliding dance. Controller produces reduced sliding with higher-tempo

hands causing instability. Partial success (0.35 cycles, 0.136 reward, real score 0.667; Figure 27).

43

Figure 28: gLO/ch02 stabilization over knee bends

Figure 29: gMH/ch02 leg-lift failure

gLO/ch02. Features abrupt freezes; selected clip is slow. Policy fails rhythmic knee bending,

prioritizing stabilization (lifetime 0.48 but reward 0.096; Figure 28).

gMH/ch02. Leg lifts with bent support leg cause complete failure (lifetime 0.24, reward

0.093; Figure 29).

Figure 30: gPO/ch01 legs focus on stability

gPO/ch01. Smooth genre; easy clip yields highest reward (0.205). Legs struggle for stability

while hands better follow reference (lifetime 0.47 cycles; Figure 30).

These Vimo evaluations, combined with ICCGAN results, confirm the adversarial frame-

work’s strengths on simpler gestures while exposing limits of the reduced-DOF model and

retargeting on complex rhythmic dance. Numerical metrics from the final training period and

side-by-side simulation frames guide refinement of the physics-based imitation pipeline toward

real-world character animation automation.

10.3 Task-Based Locomotion Simulations (LaFAN1)

This category extends the imitation backbone with goal-directed control. Reference motions are
drawn from the LaFAN1 dataset, grouping clips of walk, run, and crouched walk with diverse
starting positions and orientations. The grouped reference set enables the policy to imitate

locomotion regardless of initial pose, while a task-specific reward steers the character toward an

externally specified goal.

44

Task environment setup. Two task formulations are implemented; a third (Aiming) is imple-

mented but not evaluated here.

Target Heading. The character must move along a randomly sampled unit direction gt ? R2
resampled every 30 frames. The goal-directed reward measures alignment between the horizontal

root displacement and the target heading:

rG
t = ??xroot

t+1/??xt?, gt?.

Target Location. The character must reach a goal position pgoal at a preferred speed. The

reward is

rG
t =

?
?

exp(cid:0)?3 ??xroot

t+1/?t ? v?

t ?2/?v?

t ?2(cid:1)

1
?

if ?xt+1 ? pgoal? > ?

otherwise,

with goal radius ? = 0.5, sampling preferred speed in [1, 1.5] m/s for walk/crouch and [1, 3] m/s
for run, with random direction in [0, 2?) and timer in [3, 5] s (or [2, 3] s for run). The goal state
gt ? R4 encodes direction unit vector, distance to goal, and preferred speed.

Aiming (implemented, untested). A target aiming direction is encoded via the right-forearm
? gt?2) when activated; otherwise an

, with reward rG

t = exp(?2?dforearm

t

orientation dforearm
arm-lift bias keeps the gun raised.

t

Table 5: Task-based locomotion performance at convergence.

Motion

Lifetime Cycles

Reward

Disc Gap Score Real Score Fake Task Reward

Steps

Inference Highlight

locomotion_walk
locomotion_run
locomotion_crouch

0.31 ±0.05
0.27 ±0.07
0.39 ±0.05

0.163 ±0.065
-0.079 ±0.084
0.064 ±0.058

0.413
0.535
0.508

0.694
0.585
0.625

0.281
0.050
0.116

0.170 ±0.040
0.078 ±0.020
0.122 ±0.031

1253
2122
666

decent mimicry; struggles for wide deviation turns
kangaroo-like hopping; partial mimicry despite longest training
longest survival; train-wheel leg drag hack stabilizes turns

Figure 32: locomotion_walk – target-following with moderate turn capability

Walk. Mimicry is acceptable across the diverse reference set, with the policy reliably

orienting toward goals at small angular deviations. Lifetime 0.31 cycles, reward 0.163, and

highest score_fake (0.281) confirm strong imitation fidelity. Task reward (0.170) is highest

among the three, indicating effective heading/location tracking. Wider angular deviations expose

slower reorientation and weaker stability than crouch.

Figure 33: locomotion_crouch – wide-deviation turns via stabilizing leg-drag gait

45

Figure 31: Task-based locomotion training curves across walk, run, and crouch policies.

46

Crouch. Surprisingly strong task performance with longest survival (0.39 cycles) despite

shortest training (666 steps). The controller turns toward wider goal deviations effectively.

Inspection reveals an emergent stabilization hack: one leg moves slowly forward like a train-

wheel while the other drags as support, preventing falls. This is reflected in elevated value-loss

variance (0.14) and modest reward (0.064), signaling stable but non-textbook gait.

Figure 34: locomotion_run – high-speed kangaroo hopping with stability failure

Run. High-speed dynamics overwhelm the policy: it adopts a kangaroo-like hopping

gait, repeatedly skipping forward in attempts to stabilize before falling. Despite the longest

training (2122 steps), lifetime collapses to 0.27 cycles with negative reward (-0.079) and lowest

score_fake (0.050)—less than partial imitation success. Task reward (0.078) is correspondingly

low, confirming that mimicability remains the bottleneck before task completion can be evaluated

reliably.

The task-based experiments demonstrate that goal-directed extensions integrate cleanly

with the adversarial imitation backbone for low- and moderate-velocity gaits. High-velocity

locomotion exposes the same DOF and momentum-transfer limitations identified in earlier

categories, motivating the planned humanoid extension and richer discriminator window.

47

REFERENCES

[1] K. Bergamin, S. Clavet, D. Holden, and J. R. Forbes (2019). Drecon: data-driven
responsive control of physics-based characters. ACM Trans. Graph., 38(6). ISSN 0730-
0301. URL https://doi.org/10.1145/3355089.3356536.

[2] Y. Cai, L. Ge, J. Liu, J. Cai, T.-J. Cham, J. Yuan, and N. M. Thalmann, Exploiting
spatial-temporal relationships for 3d pose estimation via graph convolutional networks. In
2019 IEEE/CVF International Conference on Computer Vision (ICCV). 2019.

[3] Z. Cai, W. Yin, A. Zeng, C. Wei, Q. Sun, Y. Wang, H. E. Pang, H. Mei, M. Zhang,
L. Zhang, C. C. Loy, L. Yang, and Z. Liu (2024). Smpler-x: Scaling up expressive
human pose and shape estimation. URL https://arxiv.org/abs/2309.17448.

[4] N. Chentanez, M. Müller, M. Macklin, V. Makoviychuk, and S. Jeschke, Physics-based
motion capture imitation with deep reinforcement learning. In Proceedings of the 11th
ACM SIGGRAPH Conference on Motion, Interaction and Games, MIG ’18. Association
for Computing Machinery, New York, NY, USA, 2018. ISBN 9781450360159. URL
https://doi.org/10.1145/3274247.3274506.

[5] B. Degardin, J. ao Neves, V. Lopes, J. ao Brito, E. Yaghoubi, and H. Proença (2021).
Generative adversarial graph convolutional networks for human action synthesis. URL
https://arxiv.org/abs/2110.11191.

[6] H.-S. Fang, J. Li, H. Tang, C. Xu, H. Zhu, Y. Xiu, Y.-L. Li, and C. Lu (2022). Alphapose:
Whole-body regional multi-person pose estimation and tracking in real-time. URL https:
//arxiv.org/abs/2211.03375.

[7] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville (2017). Improved
training of wasserstein gans. URL https://arxiv.org/abs/1704.00028.

[8] C. Guo, X. Zuo, S. Wang, S. Zou, Q. Sun, A. Deng, M. Gong, and L. Cheng,
In Proceedings of
Action2motion: Conditioned generation of 3d human motions.
the 28th ACM International Conference on Multimedia, MM ’20. ACM, 2020. URL
http://dx.doi.org/10.1145/3394171.3413635.

[9] T. Hang, S. Gu, C. Li, J. Bao, D. Chen, H. Hu, X. Geng, and B. Guo (2024). Efficient
diffusion training via min-snr weighting strategy. URL https://arxiv.org/abs/
2303.09556.

[10] J. Ho and S. Ermon (2016). Generative adversarial imitation learning. URL https:

//arxiv.org/abs/1606.03476.

[11] J. Ho, A. Jain, and P. Abbeel (2020). Denoising diffusion probabilistic models. URL

https://arxiv.org/abs/2006.11239.

[12] D. Holden, O. Kanoun, M. Perepichka, and T. Popa (2020). Learned motion matching.
ACM Trans. Graph., 39(4). ISSN 0730-0301. URL https://doi.org/10.1145/
3386569.3392440.

48

[13] D. Holden, T. Komura, and J. Saito (2017). Phase-functioned neural networks for
character control. ACM Trans. Graph., 36(4). ISSN 0730-0301. URL https://doi.
org/10.1145/3072959.3073663.

[14] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter
(2019). Learning agile and dynamic motor skills for legged robots. Science Robotics,
4(26). ISSN 2470-9476. URL http://dx.doi.org/10.1126/scirobotics.
aau5872.

[15] C. Ionescu, D. Papava, V. Olaru, and C. Sminchisescu (2014). Human3.6m: Large
scale datasets and predictive methods for 3d human sensing in natural environments. IEEE
Transactions on Pattern Analysis and Machine Intelligence, 36(7), 1325–1339.

[16] R. Li, S. Yang, D. A. Ross, and A. Kanazawa (2021). Ai choreographer: Music con-
ditioned 3d dance generation with aist++. URL https://arxiv.org/abs/2101.
08779.

[17] Y. Liu, Y. Ren, X. Cun, A. Artola, Y. Liu, T. Zeng, R. H. Chan, and J. michel
Morel (2024). Redefining temporal modeling in video diffusion: The vectorized timestep
approach. URL https://arxiv.org/abs/2410.03160.

[18] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, and M. J. Black (2019). Amass:
Archive of motion capture as surface shapes. URL https://arxiv.org/abs/1904.
03278.

[19] J. Martinez, R. Hossain, J. Romero, and J. J. Little (2017). A simple yet effective
baseline for 3d human pose estimation. URL https://arxiv.org/abs/1705.
03098.

[20] J. Merel, L. Hasenclever, A. Galashov, A. Ahuja, V. Pham, G. Wayne, Y. W. Teh,
and N. Heess (2019). Neural probabilistic motor primitives for humanoid control. URL
https://arxiv.org/abs/1811.11711.

[21] X. B. Peng, P. Abbeel, S. Levine, and M. van de Panne (2018). Deepmimic: example-
guided deep reinforcement learning of physics-based character skills. ACM Transactions
on Graphics, 37(4), 1–14. ISSN 1557-7368. URL http://dx.doi.org/10.1145/
3197517.3201311.

[22] X. B. Peng, E. Coumans, T. Zhang, T.-W. Lee, J. Tan, and S. Levine (2020). Learning
agile robotic locomotion skills by imitating animals. URL https://arxiv.org/
abs/2004.00784.

[23] X. B. Peng, Z. Ma, P. Abbeel, S. Levine, and A. Kanazawa (2021). Amp: adversarial
motion priors for stylized physics-based character control. ACM Transactions on Graphics,
40(4), 1–20. ISSN 1557-7368. URL http://dx.doi.org/10.1145/3450626.
3459670.

[24] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville (2017). Film: Visual
reasoning with a general conditioning layer. URL https://arxiv.org/abs/1709.
07871.

[25] L. Qiu, C. Yu, Y. Li, Z. Wang, H. Huang, C. Ma, D. Zhang, P. Wan, and X. Han (2024).
Vimo: Generating motions from casual videos. URL https://arxiv.org/abs/
2408.06614.

49

[26] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017). Proximal

policy optimization algorithms. URL https://arxiv.org/abs/1707.06347.

[27] E. Todorov, T. Erez, and Y. Tassa (2012). Mujoco: A physics engine for model-based
control. 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 5026–
5033. URL https://api.semanticscholar.org/CorpusID:5230692.

[28] J. Tseng, R. Castellon, and C. K. Liu (2022). Edge: Editable dance generation from

music. URL https://arxiv.org/abs/2211.10658.

[29] J. Won, D. Gopinath, and J. Hodgins (2020). A scalable approach to control diverse
behaviors for physically simulated characters. ACM Trans. Graph., 39(4). ISSN 0730-0301.
URL https://doi.org/10.1145/3386569.3392381.

[30] P. Xu and I. Karamouzas (2021). A gan-like approach for physics-based imitation learning
and interactive character control. Proceedings of the ACM on Computer Graphics and
Interactive Techniques, 4(3), 1–22. ISSN 2577-6193. URL http://dx.doi.org/10.
1145/3480148.

[31] P. Xu, X. Shang, V. Zordan, and I. Karamouzas (2023). Composite motion learning
with task control. ACM Transactions on Graphics, 42(4), 1–16. ISSN 1557-7368. URL
http://dx.doi.org/10.1145/3592447.

[32] P. Xu, K. Xie, S. Andrews, P. G. Kry, M. Neff, M. Mcguire, I. Karamouzas, and
V. Zordan (2023). Adaptnet: Policy adaptation for physics-based character control. ACM
Transactions on Graphics, 42(6), 1–17. ISSN 1557-7368. URL http://dx.doi.org/
10.1145/3618375.

[33] W. Yu, G. Turk, and C. K. Liu (2018). Learning symmetric and low-energy locomotion.
ACM Transactions on Graphics, 37(4), 1–12. ISSN 1557-7368. URL http://dx.doi.
org/10.1145/3197517.3201397.

[34] Y. Yuan, U. Iqbal, P. Molchanov, K. Kitani, and J. Kautz (2022). Glamr: Global
occlusion-aware human mesh recovery with dynamic cameras. URL https://arxiv.
org/abs/2112.01524.

[35] M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu (2022). Motiondiffuse:
Text-driven human motion generation with diffusion model. URL https://arxiv.
org/abs/2208.15001.

[36] Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li (2020). On the continuity of rotation
representations in neural networks. URL https://arxiv.org/abs/1812.07035.

[37] W. Zhu, X. Ma, Z. Liu, L. Liu, W. Wu, and Y. Wang (2023). Motionbert: A unified
perspective on learning human motion representations. URL https://arxiv.org/
abs/2210.06551.

50

A Motion Generation Experiments

Implementation details. Training used the ADAN optimizer with an initial learning rate of
1 × 10?4. Sequences were resampled from 60 FPS to 30 FPS, yielding S = 150 frames per
five-second clip. Diffusion timesteps used T = 1000, with DDIM acceleration during sampling.
SMPL axis-angle parameters were converted to 6D rotations using orthogonalization [36]. Foot-

contact labels were derived from near-zero vertical foot and toe velocities. Training followed

the AIST++ protocol [16]: break and middle hip-hop genres were held out for testing, while the

remaining genres were used for training. Multi-view 2D keypoints served as conditioning input,

and batch size was fixed at 32. The objective of these experiments was not only to train the

generator in isolation, but also to verify that its outputs can be converted into reference motions

suitable for physically simulated control.

Training stability and lambda scheduling. Early epochs set ?joints = 0.01, ?vel = 1.0, ?foot =
0.1. The velocity loss dominated, causing collapse to static poses where Rt+1 ? Rt. Direct matrix
subtraction Rt+1 ? Rt is not physically meaningful on SO(3) and produced large gradients that
overwhelmed the diffusion objective. The model learned that predicting no motion minimized

loss.

To counteract collapse, lambda scheduling evolved: ?joints increased to 0.1 by epoch 25,
?vel reduced to 0.01 by epoch 40, and ?foot rose to 0.5. Crucially, setting any auxiliary loss to
zero caused immediate mean-pose collapse; all three losses required non-zero weights from
initialization. The final stable configuration used ?joints = 0.1, ?vel = 0.02, ?foot = 0.05.

SNR weighting strategies. Standard diffusion training with constant weighting wt = 1 yielded
poor sample efficiency because most late timesteps contributed negligible gradient after the SNR

cliff. Experiments evaluated four strategies:

• Min-cap: wt = min(SNRt, ?) with ? = 1 created a steep cliff; 70% of steps carried

near-zero weight.

• Decay min-cap: Linear or sqrt decay schedules smoothed early steps but left the cliff

intact.

• Averaging: wt = 1

2 [min(SNRt, 1) + (cid:112)1 ? t/T ] filled the dead zone, achieving 50%

sample efficiency. This variant prevented collapse and maximized data utilization.

• Normalization: Per-batch normalization of wt stabilized training but slowed convergence

when combined with high caps (? = 2).

The most stable setting used square-root decay averaging with ? = 1 and no per-batch normal-
ization, providing the best trade-off between gradient coverage and convergence speed [9].

51

Figure 35: SMPL Humanoid (34-joints) to Deepmimic Humanoid (28-joints) Retargeted Sample

52

Figure 36: Ablation of SNR weighting strategies for a T = 1000 linear diffusion process (? ?
1 ? ¯?t, benchmark
[1e ? 4, 0.02]). Plotted are the signal term
decay curves (linear, square root, and their average), and proposed SNR variants:
simple ratio ¯?t
, min-capped decays, ratio-split=0.1 blends, normalized, and scaled
1? ¯?t
weights. Area-under-curve (AUC) percentages (cap=1.0) quantify integrated weight
magnitude. Normalized SNR achieves the highest AUC (99.843%), while simple
SNR decays precipitously.

¯?t and noise term

?

?

53

Temporal modeling with PTSS. Vectorized timesteps enable frame-wise noise schedules

but risk combinatorial explosion. Probabilistic Timestep Sampling Strategy (PTSS) controls
complexity. Experiments varied p, the probability of independent per-frame sampling:

• p = 0.2 to 0.5 over epochs 1–50 showed steady loss decrease and growing prediction

variance.

• p > 0.6 introduced excessive variance, triggering SMPL forward kinematics violations in

root position scale.

• Returning to p = 0.2 after epoch 60 with learning rate increased to 2.5 × 10?4 stabilized

training.

PTSS improved temporal flexibility but required careful scheduling. In practice, low initial
values of p were essential for stability, while larger values were useful only after the model had
already learned coarse motion structure [17].

Model architecture ablations. Two FiLM implementations were compared. Version 1 shared

a single linear layer across all modulation points within a denoiser block. Version 2 used three

separate linear layers for self-attention, cross-attention, and MLP outputs. Both applied identical

parameters across frames.

• Small models (d = 128, nheads = 4) showed no performance gap.

• Large models (d = 256, nheads = 8) revealed Version 2 converged marginally slower due

to increased parameters.

• Version 1 was selected for production runs as parameter reduction mitigated overfitting on

the 26k-sample dataset.

• The difference was negligible; architectural novelty lies in conditioning strategy, not FiLM

multiplicity.

Convergence monitoring and deployment use. Tracking loss component magnitudes guided

training:

• Ljoints target: 0.01–0.02. Below 0.01 indicated overfitting; above 0.03 required higher

?joints.

• Lvel target: 0.05–0.10. Below 0.01 signaled static predictions; above 0.15 produced

erratic motion.

• Lfoot target: 10?4–10?3. Above 0.01 revealed excessive foot sliding; below 10?4 over-

constrained naturalness.

54

Prediction standard deviation steadily increased from near-zero to stable values around 0.1–0.2,

confirming the emergence of dynamic motion. Beyond quantitative convergence, the important

pipeline-level outcome was successful retargeting of generated clips into simulation-ready

references. This was verified on multiple converted motion families, including checkpoints

corresponding to ‘gBR‘, ‘gHO‘, ‘gJS‘, ‘gLH‘, ‘gLO‘, ‘gMH‘, and ‘gPO‘, which were used as

motion inputs for downstream control experiments.

B

ViMo-Flow: Qualitative Results

Two transformer-diffusion configurations are evaluated: small model (dmodel = 128, nheads = 4)
and big model (dmodel = 256, nheads = 8). Each run shows ground-truth reference frames,
deterministic denoising predictions ˆx0 at selected timesteps, and stochastic generation samples.

Figure 37: Ground-truth motion sequence (small model v1, no PTSS).

Figure 38: Deterministic predictions ˆx0 at timesteps t ? {100, 200, 300, 400, 500} (small model

v1, no PTSS).

Figure 39: Stochastic samples at timesteps t ? {100, 200, 300, 400, 500} (small model v1, no

PTSS).

55

Small model v1, no PTSS (ViMo baseline), 50 epochs

Figure 40: Ground-truth motion sequence (small model v1, PTSS).

Figure 41: Deterministic predictions ˆx0 at t ? {50, 100, 150, 200, 250, 300, 350, 400} (small

model v1, PTSS).

Figure 42: Stochastic samples at t ? {50, 100, 150, 200, 250, 300, 350, 400} (small model v1,

PTSS).

Small model v1 with PTSS, 65 epochs

56

Figure 43: Ground-truth motion sequence (small model v2, PTSS).

Figure 44: Deterministic predictions ˆx0 at t ? {50, 100, 150, 200, 250} (small model v2, PTSS).

Figure 45: Stochastic samples at t ? {50, 100, 150, 200, 250} (small model v2, PTSS).

Small model v2 with PTSS, 65 epochs

Figure 46: Example 1: ground-truth motion sequence (big model v1, PTSS).

57

Figure 47: Example

ˆx0
deterministic
{50, 100, 150, 200, 250, 300, 350, 400, 450} (big model v1, PTSS).

predictions

1:

at

t

?

Figure 48: Example 1: stochastic samples at t ? {50, 100, 150, 200, 250, 300, 350, 400, 450} (big

model v1, PTSS).

Figure 49: Example 2: ground-truth motion sequence (big model v1, PTSS).

58

Figure 50: Example 2: deterministic predictions ˆx0 at t ? {50, 100, 150, 200, 250, 300, 350} (big

model v1, PTSS).

Figure 51: Example 2: stochastic samples at t ? {50, 100, 150, 200, 250, 300, 350} (big model

v1, PTSS).

Big model v1 with PTSS, 100 epochs

59


