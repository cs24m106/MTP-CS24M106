<!-- Slide number: 1 -->
# Introduction
Untapped Solution:Everyday online videos contain billions of clips spanning martial arts, ballet, occupational tasks, and impromptu performances, offering an escape from dataset scarcity

Extraction Impossibility: Traditional pose estimation pipelines fail on such "casual" video due to dynamic camera work, shot transitions, occlusion, and unknown camera parameters, causing frame-wise errors to accumulate into jittery, implausible trajectories

Generative Reframing:ViMo-Flow addresses this by treating video-to-motion as a synthesis problem rather than reconstruction—using a diffusion model to generate plausible 3D motion sequences conditioned on 2D pose trajectories, bypassing explicit camera calibration entirely
Core Need: Realistic 3D human motions are the "soul" of virtual characters, essential for animation, gaming, and metaverse applications, yet traditional creation relies on expensive motion-capture systems or labor-intensive manual animation

Limitation of Modern Methods: Recent data-driven approaches using diffusion models, VAEs, and GANs improve synthesis but remain constrained by scarce, narrow-domain motion datasets (AMASS, Human3.6M, AIST++), limiting diversity and generalization to styles like classical dance or figure skating

Dependency Bottleneck: Existing methods require matched ground-truth pairs (text-to-motion, audio-to-motion), binding them to predefined categories and preventing zero-shot generation of unseen behaviors—effectively creating a closed vocabulary of actions
pg.no. 1

<!-- Slide number: 2 -->
Related Works
# Traditional Paradigms in Motion Generation & 3D Reconstruction
The Critical Gap: Breakdown in Real-World Conditions
Conditional Motion Generation
Data-Driven Paradigm: The field has shifted from manual animation/mocap to deep generative models (VAEs, GANs, Diffusion) for scalable, automated motion synthesis.
Conditioning Signals: Models are conditioned on various inputs:
Categories/Actions: Models like Action2Motion generate motions from discrete action labels (e.g., "walk," "kick").
Text & Music: Diffusion models (MotionDiffuse) and other architectures (EDGE) produce motions from text descriptions or music.
Inherent Limitation: These models are fundamentally limited by their training data. They learn the distribution of motions present in curated 3D datasets (e.g., AMASS, Human3.6M, AIST++) and struggle to generate styles or actions outside this "closed vocabulary.“

3D Pose Estimation from Video
Standard Two-Stage Pipeline:
2D Detection: Extract 2D keypoints from video (using models like OpenPose, AlphaPose).
3D Lifting: Regress 3D poses from 2D inputs using CNNs, GCNs (e.g., Cai et al.), or simple MLPs (Martinez et al.).
Assumption of Known Cameras: These methods (including SMPLer-X) perform well but assume static cameras or known/estimable camera parameters.
Brittle Under Real Conditions: The 2D-to-3D lifting objective becomes ill-posed when camera parameters are unknown or dynamic, leading to accumulated errors, jitter, and physically implausible motions.
The Challenge of “Casual Video”
Characteristics: User-generated content features: Dynamic camera work (zoom, pan, shake), Rapid shot transitions and montage editing, Frequent occlusions (objects, self-occlusion), Varied and cluttered environments.
Impact on Pose Estimation:
Frame-wise 3D predictions accumulate drift.
Results are jittery and lack temporal coherence.
Methods like GLAMR attempt global smoothing but are computationally expensive and fail under aggressive editing.
Problem: The single-view assumption and explicit regression approach are too brittle for the noise and variability of casual footage.

Consequences for State-of-the-Art
Closed-World Generation:
Action-conditioned models (e.g., Action2Motion) can only generate motions from the 10-30 action categories they were trained on.
They lack a mechanism to compose or generate unseen behaviors, limiting their practical application.
Dependence on Curated Data: High-fidelity generation (text-to-motion, music-to-dance) relies on scarce, expensive, and narrowly-scoped 3D motion-capture datasets.
The Diversity Bottleneck: This reliance on limited 3D corpora creates a persistent bottleneck, preventing models from capturing the full spectrum of human motion found in the real world.
pg.no. 2

<!-- Slide number: 3 -->

# Problem Statement
Input Specification
Source: A casual video of a single person.
Processed Input: A 2D pose sequence tensor p.
Dimensions:
Frames (S): 150 frames (5 seconds at 30 FPS).
Joints (J?d): 17 joints (COCO format).
Channels (C_in): 3 (x, y, confidence score).
Handling Noise: Unreliable joints (low confidence) are masked out.
Output Specification
Format: A 3D motion sequence M.
Components:
Joint Rotations (R): 24 joints represented in continuous 6D rotation format (144 dimensions).
Foot Contact Labels (f): 4 binary labels (left/right ankle and toe) to prevent foot sliding.
Root Position (t): 3D global translation of the root joint.
Goal: The output must preserve the style and rhythm of the input video and be suitable for physics-based animation.

![](GoogleShape124p23.jpg)
pg.no. 3

<!-- Slide number: 4 -->
# 6D Rotation Representation
A rotation matrix R?R^(3×3)  is a square matrix that describes how a 3D object or joint rotates in space. It is defined by two critical mathematical properties: Orthonormality and Determinant = 1.

Advantages: Rotation matrices are geometrically intuitive and computationally straightforward—applying a rotation is simply a matrix-vector multiplication. They explicitly encode the full spatial orientation.?
Disadvantages: Rotation matrices are over-parametrized—they use 9 numbers to represent only 3 deg of freedom (the dim of the 3D rotation group SO(3)). If the network directly predicts all 9 elements, there is no guarantee the output will satisfy the orthonormality and determinant constraints. Post-processing steps like QR decomposition or Gram-Schmidt orthonormalization are needed to project the output onto the valid rotation manifold, but these operations are not directly differentiable or optimized through the network, leading to suboptimal learning.

![](GoogleShape169p28.jpg)

![](GoogleShape170p28.jpg)

![](GoogleShape178p29.jpg)

![](GoogleShape179p29.jpg)

![](GoogleShape180p29.jpg)

![](GoogleShape181p29.jpg)

![](GoogleShape182p29.jpg)
pg.no. 4

<!-- Slide number: 5 -->
# Overview
pg.no. 5

<!-- Slide number: 6 -->

# AIST++ Dataset
A large-scale dataset of 3D dance motions. Contains 1,408 sequences. Each sequence name corresponds with a set of multi-view video names in the AIST Dance Video DB.
Human Motion Sequence:
Each SMPL-format human motion sequence is stored in a .pkl file with the following attributes:
‘smpl_poses’: Sequences of SMPL pose parameters. Array shape is (N, 24, 3).
‘smpl_trans’: Motion 3D trajectory. Array shape is (N, 3).

Keypoints2d Annotation: Multi-view frame-by-frame 2D keypoints detection results. Array shape is (9, N, 17, 3), where
    - The first dim represents Individual environment settings, each with 9 cameras.
    - J2D = 17 (no.of joints in 2d repr. w.r.t coco semantics).
    - The last dim, i.e. each joint contains (x, y, confidence).
    - NOTE: keypoints are given directly in image pixel coordinates, not normalized or made relative to a smaller scale. The AIST videos are 1080p resolution, with frames sized at 1920×1080 pixels — the keypoints are not scaled.

![](GoogleShape220p34.jpg)
COCO 17 keypoint order
pg.no. 6

<!-- Slide number: 7 -->
Pose Estimators

![](GoogleShape229p35.jpg)

![](GoogleShape231p35.jpg)
OpenPose
AlphaPose
Role in the Pipeline
Purpose: To extract the input 2D pose sequences from casual, monocular video.
Models Used: State-of-the-art detectors OpenPose and AlphaPose.
Handling Real-World Noise
Confidence Scores: Both models provide per-joint confidence scores.
Robustness Strategy: Low-confidence detections (often due to occlusion or motion blur) are masked out to prevent corrupting the conditioning signal.
Multi-View Handling: In scenarios with multiple views, the highest-confidence detection per joint is selected or aggregated.
Key Point: The pipeline is explicitly designed to be resilient to the noisy and incomplete outputs from these estimators, which is common in casual video.

![](GoogleShape230p35.jpg)
pg.no. 7

<!-- Slide number: 8 -->
# Preprocessing

![](GoogleShape204p32.jpg)
Temporal Alignment:
Resample sequences from 60 FPS to 30 FPS.
Partition into non-overlapping clips of 150 frames (5 seconds).

3D Motion Representation:
Convert SMPL axis-angle rotations to a continuous 6D rotation representation to avoid gimbal lock and ease optimization.
Construct the final ground-truth motion tensor to include 6D rotations, foot contacts, and root translation.

2D Pose Normalization:
A critical step: 2D poses are centered relative to the root joint.
This ensures the model conditions on relative joint positions instead of absolute pixel coordinates, enabling generalization across different video scales and camera views.

Foot Contact Labeling: Pre-computed from foot and toe vertical velocities (near zero velocity indicates contact).
SMPL (Tool used for Forward Kinematics in implementation)
pg.no. 8

<!-- Slide number: 9 -->
# ViMo Pipeline

![](Picture5.jpg)
pg.no. 9

<!-- Slide number: 10 -->
A transformer based diffusion model is leveraged to align the synthesis motion to the 2D pose sequences.
[18] ? ref here the DDPM paper published at 2020.
# Diffusion Formulation

![](GoogleShape252p38.jpg)
Why Direct Prediction?: Structured outputs (orthonormal rotation matrices, discrete contacts) require precise parameterization; predicting noise would demand additional projection steps that degrade gradient flow and complicate training stability.
Better suited for structured output space: 6D rotation representation, root translation, binary foot-contact labels
Avoids complex post-processing required for noise prediction in constrained parameter spaces

![](GoogleShape253p38.jpg)
Posterior Computation: Closed-form expression enables both deterministic (DDIM) and stochastic (DDPM) sampling.
Sampling Equation (Posterior):

![](GoogleShape254p38.jpg)
Ref DDPM (2020)’s Diffusion Formulation: Model predicts epsilon and sample is obtained from substituting epsilon onto the close form expression:

![](Picture12.jpg)

![](Picture9.jpg)

![A math equation with numbers and symbols AI-generated content may be incorrect.](Picture22.jpg)
pg.no. 10

<!-- Slide number: 11 -->
# Network Architecture
FiLM Conditioning
Three-Block Transformer Stack: Each denoiser block contains self-attention ? cross-attention ? MLP in sequence
Information Flow:
Self-attention extracts temporal context from noisy motion input across all 150 frames
Cross-attention conditions the model on extracted 2D pose tokens from video
FiLM blocks dynamically modulate intermediate activations using timestep and pose encoding
Design Variants:
Version 1: Single gamma/beta parameter pair shared across self-attention, cross-attention, and MLP modulation points—parameter-efficient, enforces consistent conditioning
Version 2: Separate linear layers for each modulation point—finer-grained control but 3× parameters
Selection: Version 1 chosen for production to mitigate overfitting on 26k-sample dataset
Modulation Mechanism: Feature-wise Linear Modulation applies affine transformationy = ?(c) ? x + ?(c)
Condition Vector Construction: c = concat[embed(t), encode(p)] where:
embed(t) provides timestep embedding capturing diffusion progress
encode(p) generated by lightweight temporal transformer processing full 2D pose trajectory
Encoding Network: Compact temporal transformer captures pose shape, rhythmic patterns, and stylistic qualities across 150-frame sequences into a dense representation
Dynamic Alignment: FiLM translates the combined condition into per-channel scale (?) and shift (?) parameters, synchronizing denoising behavior with video content and preserving temporal coherence

pg.no. 11

<!-- Slide number: 12 -->

# Sampling and Training Strategy
Classifier-Free Guidance
Probabilistic Timestep Sampling Strategy
Problem: Using independent timesteps per frame leads to a combinatorial explosion (1000^N for N frames).
Solution: PTSS introduces a probability  ‘p’  to decide:
With prob.  ‘p’ : Each frame gets an independently sampled timestep.
With prob.  ‘1-p’ : A single timestep is broadcast to all frames.
Benefit: This hybrid approach balances temporal expressivity and training efficiency, preventing overfitting to a fixed noise schedule.

![A diagram of a video transmission](Picture10.jpg)
pg.no. 12

<!-- Slide number: 13 -->
Loss Functions

![](GoogleShape260p39.jpg)

![](GoogleShape261p39.jpg)

![](GoogleShape262p39.jpg)
pg.no. 13

<!-- Slide number: 14 -->
# Applications
3D Motion Dataset Construction:
Key Accomplishment: Collected 52 Chinese classic dancing competition videos and generated 750 high-quality 3D motion clips spanning 63 minutes of motion data.?
Downstream Benefits: This dataset enables downstream tasks such as music-to-motion generation, motion recognition, motion prediction, and expands dance genre representation (Chinese classical dance was previously absent from major datasets).?
Scalability: Demonstrates potential to expand datasets by 5x to 50x by leveraging the abundance of freely available video content online.

Few-Shot Dancing Stylization:
Paradigm Shift: Overcomes the constraint of existing music-to-dance models (e.g., EDGE) that are limited to predefined categories (10 street dance categories in AIST dataset).
Technical Innovation: Uses zero-convolution blocks for parameter-efficient fine-tuning—keeps the pre-trained music-to-motion knowledge frozen while adaptively learning new style distributions, preventing catastrophic forgetting.
User Study Results: Zero-convolution adaptation showed 56–66% win rate versus direct fine-tuning (34–44%), demonstrating significantly better style transfer quality.

![](GoogleShape111p21.jpg)
pg.no. 14
pg.no. 14

<!-- Slide number: 15 -->
Video-guided Motion Completion
Enables three essential editing tasks—
motion in-betweening (filling intermediate frames between keyframes),
in-filling (completing missing motion segments), and
blending (seamlessly transitioning between different motion styles).
Efficiency: Provided only partial keyframes or reference clips; the model automatically generates plausible, smooth intermediate motions, significantly reducing manual work.
Technical Approach: Leverages diffusion inpainting techniques—applies a binary mask during denoising to preserve constraint regions while generating natural transitions in unconstrained frames.

![](GoogleShape275p41.jpg)
An Illustration of Zero-Conv Blocks

![](GoogleShape274p41.jpg)

<!-- Slide number: 16 -->

# Experiments
SNR Weighting Strategy Ablation
• Baseline Problem: Constant weighting w?=1 achieved only 32–36% sample efficiency due to SNR "cliff" (most timesteps near-zero gradient)
Evaluated Variants:
Min-cap (?=1): Steep cliff, 70% steps contributed negligible weight
Decay min-cap: Smoothed early steps but cliff persisted
SNR ? 50% efficiency, prevented collapse, maximized data utilization
Normalization: Stabilized training but slowed convergence
Recommended: min sqrt decay, no per-batch normalization ? doubled effective gradient flow vs. min-cap
Temporal Modeling with PTSS
Motivation: Vectorized timesteps prevent combinatorial explosion (1000^N combinations) while enabling frame-wise flexibility
Probabilistic Sampling Strategy: Parameter p controls independent vs. shared timestep sampling
p=0.2–0.5 (epochs 1–50): Steady loss decrease, growing prediction variance
p>0.6: Excessive variance ? SMPL forward kinematics violations in root scale
p=0.2 with lr?2.5×10?? after epoch 60: Restabilized training
Best Practice: Start with low p, gradually increase, reset to low p if divergence detected
Key Benefit: Improved beat-alignment scores without increasing training time; primary advantage is preventing overfit to fixed schedules
Setup & Implementation
Optimizer: ADAN with initial learning rate 1×10??, decayed to 0.02
Data Processing: Resampled 60FPS ? 30FPS, 150 frames (5 seconds) per clip; multi-view 2D keypoints as conditioning
Diffusion Config: T=1000 timesteps, accelerated via DDIM with 50 sampling steps; batch size fixed at 32
Representation: SMPL axis-angle converted to 6D rotations via orthogonalization; foot contacts derived from vertical velocity thresholds
Training Data: AIST++ dataset (8 genres for training, 2 held out for testing)
Training Stability & Lambda Scheduling
Initial Collapse: Early configuration (?_joints=0.01, ?_vel=1.0, ?_foot=0.1) caused velocity loss dominance ? static poses (R????R?)
Root Cause: Direct matrix subtraction on SO(3) not physically meaningful; produced overwhelming gradients
Evolution Strategy:
?_joints ? 0.1 by epoch 25
?_vel ? 0.01 by epoch 40
?_foot ? 0.5 gradually
Critical Finding: Setting any auxiliary loss to zero triggered immediate mean-pose collapse; all three required non-zero weights throughout
Final Stable Config: ?_joints=0.1, ?_vel=0.02, ?_foot=0.05

pg.no. 16

<!-- Slide number: 17 -->
Architecture Ablations (FiLM Strategies)
Version 1 (Shared): Single linear layer across all modulation points ? parameter-efficient, enforces consistent conditioning
Version 2 (Separate): Distinct layers for self-attention, cross-attention, MLP ? finer control but 3× parameters
Small Models (d=128, n_heads=4): No performance difference
Large Models (d=256, n_heads=8): Version 2 converged marginally slower due to parameter increase
Production Choice: Version 1 selected to mitigate overfitting on 26k-sample dataset; difference negligible, novelty lies in conditioning strategy not FiLM multiplicity
Convergence Metrics & Final Performance
• Loss Component Targets:
L_joints: 0.01–0.02 (below 0.01=overfitting; above 0.03=??_joints)
L_vel: 0.05–0.10 (below 0.01=static; above 0.15=erratic)
L_foot: 10??–10?³ (above 0.01=excessive sliding; below 10??=overconstrained)
Training Dynamics: Prediction std dev increased from near-zero ? stable 0.1–0.2, confirming dynamic motion emergence
Final Results: FID_k=15.70, Div_k=4.85, beat-alignment score 0.239 on AIST++ test set
Benchmark: To Outperform MotionBERT baseline across all metrics

![A graph with different colored lines](Picture11.jpg)

<!-- Slide number: 18 -->
# Conclusion
Broader Implications: ViMo reveals that motion generation diversity and flexibility can be radically extended by treating abundant online video as a primary data source rather than relying on expensive motion-capture systems and limited manually-curated datasets. This paradigm shift positions casual video as an untapped reservoir for scalable, democratized motion capture, unlocking new possibilities for animation, interactive media, virtual production, and metaverse content creation.
Future Directions: This represents a first attempt at exploiting casual video content—the authors encourage future research to extend toward:
Complex Scenarios: Multi-person motion disentanglement and scene-level semantic constraints
Full Pipeline Integration: Physics-based imitation learning and adaptive policy refinement for simulation-ready control
Broader Ecosystem: Interactive media, virtual characters, and metaverse applications
Continued Development: Transformative potential awaits as the framework matures beyond single-character motions
Key Achievements: ViMo-Flow successfully addresses a previously under-explored problem—generating diverse, realistic 3D human motions directly from casual videos with dynamic cameras, occlusions, and complex editing where traditional methods catastrophically fail. The framework achieves this through a simple but effective diffusion-based approach that conditions on 2D pose sequences, bypassing the need for explicit camera parameter estimation that cripples conventional reconstruction pipelines.
Demonstrated Real-World Impact: The work opens three transformative applications:
Large-Scale Dataset Construction: Processed 52 Chinese classical dance videos into 750 high-quality 3D motion clips (63 minutes), addressing critical genre scarcity and demonstrating 5×–50× scalability potential using freely available online content
Few-Shot Stylization: Zero-convolution blocks enable arbitrary dance style transfer with minimal data while keeping pre-trained backbone frozen—achieving 56–66% win rate over direct fine-tuning and preventing catastrophic forgetting
Video-Guided Motion Editing: Seamless in-betweening, in-filling, and style blending (e.g., Charlie Chaplin ? Michael Jackson transitions) via masked denoising, eliminating labor-intensive manual keyframing
pg.no. 18
