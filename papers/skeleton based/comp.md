Composite Motion Learning with Task Control

PEI XU, Clemson University, USA and Roblox, USA
XIUMIN SHANG, University of California, Merced, USA
VICTOR ZORDAN, Roblox, USA and Clemson University, USA
IOANNIS KARAMOUZAS, Clemson University, USA

3
2
0
2

y
a
M
5

]

R
G
.
s
c
[

1
v
6
8
2
3
0
.
5
0
3
2
:
v
i
X
r
a

Fig. 1. Example of a physically simulated character performing composite motion with locomotion and aiming a weapon. The colors show the automatic
mixing of the combined inputs that change dynamically over time based on the state. As indicated in the inset, red denotes body parts that are vital for
locomotion while blue for aiming respectively. Our multi-objective approach learns this mixture along with imitation from two disparate reference motions
and two goal-directed task rewards for each action.

We present a deep learning method for composite and task-driven motion
control for physically simulated characters. In contrast to existing data-
driven approaches using reinforcement learning that imitate full-body mo-
tions, we learn decoupled motions for specific body parts from multiple
reference motions simultaneously and directly by leveraging the use of mul-
tiple discriminators in a GAN-like setup. In this process, there is no need
of any manual work to produce composite reference motions for learning.
Instead, the control policy explores by itself how the composite motions can
be combined automatically. We further account for multiple task-specific
rewards and train a single, multi-objective control policy. To this end, we pro-
pose a novel framework for multi-objective learning that adaptively balances
the learning of disparate motions from multiple sources and multiple goal-
directed control objectives. In addition, as composite motions are typically
augmentations of simpler behaviors, we introduce a sample-efficient method
for training composite control policies in an incremental manner, where we
reuse a pre-trained policy as the meta policy and train a cooperative policy
that adapts the meta one for new composite tasks. We show the applicability
of our approach on a variety of challenging multi-objective tasks involving
both composite motion imitation and multiple goal-directed control.

CCS Concepts: • Computing methodologies ? Animation; Physical sim-
ulation; Reinforcement learning.

Authors’ addresses: Pei Xu, Clemson University, 1240 Supply Street, North Charleston,
SC, 29405, USA, Roblox, USA, peix@clemson.edu; Xiumin Shang, University of Cali-
fornia, Merced, USA, xshang@ucmerced.edu; Victor Zordan, Roblox, USA, Clemson
University, USA, vbz@clemson.edu; Ioannis Karamouzas, Clemson University, USA,
ioannis@clemson.edu.

Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM.
0730-0301/2023/8-ART $15.00
https://doi.org/10.1145/3592447

Additional Key Words and Phrases: character animation, physics-based
control, motion synthesis, reinforcement learning, multi-objective learning,
incremental learning, GAN

ACM Reference Format:
Pei Xu, Xiumin Shang, Victor Zordan, and Ioannis Karamouzas. 2023. Com-
posite Motion Learning with Task Control. ACM Trans. Graph. 42, 4 (Au-
gust 2023), 18 pages. https://doi.org/10.1145/3592447

INTRODUCTION

1
Despite significant advancements in physics-based character con-
trol, the majority of existing techniques rely on reference data con-
sisting of motion capture recordings of an expert performing the
behavior of interest [Bergamin et al. 2019; Chentanez et al. 2018;
Lee et al. 2019; Park et al. 2019; Peng et al. 2018, 2022, 2021; Won
et al. 2020; Xu and Karamouzas 2021]. While such reference data
is paramount to train motor control policies that lead to natural
and robust control, in this paper, we are interested in synthesiz-
ing composite behaviors for physically simulated humanoids by
combining multiple motion capture reference clips into the training
of a single policy. Further, we augment these imitation controllers
with task-specific rewards to train the policy to accomplish specific
functional tasks at the same time. To this end, we propose a novel
multi-objective learning framework that builds composite motion
behaviors through multiple discriminators, each with its own dis-
tinct reference motion as well as task-level control. Our framework
is based on deep reinforcement learning, and allows us to adaptively
balance the learning of disparate motions from multiple sources and
also multiple goal-directed control objectives.

The motivation for this technique is twofold. First, humans are
capable of sophisticated behaviors, including performing multiple
tasks simultaneously, such as walking and gesturing or using a

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

2

• Xu, P. et al

mobile phone. To accomplish this with virtual characters, existing
control approaches need to be extended to accommodate the ability
to train with multiple objectives as a goal. Second, with limited ex-
ception, most current control frameworks rely on imitation with the
style of a behavior being derived from reference motion examples.
Our aim is to be able to combine examples automatically through
what we call “composite motion control” to avoid the need to con-
tinuously seek new example motions for every new permutation
of combined behaviors. We also explore the ability to add multiple
task objectives to support our aim of multi-objective control.

The core difference of our approach from existing imitation learn-
ing approaches is decoupling full-body control during training,
turning imitation and goal-directed full-body training into a multi-
objective learning framework. To this end, we propose a modifica-
tion to generative adversarial networks (GANs) to accommodate
multiple discriminators (for each subtask in the desired end behav-
ior) and to incorporate the mixing of the behaviors as a part of the
training. In this way, we sidestep the need to dictate weights for
combining the subtasks as well as the need to shape careful reward
functions manually for each new composite behavior. In addition,
as we expect composite motions to often be augmentations from
simpler behaviors, we introduce a method for learning composite
motion control policies from existing policies through incremental
learning. To this end, we train a meta policy, for example for walk-
ing, and then train a new policy to cooperate with the meta policy,
producing a composite motion control policy significantly faster
than learning from scratch. Thus, we can quickly add on to walking
new activities from reference data such as punching or waiving,
even if we do not have examples of these activities being combined
previously with the meta policy.

One naive approach to produce the composite motions we target
is to blend motion capture clips to produce a single new motion, and
perform traditional imitation learning from there. This suggested
technique may be plausible for simple composite behaviors, like
waiving an arm while walking as the two behaviors do not use the
same joints, nor do they influence each other greatly, and therefore
the blending can be done by simple splicing in a way that is fixed
over time. Even so, there is no guarantee of physical plausibility
without subsequent training – and the approach does not scale
for more complex behaviors which may have more complicated
tradeoffs between body parts used, especially over time. In contrast,
our approach offloads the need to create this weighting as it is
produced automatically by the policy as a part of the dictated action.
Likewise, the output of our system is automatically guaranteed to
be physically valid. Finally, our approach also has the capability to
add task-directed goals, such as walk to a specified location, which
is not possible without significant manual effort being added to the
naive approach described.

Overall, this paper makes the following contributions:

• We introduce a novel approach for physics-based character
control that decouples full-body control in order to learn
imitation and task goals from disparate sources and across
distinct body parts.

• To this end, we extend GAN-style reinforcement learning and
introduce a multi-objective learning framework to support

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

multiple discriminators and automatic weighting of imitation
and goal-driven subtask rewards.

• We propose an incremental learning scheme that uses a meta-
policy from an existing behavior to augment the behavior
with new subtasks, producing a composite motion control
policy that can be learned significantly faster than learning
from scratch. Our scheme automatically learns weights across
the body that are state dependent in order to effectively mix
the original behavior with a new subtask in a temporally
dynamic fashion.

2 BACKGROUND AND RELATED WORK
2.1 Physics-Based Character Control
Developing controllers for physically simulated humanoids has
wide applications in computer graphics, robotics, and biomechanics.
Over the years, a number of trajectory optimization approaches for
physics-based control have been proposed that leverage heuristics
or feedback rules [Coros et al. 2010; De Lasa and Hertzmann 2009;
Wampler et al. 2014; Ye and Liu 2010a; Zordan et al. 2014], includ-
ing open-loop control schemes[Liu et al. 2015, 2010; Mordatch et al.
2012], close-loop feedback control [da Silva et al. 2017; Mordatch and
Todorov 2014] and model predictive control approached [Hämäläi-
nen et al. 2015; Kwon and Hodgins 2010; Tassa et al. 2012, 2014].
Given the difficulty in controller design, which often involves multi-
ple optimization objectives, data-driven methods using demonstra-
tions from real humans has also drawn a lot of attention [Da Silva
et al. 2008; Kwon and Hodgins 2017; Lee et al. 2010; Liu et al. 2016,
2012; Muico et al. 2009; Sok et al. 2007; Yin et al. 2007; Zordan and
Hodgins 2002].

In recent years, with the advancement of machine learning tech-
niques, deep reinforcement learning frameworks have gained a
lot of popularity for training physics-based character controllers.
While some works [Karpathy and Van De Panne 2012; Won et al.
2018; Xie et al. 2020; Yu et al. 2018] purely rely on reward func-
tions designed heuristically or using curriculum learning to perform
control and encourage the character to act in an expected, human-
preferred style, most recent works leverage motion capture data
to perform imitation learning in order to generate high-fidelity,
life-like motions. DeepLoco [Peng et al. 2017] employs a hierar-
chical controller to perform walking-style imitation in navigation
tasks for a physically simulated character. DeepMimic [Peng et al.
2018] combines imitation learning with goal-conditioned learning,
and enables a physics-based character to learn a motor skill from
a reference motion collected by motion capture or handcrafted by
artists. Chentanez et al. [2018] explore the training of recovery
policies that would prevent the character from deviating signifi-
cantly from the reference motion. While the aforementioned works
rely on a phase variable to synchronize with the reference motion,
DReCon [Bergamin et al. 2019] utilizes a motion matching tech-
nique to find the target pose from a collection of reference motions
dynamically in response to user control input.

Besides relying on direct tracking of reference motions, researchers
have offered a number of ways to extend the use of reference data in
various ways. For example, Park et al. [2019] leverage the kinematic
characteristics of unorganized motions to generate target poses for

the control policy to imitate. UniCon [Wang et al. 2020] adopts a
similar strategy, where a high-level motion scheduler is employed
to provide the target pose for the low-level character controller.
MotionVAE [Ling et al. 2020] employs data-driven generative mod-
els using variational autoencoders to generate target motion poses
for a reinforcement learning based controller. A similar model is
employed by Won et al. [2022] and tested with various goal-directed
downstream tasks. To ensure synthesis of desired motions, these
approaches rely on carefully designed reward functions to assess
the controlled character motion. Drawn from GAIL [Ho and Ermon
2016; Merel et al. 2017], AMP [Peng et al. 2021] and ICCGAN [Xu
and Karamouzas 2021] avoid manually designing reward functions
by exploiting the idea of generative adversarial network (GAN)
and relying on a discriminator to obtain the imitation reward for
training.

Beyond the simple use of full-body motions, many works explore
motion generation by combining together multiple basic motions
with respect to different body parts [Alvarado et al. 2022; Jang et al.
2022, 2008; Liu and Hodgins 2018; Soga et al. 2016; Starke et al. 2021;
Yazaki et al. 2015]. However, these works focus on the editing and
synthesis of motion animation or using inverse kinematic solvers,
and do not work well with current frameworks for controlling phys-
ically simulated characters using reinforcement learning. To date,
existing works for physics-based character control solely focus on
the learning of full-body motions. As complementary to such works,
in this paper, we target composite motion learning from multiple
references without needing to generate any target full-body motion
for tasks involving both goal-directed control and imitation control.

2.2 Training Efficiency
Characters employed during physics-based control typically are
highly articulated with many degrees of freedom defined in con-
tinuous action spaces. Given the vast feasible choices of action,
controlling so many degrees of freedom is essentially ambiguous,
resulting in control problems that are under specified and highly
dimensional. A qualified control policy usually needs millions of
samples for training. The time consumption depends on the ex-
ploited algorithms and the motion complexity, varying from tens
of hours to several days. While some works such as [Yang and Yin
2021] explore approaches to speed up the training by improving
the reinforcement learning algorithm itself, a lot of attention has
been recently drawn on sample-efficient training by reusing pre-
trained policies or action models for fast new motion learning. For
example, many recent approaches employ mixture of experts (MoE)
models [Peng et al. 2019; Won et al. 2020, 2021], where a batch of
pre-trained expert policies are exploited to provide primitive actions
that are combined by a newly trained policy to generate the final
actions. Other approaches explore using pre-trained latent space
models such as variational autoencoders [Ling et al. 2020; Won et al.
2022] and GAN-based models [Peng et al. 2022] to facilitate the
training of a control policy. In such approaches, the latent space
model encapsulates a variety of reference motions and is used by
the control policy to generate motions for a specific task. The works
in [Merel et al. 2019, 2020] combine MoE with a latent space model
and rely on an encoder-decoder architecture to perform distillation

Composite Motion Learning with Task Control

•

3

for motion learning. Ranganath et al. [2019] utilize principal compo-
nent analysis to extract coactivations from reference motions and
use them as the atomic actions for motor skill learning.

Despite achieving impressive results, exploring the latent space
or learning how to combine expert policies is not always easier com-
pared to performing exploration directly in the original action space.
We note that all of these works focus only on reusing models that
provide full-body motions. In contrast, we propose an incremental
learning approach that allows a newly trained policy to take only
partial actions from a pre-trained policy, and add on that to generate
composite motions. Our approach can largely reduce the training
time for composite and multi-objective tasks involving multiple
imitation and goal-directed objectives as compared to training from
scratch.

2.3 Multi-Objective Control
In multi-objective character control, the reward function of the un-
derlying optimization problem is expressed as the weighted sum
of multiple, possibly competing, goals. Depending on the task in
hand, we seek for objective terms that encourage the character to
accomplish behavior goals, follow reference motion and/or style,
adopt certain behavior characteristics such as low energy move-
ment, attaining specified goals, etc., resulting in an extensive list of
objective terms (see [Abe et al. 2007; Macchietto et al. 2009; Muico
et al. 2009; Peng et al. 2018; Wu and Zordan 2010; Ye and Liu 2010a,b]
for some examples). But how we handle all these competing objec-
tives to create coherent, natural, and coordinated control remains an
open question. A common solution is to employ a manual weighting
scheme based on intuition, experience, and trial and error. However,
such approaches often require excessive, often tricky manual effort
to obtain desired results. While prioritized-based schemes have been
employed that optimize each term in the reward function based on
a given priority [De Lasa and Hertzmann 2009; De Lasa et al. 2010],
such schemes cannot automatically address the problem of multiple
competing objectives.

This problem becomes worse within a reinforcement learning
setting, as small changes in the reward function can have a signifi-
cant impact on the resulting behavior. It may need laborious work
to finetune the weight of each objective to ensure that the control
policy can effectively balance the learning of multiple objectives in
a desired way. For tasks with hierarchical objectives, hierarchical
reinforcement learning with multiple controllers can be employed,
where a different controller is selected at different task levels [Clegg
et al. 2018; Nachum et al. 2019; Peng et al. 2017; Xie et al. 2020].
However, such approaches cannot work for nonhierarchical tasks,
where different objective terms need to simultaneously be optimized
such as when the character has to perform composite motion imi-
tation and goal-directed control as in our problem domain. In our
approach, we propose the use of a multi-critic optimization scheme,
where each objective is regarded as an independent task and is as-
signed a separate critic. By evaluating each objective independently,
the contribution (gradient) of each objective can be normalized into
the same scale, and, thus, the control policy will be updated toward
each objective at the same pace. As such, we avoid scalarizing and

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

4

• Xu, P. et al

Fig. 2. Overview of the proposed system for composite motion learning with task control. Under the framework of reinforcement learning combined with a
GAN-like structure for motion imitation, our approach employs a multi-critic architecture to train a physics-based controller involving multiple objectives.
Based on this system, we further propose an optional incremental learning scheme that allows the control policy to fast learn new composite motions and
tasks by reusing a pre-trained, meta policy.

weighting the rewards or priorities of multiple objectives. In addi-
tion, our approach provides a simple solution to adaptively balance
the multiple objectives during policy updating without needing to
find or estimate the Pareto front.

3 OVERVIEW
Our approach enables a physically simulated character to perform
composite motions through imitating partial-body motions from
multiple reference sources directly and simultaneously. This scheme
turns the full-body motion imitation task into a multi-objective op-
timization problem, to which we can further introduce extra objec-
tives for goal-directed control. We refer to Fig. 2 for an overview of
our proposed system for composite motion learning with task con-
trol. We employ a GAN-like structure combined with reinforcement
learning to train the control policy imitating the given reference
motions. As such, we do not have to manually design a reward
function for imitation learning or explicitly track a target pose from
the reference motions. To learn composite motions, we decouple
the full-body motion into several partial-body groups each of which
imitates its own references. Based on this GAN-like structure, we
propose a multi-objective learning framework that exploits multi-
ple critics at the same time to help the control policy learn from
multiple objectives, involving both composite motion imitation and
goal-directed task control in a balanced way (Section 4). To acceler-
ate training, we further consider an optional incremental learning
scheme that reuses a pre-trained policy as the meta policy and al-
lows a cooperative policy to adapt the meta one for new composite
tasks (Section 5).

4 COMPOSITE MOTION LEARNING
Given a physically simulated character, we seek to train a control
policy ? (a? |s? , g? ) that simultaneously imitates motions from mul-
tiple reference ones, each focusing on specific body parts, while
possibly completing specific goal tasks. At each time step ?, the

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

control policy takes the character state s? and a dynamic goal state
variable g? as the input and outputs the control signal (action) a? .
We let g? be an empty variable if no goal-directed control is involved.
In the following, we detail our proposed approach for training ?
that decouples full-body motion allowing imitation performance to
be evaluated and improved with respect to specific body parts, and
converts the underlying composite motion learning problem into a
multi-objective optimization problem.

?link
?=1

4.1 Full-Body Motion Decoupling
At each time step ?, we represent the character pose as P?
:=
, where ?? ? R3 and ?? ? R4 are the posi-
{(?? , ?? , (cid:164)?? , (cid:164)?? )|? }
tion and orientation (measured in the unit of quaternion) of each
body link respectively, and (cid:164)?? ? R3 and (cid:164)?? ? R3 are the linear and
angular velocities respectively. Given the geometry model and joint
constraints of the simulated character, this representation can be
converted into a joint space one defined by the skeletal joints’ local
position and velocity and the root’s global position and orientation.
Let M ? { ˜P? }? be the collection of reference motions which may
contain multiple clips of pose trajectories { ˜P? }? as the reference.
To perform imitation learning, existing approaches either use a
carefully designed reward function to compute the error between
P? +1 and ˜P? +1 [Bergamin et al. 2019; Chentanez et al. 2018; Park
et al. 2019; Peng et al. 2018; Won et al. 2020], or employ an evaluator
to assess the transfer P? ? P? +1 without explicitly comparing
to any specific poses in the reference motions [Merel et al. 2017;
Peng et al. 2021; Xu and Karamouzas 2021]. The former approaches
usually need a motion tracking or generation mechanism to retrieve
˜P? +1 from the reference motions. The latter typically build on the
framework of adversarial generative networks (GANs) and rely
on a discriminator to evaluate the transfer. Some approaches take
poses from more than one frame during imitation performance
evaluation in order to apply more constraints on the pose trajectory.

Discriminator EnsembleDiscriminator EnsembleDi˜oitReference MotionsMiControl Policy?Multiple CriticsoitGoal-Directed Task RewardsrDitrgitSimulated CharacterMeta Policymeta? ,statgmetatPhysics Simulator120Hz30HzIncremental      Learning      CritickPD ControllerImitation Rewardsgt ,smetatametatNevertheless, all these approaches leverage the full-body character
pose P? and reference pose ˜P? ? M to perform imitation learning,
and thus intend to learn the full-body motions in M.

To learn composite motions, ideally, we want the simulated char-
acter’s partial body motions to come from different reference sources
at a given time step ?, i.e., the transfer of pose trajectory P?
? ??? :? ?
P?
? +1

should satisfy

{P?

? ???

, · · · , P?

? , P?

? +1} ? M?,

(1)

where P?
? ? P? is a partial-body pose from the simulated charac-
ter, and M? ? { ˜P?
? }? is the reference motion collection containing
only poses of the partial body group ?. The full-body motion is con-
strained by using multiple M? at the same time. Here, we follow Xu
and Karamouzas [2021] and use a pose trajectory having ?? + 2
frames for imitation performance evaluation. The larger ?? is, the
stricter the evaluation will be, as an error occurring at an earlier
time step would negatively influence the evaluation of the following
steps.

? and some other partial-body poses P

Typical partial body groups for a humanoid character would be
the upper and lower body, arms, and torso. For example, we can
let Mupper be a collection of greeting motions involving the upper
body (arms, hands, torso and head), and Mlower be walking motions
involving the lower body (pelvis, legs and feet). Then, the full body
motion is expected to be the composite of Mupper (greeting) and
Mlower (walking). To coordinate the motions from multiple body
?
groups, we can let P?
? share
upper
and Plower
some common body link states. For example, let P
?
?
share the state of one leg to avoid ipsilateral walking. Correspond-
ingly, the leg state should be included in both Mupper and Mlower
for the control policy to learn. We refer to Sections 6 and 7 for body
splitting schemes used in our experiments, including typical upper
and lower body decoupling schemes and more tailored ones for
specific tasks such as juggling while walking. After decoupling the
character’s full-body motion into multiple sets of {P?
? }? , we perform
imitation learning with respect to each body group independently,
where the control policy is expected to explore how to combine
partial-body motions by itself without needing any full-body, com-
posite motions to be provided as the reference.

Imitation Learning

4.2
To perform imitation learning, we build our approach off of GAN-
like frameworks [Ho and Ermon 2016; Merel et al. 2017], which
utilize a discriminator to evaluate imitation performance and gen-
erate reward signals for policy optimization using reinforcement
learning algorithms. However, instead of using only one discrimi-
nator to perform full-body imitation performance evaluation, we
employ multiple discriminators simultaneously, each of which deals
with a body part group ? associated with a collection of partial-body
reference motions M? . Based on this framework, we can avoid de-
signing reward functions to compute the imitation error for each
specific body part group. Furthermore, each discriminator can take
only its own interested body link states as input during training.
Therefore, the provided M? can still be a collection of full-body
motions, but there is no need to explicitly generate any partial-body
motions during preprocessing.

Composite Motion Learning with Task Control

•

5

To stabilize the adversarial training process, we introduce a hinge
loss [Lim and Ye 2017], gradient penalty term [Gulrajani et al. 2017],
and an ensemble technique for training of discriminators as pro-
posed in [Xu and Karamouzas 2021]. Following the literature, given
? as the observation sampled from the simulated character and ˜o?
o?
?
as that sampled from the reference motions M? , the ?-th ensemble
of ? discriminators, ?? = {??
? |? = 1, · · · , ? } is trained using the
loss function:

L?? =

1
?

?
??

(cid:16)

?=1

E? (cid:2)max(0, 1 + ??

? ))(cid:3) + E? (cid:2)max(0, 1 ? ??
?
? (o

?
? ))(cid:3)
? (˜o

+?GP

E?

(cid:104)

(||?^o?

?

??

?
? (^o

? )||2 ? 1)2(cid:105) (cid:17)

(2)

where ^o?
? + (1 ? ?)˜o?
gradient penalty coefficient.

? = ?o?

? with ? ? Uniform(0, 1) and ?GP is

According to Eq. 1, we define the observation space of a discrimi-

nator as

(3)

? ???

, · · · , P?

?
? := {P?
o

? , P?
? +1}.
In principle, the discriminator relies on o?
? to evaluate the con-
trol policy’s performance during the state-action-state transition
(s? , a? , s? +1). The observation space theoretically should satisfy
o?
? ? {s? , s? +1}. Otherwise, the discriminator may rely on features
unknown to the control policy, and thus it cannot effectively evalu-
ate the policy’s performance. Given that the control policy ? in our
formulation is still a full-body control policy, we simply define s? as
a full-body motion state:

s? := {P? ??, · · · , P? }
where ? ? ?? for all ?. We refer to the Appendix in the supple-
mentary material for more details about the state and observation
representation.

(4)

The hinge loss function provides a linear evaluation between
[?1, 1] to measure the similarity of a given pose trajectory sample
o?
? to any sample in the reference motions. Therefore, we define the
reward term that evaluates the policy’s imitation performance with
respect to M? for the body part group ? at time ? as:

? ??
?

(s? , a? , s? +1) =

1
?

?
??

Clip (cid:16)

??

?
? ), ?1, 1
? (o

(cid:17)

.

(5)

?=1
It must be noted that even though o?
? in Eq. 2 have the same
subscript ?, they are paired only for the gradient penalty compu-
tation (last term in Eq. 2). The discriminator ensemble here only
evaluates the pose trajectory o?
? independently, rather than com-
paring it against any specific target trajectory. Therefore, ˜o?
? can be
randomly sampled from the reference motions by interpolation.

? and ˜o?

Overall, by employing multiple discriminator ensembles at each
time step ?, we will have a set of rewards, {? ??
? }?? , to evaluate
the policy’s performance of controlling the character to perform
composite motions, i.e. simultaneously imitating different sets of
reference motions corresponding to specific partial body parts. By
doing so, we convert the task of composite motion learning to
a multi-objective optimization problem under the framework of
reinforcement learning.

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

6

• Xu, P. et al

4.3 Multi-Objective Learning
We consider policy optimization of a typical on-policy policy gradi-
ent algorithm by maximizing

(6)

L? = E? [?? log ? (a? |s? , g? )],
where s? and g? are the given character’s and goals’ state variables
respectively, and ?? is the advantage which is typically estimated
by {?? }? ?? . In the common actor-critic architecture, a separate net-
work (critic) is updated in tandem with the policy network (actor).
The critic is employed to provide state-dependent value estimation,
? (s? ) = E? [(cid:205)? ?? ?????? ] = E? [?? + ?? (s? +1)], based on which
?? can be estimated with less variance, where ? is the discount
factor regulating the importance of the contribution from future
steps. To stabilize the training, standardization is often applied on
?? where the standardized advantage ¯?? is used in place of ?? for
policy updating.

?=1

????

A typical solution for multi-objective tasks in reinforcement learn-
ing is to simply add together all objective-related reward terms, ??
? ,
with some weights ?? , i.e., ?? = (cid:205)?
? for a ?-objective prob-
lem. In such a way, we still have a scalar reward that can be used
with Eq. 6 for policy updating. In practice, though, given that con-
flicts may exist among the different reward terms, manually tuning
the values of ?? to balance the composite objective of the character
is not an intuitive task. For example, we may need the policy to put
more effort into learning a difficult partial-body motion, instead of
even with a trade-off in learning other motions, rather than only
focusing on the easy ones to keep achieving a higher associated
reward. In addition, our proposed approach performs reward estima-
tion by employing multiple discriminators simultaneously, which
are modeled by neural networks. This scheme brings a lot of uncer-
tainty, as the reward distributions from different discriminators may
differ a lot depending on the given reference motions, which could
be unpredictable before training. Such a problem would deteriorate
if we further introduce a set of goal-directed tasks, each having
its own associated reward term which may compete against the
imitation reward terms.

To balance the contributions of multiple objectives during policy
updating, we propose to model the multi-objective learning problem
as a multi-task one, where each objective is taken into account as an
independent task and has a fixed importance during policy updating.
To do so, instead of using ?? = (cid:205)? ????
? , we compute the advantage
of ??
? with respect to {??
? }? ?? independently. Then, the optimization
process becomes maximizing

?
??

E?

L? =

(cid:104)
?? ¯??

? log ? (a? |s? , g? )

(cid:105)

,

?=1
where (cid:205)? ?? = 1 and ¯??

? is the standardization of ??

? , i.e.

¯??

? =

??
? ? E? [??
? ]
??
Var? [??
? ]

.

(7)

(8)

This optimization process is equal to updating the policy with re-
spect to each objective independently but always at the same scale
proportional to ?? . The introduction of ?? gives us more flexibility
to adjust the contributions toward each objective when conflicts

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

occur during policy updating. However, under our testing, a simple
choice of ?? = 1/?, which means each objective is equally impor-
tant, works well for most cases. We refer to the Appendix in the
supplementary material for the choice of ?? in our tested composite
tasks.

During implementation, we can rewrite Eq. 7 as

L? = E?

(cid:104)(cid:16)??
?

(cid:17)

?? ¯??
?

log ? (a? |s? , g? )

(cid:105)

(9)

such that the policy update can be done through backward propa-
gation in one pass. From this equation, we can see that the nature
of our approach is to introduce a dynamic coefficient constrained
by the standard deviation of {??
? }? for each objective ?. As such,
the policy will be updated with respect to each objective adaptively.
This separation of objectives leads to a single-policy multi-critic
architecture. In Fig. 2, for example, we have two imitation related
reward terms (yellow and green) for upper and lower body imita-
tion respectively, and two goal-directed task reward terms (red and
blue). Accordingly, we employ four critics denoted by Critic? in
the figure. Each Critic? only participates in the estimation of ??
? ,
and takes the reward associated with the objective ?, i.e. {??
? }? , for
training.

Though the policy update is balanced through the proposed multi-
critic architecture, the state values, which are decided by {??
? }? ,
could differ still drastically with respect to each objective depending
on the difficulty of given reference motions or the reward distribu-
tions of the goal-related tasks. To mitigate this issue and stabilize the
training of critics, we introduce the value normalization scheme of
PopArt [van Hasselt et al. 2016]. The value target under this scheme
is normalized by the moving average and standard deviation for
the critic network training. The output of a critic is unnormalized
before joining the process of advantage estimation. Besides main-
taining a normalizer for value targets, PopArt is designed to preserve
the output precisely. Namely, with PopArt, the output of a critic is
identical before and after the normalizer updates given the same
input state s? and g? . Such a design is to prevent the normalization
from affecting the value state estimation, thereby stabilizing the
policy training. In our implementation, each critic Critic? (s? , g? )
has its own normalizer with a scalar scale and shift estimated in-
dependently with respect to its associated objective ?. As we show
in Section 6.6, the introduction of PopArt helps improve the policy
performance as also demonstrated by previous works [van Hasselt
et al. 2016; Yu et al. 2021].

INCREMENTAL LEARNING

5
Besides being able to perform a range of composite motions, hu-
mans typically learn such motions in an incremental manner. For
example, if we know how to walk, we should be able to quickly
learn how to hold our phone while walking. There is no need to
relearn walking from scratch. Based on this intuition, we propose
an incremental learning scheme for fast composite motion learning.
Instead of training a policy completely from scratch, we reuse a
pre-trained policy as a meta policy ? meta that allows the simulated
character to perform a basic set of motions (walking in the previ-
ous example). Given ? meta, we train a new policy ? to cooperate

ALGORITHM 1: Multi-Objective Incremental Learning
1 Prepare the meta policy ? meta;
2 initialize the policy network ? ;
3 initialize the critic network Critic? where ? = 1, · · · , ? given ?

objectives in the task;

4 initialize policy replay buffer T and reward buffer R;
5 prepare reference motions M? for each discriminator ensemble ?? ;
6 while training does not converge do
7

? ? environment updates with character

? from the state pair of s? and s? +1 for

?

?

);

T ? ?, R ? ?;
for each environment step ? do
, gmeta
?
);

ameta
? ? meta ( · |smeta
?
a? ? ? ( ·, |s? , g? , ameta
s? +1, g? +1, rg?
control signal of a? ;
extract observation o?
each discriminator ensemble ?? ;
T ? T ? { (s? , a? , {o?
R ? R ? {? ?
s? ? s? +1; g? ? g? +1;
extract smeta
and gmeta
?

?

end
for each discriminator ensemble ?? do

draw samples ˜o??
?
update ?? using o?
for each o?
? in T do

from M? ;
? from T and ˜o?

? }? ) };
? +1 } for each term ? in rg?

?

;

from s? and g? respectively

? based on Eq. 2;

compute step-wise imitation reward ? ??
R ? R ? {? ??
?

}

?

based on Eq. 5;

end

? }? in R do

? using {? ?

? }? ?? and state value

end
for each reward term collection {? ?
compute advantage ??
estimation from Critic? (s? , g? ) unnormalized by PopArt;
compute value target ? ?
? based on ??
? ;
update the normalizer for Critic? based on ? ?
PopArt;
get normalized value target ¯? ?
get normalized advantage ¯??

? by PopArt;
? based on Eq. 8

? using

end
for each policy update step do

update ? using { (s? , a? , { ¯??
update each critic network Critic? using { ¯? ?

? }? ) }? based on Eq. 9;
? }?

8

9

10

11

12

13

14

15

16

17

18

19

20

21

22

23

24

25

26

27

28

29

30

31

32

33

34

35

end

36
37 end

with the meta policy, performing new composite motions by action
addition (holding a phone + walking).

Formally, let ? (a? |s? , g? ) := N (?? , ? 2

? ) denote a Gaussian-based
policy. By introducing a meta policy ? meta, we define the policy,
which is trained to cooperate with ? meta for new composite motions
as

meta
? (a? |s? , g? , a
?

) := N

(cid:16)

= N

(cid:16)

(cid:17)

?? , ? 2
?
?? + w? Stop (cid:16)

+ w? Stop (cid:16)
(cid:17)

meta
a
?

(cid:17)

meta
a
?

(cid:17)

,

, ? 2
?

(10)

Composite Motion Learning with Task Control

•

7

?

, gmeta
?

? ? meta (·|smeta

where the weight vector w? has the same dimension with ameta
, and
ameta
) is drawn from the meta policy. w? are
?
defined as a set of weights each of which is associated with a DoF in
the action space of the meta policy. In our implementation, w? , ??
and ?? are obtained by a neural network taking s? and g? as input,
and thus are learnable. We put a "gradient stop" operator, Stop(·),
on ameta
, which means that the meta policy is fixed and will not be
?
updated with ?.

?

?

Using this incremental learning scheme, the new, cooperative
policy adds its own action to the meta action ameta
. The weight
vector w? decides the reliance of ? on the meta policy ? meta with
respect to each DoF in the action space. The bigger an element in
w? is, the more the cooperative policy relies on the meta policy to
control the corresponding DoF. As such, ? is trained incrementally
to learn new composite motions by reusing the meta policy partially.
This scheme does not require that ameta
and a? must have exactly
the same dimension, as we can assume zero values for the missing
dimensions in ameta
or ignore the extra, uninteresting dimensions
in ameta
. Compared to a mixture-of-experts (MoE) model, where the
?
action is obtained by a linear combination of the actions from mul-
tiple expert policies, our approach focuses on reusing partial-body
motions from the meta policy. It would be very difficult for a MoE
model to keep, for example, only the lower-body motion of one ex-
pert and replace the upper-body motion with that of another expert
through a linear combination of the experts’ full-body motions.

?

?

?

With the introduction of ? meta, we can replace ? (a? |s? , g? ) in
Eq. 7 with ? (a? |s? , g? , ameta
), and perform composite motion learn-
ing with goal-directed control under our proposed multi-objective
learning framework. We refer to Algorithm 1 for the outline of
the proposed multi-objective learning framework with incremental
learning. To train a composite policy completely from scratch with-
out using incremental learning, we can simply ignore ? meta and use
? (a? |s? , g? ) solely in Algorithm 1.

6 EXPERIMENTS
In this section, we experimentally evaluate our approach on multi-
ple challenging composite motion learning tasks. We show that our
approach can effectively let motor control policies learn composite
motions from multiple reference motions directly without manually
generating any full-body motion as reference. Besides evaluating
the imitation performance, we also apply our approach on several
goal-directed control tasks combined with composite motion learn-
ing from unstructured reference data. The results demonstrate that
our proposed approach can successfully tackle complex tasks balanc-
ing the learning of multiple objectives involving both partial-body
motion imitation and goal-directed control. Finally, we perform ab-
lation studies on our proposed multi-objective learning framework
and incremental learning scheme.

Implementation Details

6.1
We run physics-based simulations using IsaacGym [Makoviychuk
et al. 2021], which supports simulation with a large number of in-
stances simultaneously by leveraging GPU. The simulated humanoid
character has 15 body links and 28 DoFs, where the hands are fixed
with the forearms and are uncontrollable. In the tasks involving a

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

chat history: https://chatgpt.com/share/68cc739c-ff80-8009-b83d-cc610405e883

8

• Xu, P. et al

(a) Policy Network

(b) ?-Head Critic

(c) Discriminator Ensemble

Fig. 3. Network structures. ? denotes the concatenation operator and ?
denotes the average operator.

tennis player, we add 3 DoFs on the right wrist joint such that the
character can control the racket more agilely, though the racket is
fixed on the right hand. The simulation runs at 120Hz and the con-
trol policy at 30Hz. Differing from the previous works that employ
a stable PD controller [Tan et al. 2011] for character control [Lee
et al. 2022, 2021; Park et al. 2019; Peng et al. 2018, 2021; Won et al.
2020, 2022; Xu and Karamouzas 2021] we employ a normal, linear
PD servo for faster simulation.

We use PPO [Schulman et al. 2017] as the base reinforcement
learning algorithm for policy training and Adam optimizer [Kingma
and Ba 2014] to perform policy optimization. To embed the charac-
ter state s? and the discriminator observation o?
? sequentially, we
employ a gated recurrent unit (GRU) [Chung et al. 2014] with a
256-dimension hidden state to process these temporal inputs. The
embedded character state feature is concatenated with the dynamic
goal state g? if goal-directed control is involved, and then passed
through a multilayer perceptron with two full-connected (FC) layers.
The control policy is constructed as Gaussian distributions with in-
dependent components. The output of the policy network includes
the mean ?? and standard deviation ?? parameters of the policy
distribution as well as a weight vector w? when incremental learn-
ing is exploited. The multiple critics in our multi-objective learning
framework are modeled by a multi-head neural network. Similarly
to the critic networks, we model a discriminator ensemble using a
multi-head network. The outputs are averaged by Eq. 5 to produce
the reward signal. All the network structures are shown in Fig. 3, in
which we assume that there are ? objectives in total. We refer to
the Appendix in the supplementary material for the representation
of g? in our designed goal-directed tasks, and all hyperparameters
used for policy training.

All the tested policies were trained on a machine equipped with
an Nvidia V100 GPU. It typically takes about 1.5h to train a policy
using a fixed budget of 20M samples (environment steps), for a
pure composite motion imitation task. For complex tasks involving
goal-directed control, it takes about 15 to 30 hours and requires
about 2 × 108 to 4 × 108 samples to train a policy from scratch. By
exploiting our incremental learning scheme to reuse a pre-trained
meta policy, we can shorten the training time to about 30 minutes
to 2 hours depending on the difficulty of the tasks.

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

6.2 Data Acquisition
All the motion data used for training are obtained from the LAFAN1
dataset [Harvey et al. 2020] and other commercial and publicly
available motion capture datasets recorded at 30Hz. For single-clip
imitation, we synthesize short reference motion clips of 1-3 seconds
long (cf. Table 1). For tasks with goal-directed control, we extract
several collections of motions (cf. Table 2), each of which contains
multiple clips of reference motions with lengths varying from about
15 to 70 seconds. The juggling motion involves a single trial of a
subject performing juggling while standing on a skate, while the
collection of tennis swing motions contains four trials of forehand
swings captured from different subjects. We retarget the local joint
position from those motion data to our character model without
extra manual reprocessing. We demonstrate that policies trained
with our approach can perform motion synthesis from unstructured
data for goal-directed control, and can explore how to perform
composite motions by combining the partial-body motions from
the reference motions without needing any manual processing for
motion blending.

Imitation Performance

6.3
In Fig. 4, we highlight motion pose snapshots captured from some
of our trained policies for composite motion learning. Each com-
posite motion is learned based on two reference motion clips, one
for the upper body and the other one for the lower body. From top
to bottom, the names of corresponding motions are listed in Ta-
ble 1. Overall, policies trained with our approach can perform very
challenging composite motor skills by using the character’s upper
and lower body part groups at the same time. For example, in the
motion combination of chest open and jumping jack (1st row), the
control policy must keep the character’s body balanced to perform
the chest-open motion during jumping in the air, which is a pretty
challenging task even for humans. Similar challenges arise when
doing squats with the chest open (3rd row) and lunges with waist
twisting (4th row). Besides simply following the two partial-body
reference motions at the same time, the control policies must master
how the partial motions could be combined such that the full-body
motion is physically plausible. In the 4th row, for example, it is
impossible for the character to keep twisting its waist while doing
lunges at quite different frequencies. Similarly, in the motion combi-
nation of punch and walk (6th row) and that of punch and run (7th
row), the character’s foot has to contact the ground first in order
to perform the punch action with the torso leaning forward. The
control policy, thereby, must know when the punch action is doable
and arrange the motion combination by itself, rather than strictly
following the reference motions. Our approach does not require the
given reference motions to be perfectly synchronized. The control
policies take the character state as input and perform composite
motions accordingly. Furthermore, the proposed dynamic sampling
rate (see Appendix) allows the control policy to adjust the motion
speed within an acceptable range for better motion combining.

To quantitatively evaluate the imitation performance, following
previous literature [Harada et al. 2004; Peng et al. 2021; Tang et al.
2008; Xu and Karamouzas 2021], we leverage the technique of fast
dynamic time warping (DTW) and measure the imitation error as

GRU st gt wt ?t log?tFC
(1024)FC?(512)GRU st gt v1t vKtGRU oit… v2t rDitFC
(1024)FC?(512)FC
(256)FC?(128)…GRU oit rDitFC
(256)FC?(128)GRU st gt v1t vKt… v2tFC
(1024)FC?(512)…GRU oit rDitFC
(256)FC?(128)…Composite Motion Learning with Task Control

•

9

Table 1. Imitation performance when learning composite motions from
single clips of reference motions.

Composite Motion
Chest Open
Front Jumping Jack (lower)
Front Jumping Jack (upper)
Walk In-place
Chest Open
Squat
Waist Twist
Leg Lunge
Hand Waving
Walk
Punch
Walk
Punch
Run

Length [s]
2.10
1.80
1.80
2.10
2.10
1.67
3.37
3.67
1.80
1.10
1.30
1.10
1.30
0.76

Imitation Error [m]
0.11 ± 0.02
0.16 ± 0.03
0.30 ± 0.03
0.29 ± 0.02
0.10 ± 0.01
0.09 ± 0.01
0.15 ± 0.04
0.13 ± 0.02
0.06 ± 0.03
0.09 ± 0.02
0.11 ± 0.02
0.10 ± 0.01
0.17 ± 0.03
0.14 ± 0.01

follows:

?? =

1
? ?

link

? ?
link
??

?=1

||?? ? ˜?? ||,

(11)

link = |{P?

where ? ?
? }| is the number of interesting body links in the
?-th body part group, ?? ? R3 is the position of the body link ? in the
world space at the time step ?, and ˜?? is the body link’s position in
the reference motion. The evaluation results are shown in Table 1.
Our approach can imitate the reference motions closely and balance
the imitation of the two partial-body motions well. As can be seen,
there is no big gap between the two imitation errors in a given
composite motion combination, which means that policies trained
with our approach do not just follow only one reference motion and
ignore the other one. In contrast, without using our proposed multi-
objective learning framework, the policy could prefer to track only
one reference motion that is easy to follow. We refer to Section 6.6
for the related ablation study.

6.4 Goal-Directed Motion Synthesis
To test our approach with more complex tasks involving both com-
posite motion learning and goal-directed control, we designed five
goal-directed tasks, as shown in Figs. 5 and 6. In the Target Heading
and Target Location tasks illustrated in Figs. 5a and 5b, the char-
acter is asked to respectively go along a target heading direction
and toward a target location at a preferred speed. Besides the goal-
directed objective, two motion imitation objectives are employed:
one is for the lower-body and the other one is for the upper body.
Differing from the examples shown in Fig. 4 where the walking and
running motions are just single, short clips containing only one
gait cycle, here we use a collection of unstructured walking and
running motions as the reference for the lower body, as listed in
Table 2. In the three examples shown in Fig. 5a, the upper body
motions are learned from single reference motion clips, which are
chest open, jumping jack, and punch respectively, as depicted by
the small snapshots in the figure. In the examples shown in Fig. 5b,
we use the motion collection of tennis footwork as the reference

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

Fig. 4. Composite motions learned from multiple single-clip reference mo-
tions. The two snapshots shown on the left side of each row are the reference
motions for the upper and lower body respectively.

10

• Xu, P. et al

(a) Tasks: Target Heading (Directional Walking with Various Upper-Body Motions)

(b) Task: Target Location (Run) with Tennis Racket Holding

(c) Task: Tennis Swing (Forehand Swing with Footwork)

Fig. 5. Motion synthesis with composite motion learning and goal-directed control. Pose snapshots shown in the small windows are captured from the
reference motions.

(d) Task: Target Location (Walk) while Juggling

Table 2. Motion collections used for goal-directed control.

Motion Collection
Crouch
Walk
Run
Tennis Footwork
Tennis Swing
Aiming
Juggling

# of Clips
4
8
4
2
4
2
1

Length [s]
88.87
334.07
282.87
31.67
13.33
48.77
24.63

for the control policy to learn how to hold the racket. This task
is relatively harder, as the reference motions for both the upper
and lower body are unstructured. While following the reference
motions closely, the control policies trained with our approach can
effectively coordinate the character’s upper and lower body poses
to perform the composite motions during goal-steering navigation.
In the task of Tennis Swing, the character is expected to hit the
ball successfully with a forehand. The provided collection of ten-
nis swing motions contains four trials, where the subject performs
forehand swings while standing still. The tennis ball in our im-
plementation is generated randomly in a small region near the

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

character. As such, the control policy has to rely on the lower-body
footwork motions to properly adjust the pose and position of the
character relative to the tennis ball, while it relies on the upper
body swing motions to swing effectively and on time. We note that
the goal-directed reward in our design only evaluates the effective-
ness of hitting based on the ball’s outgoing speed and destination.
The motion otherwise is decided completely by the control policy,
which leverages two discriminator ensembles to perform imitation
learning for the upper and lower body respectively.

The Tennis Swing task is challenging, as it is easy for the controlled
character to solely hit the ball, but instead it is asked to do so
by combining the motions from the reference collection (tennis
swing for the upper body and tennis footwork for the lower body).
The policy needs some exploration before finding a way to utilize
poses from the reference motions to perform swings. In this process,
imitation learning would fail if the policy simply tries to pursue a
higher reward by simply hitting the ball. However, when the policy
is trained using our proposed multi-objective learning framework, it
can balance the imitation and goal-directed objectives, and perform
forehand swings in the style of the reference motions. Additionally,
while we provide only a small set of upper and lower body motions
as the reference (cf. Table 2), the control policy successfully learns

Composite Motion Learning with Task Control

•

11

(a) Meta Policy Tasks: Target Location (Crouch, Walk and Run)

(b) Incremental Learning Tasks: Directional Aiming while Location Targeting (Crouch, Walk and Run).

Fig. 6. Demonstration of incremental learning tasks, where goal-directed aiming motions are added to various locomotion behaviors from the meta policies.

how to combine the motions automatically to finish the task. In
contrast, if we just leverage full-body reference motions, extra work
is needed to generate various motions for the policy to learn. In
addition, there are not enough demonstrations for the policy to
perform tennis swings correctly in a human-like style by utilizing,
for example, only standing swing motions without footwork.

Figure 5d shows another challenging composite task: Target Loca-
tion while Juggling, where the character needs to juggle three balls
while walking to the target location. This composite task involves
four objectives: two imitation objectives and two goal-directed tasks
of juggling and locomotion. In our experiment, when a ball is rela-
tively close to a hand, it is assumed to be caught by and attached to
that hand. The ball is automatically detached from hand at a fixed
interval of 20 frames. In order to perform juggling successfully and
successively, after a hand releases its ball, it must catch in time a fly-
ing target ball which was thrown by the other hand. This task is very
challenging, as the control policy must explore how to perform ball
throwing and catching in concert with the location-targeting task.
Besides the difficulty of throwing and catching balls, the juggling
reference motion involves a subject balancing on a skateboard with
the body swaying from side to side 1. This increases the difficulty
of composite motion learning to generate normal walking poses.
Differing from the other examples that use a lower and upper-body
split, here we decouple the body parts into two groups, where one
group consists of the character’s arms to imitate the juggling motion
and the other group includes the rest of the body parts (torso, head,
pelvis, and legs) taking the collection of walking motions as refer-
ence data. In such a way, our approach can effectively eliminate
the body swings in the juggling reference motion, and generate

1FreeMoCap Project: https://github.com/freemocap/freemocap

composite motions with the upper body moving naturally during
goal-steering navigation.

The other goal-directed task explored in this study is Aiming, in
which the character holds a toy weapon in its right hand and is
expected to aim it toward a specific direction. In our experiments,
that task is designed mainly to demonstrate the effectiveness of our
proposed incremental learning scheme, which will be elaborated
in the next section. We refer to the Appendix for the details of the
setup of all of our goal-directed tasks, and the supplementary video
for related animation results.

Incremental Learning

6.5
In Fig. 6, we show tasks used to test our proposed incremental
learning scheme. The first row depicts three meta policies of loco-
motion, which are trained for the Target Location task completely
from scratch using our proposed multi-objective learning frame-
work. In contrast to previous examples, there is only one imitation
objective about the full-body during training here, as shown by
the snapshots on the top-left corner of the figure. In the 2nd row
of the figure, we show the cooperative policies that are trained by
incremental learning, while reusing the pre-trained, meta policies.
In addition to the Target Location task, a new goal-directed task of
Aiming is introduced during training the cooperative policies. The
controlled character in this task needs to adjust its right forearm and
let the toy pistol aim toward a goal direction specified dynamically.
The goal of this experiment is to demonstrate that the cooperative
policies can properly exploit the meta policies to perform styled
locomotion behaviors while quickly learning upper-body motions
from the newly provided aiming reference motions, which also in-
volve a new goal-directed task that is never seen by the meta policies.
In Fig. 7, we visualize the weight vector w? (cf. Eq. 10) for each DoF

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

12

• Xu, P. et al

(a) Aiming+Crouch

(b) Aiming+Walk

(c) Aiming+Run

(d) Crouch+AimingWalk

Fig. 7. Visualization of the incremental learning weight w? (cf. Eq.10). The azure character shows the behavior from the meta policy. The colored character is
controlled by the cooperative policy. The body link color identifies the weight for the associated DoF. The redder color represents higher weights, which means
that the cooperative policy relies more on the meta policy to control the corresponding body parts of the character. The bluer color represents lower weights,
which means that the cooperative policy mainly relies on itself to control the related body parts.

Fig. 8. Distributions of the incremental learning weights w? for the tasks of Aiming+Crouch and Crouch+AimingWalk (cf. Fig. 7). The x-axis depicts the
learned weights and the y-axis shows the corresponding distribution density, normalized by the total number of samples per body part grouping. The color
saturation binds the weight range for higher distribution density, with brighter colors highlighting weights greater than 0.5. In the first task, the lower body is
mainly controlled by the meta Crouch policy (high weights), while in the second task the AimingWalk meta policy mainly influences the upper body.

by coloring the associated body link. The first three examples show
the results obtained when we add the aiming motions to the meta
policies of locomotion. The fourth example shows the correspond-
ing result of adding the crouch motion to the meta policy of aiming
and walking. As opposed to the previous meta policies, this meta
policy has four objectives: two imitation objectives for the upper
(aiming) and lower (walking) body respectively, one Target Location
task and one Aiming task.

As shown in the figure, in the three Aiming+Locomotion tasks
where the meta policies are pre-trained for locomotion, the coopera-
tive policies rely more on the meta policy for lower-body actions and
control the upper-body parts for aiming primarily by themselves. In
contrast, in Crouch+AimingWalk, we want the cooperative policy
to replace the walking motions from the meta policy with crouching
while keeping the upper-body motion of aiming. Here, as can be
seen in the fourth case of the figure, the cooperative policy exploits
the meta policy to perform aiming actions but performs crouching
mainly on its own. In Fig. 8, we also plot the distribution of weights
based on the collection of 5,000 consecutive frames from the Aim-
ing+Crouch and Crouch+AimingWalk tasks. The statistical results
are consistent with the above studied cases.

As an additional experiment, in Fig. 9, we show that control poli-
cies trained with our approach can support the interactive control
scheme proposed by Xu and Karamouzas [2021]. In this experi-
ment, we let the character perform a variety of locomotion styles

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

by switching the three trained Aiming+Locomotion policies inter-
actively in response to external control signal provided by the user,
and navigate to and aim at the target directions specified by the
user dynamically.

6.6 Ablation Studies
We refer to the previous literature of ICCGAN [Xu and Karamouzas
2021] for ablation studies with respect to each component in the
employed GAN-like structure for motion imitation, and to [Peng
et al. 2021; Xu and Karamouzas 2021] for related analyses on the
robustness of control policies trained using GAN-like structures
combined with reinforcement learning. Here, we focus on the studies
of the proposed multi-objective learning framework and incremental
learning scheme.

In Fig. 10, we compare the performance of our proposed multi-
objective (MO) learning framework to two baselines using three
composite motion learning tasks from Section 4.1. The first baseline
leverages our MO learning framework but does not make use of
PopArt to normalize the value targets of each critic (w/o PopArt).
The second baseline simply adds the rewards from the two discrimi-
nators together and models the composite motion learning task as
a typical reinforcement learning problem (w/o MO). Both baselines
are trained with our motion decoupling scheme described in Sec-
tion 4.1 and simultaneously leverage two discriminators, one for the
upper-body motion and one for the lower body. As can be seen from

010.00.10.20.30.40.00.10.20.3Aiming+Crouch0.00.10.20.30.40.00.10.20.30.00.10.20.30.40.00.10.20.30.00.10.20.30.40.00.10.20.30.70.80.91.00.00.10.20.30.70.80.91.00.00.10.20.30.70.80.91.00.00.10.20.30.70.80.91.0Chest0.00.10.20.3Crouch+AimingWalk0.70.80.91.0Head0.00.10.20.30.70.80.91.0UpperArms0.00.10.20.30.70.80.91.0Forearms0.00.10.20.30.00.10.20.30.4Thighs0.00.10.20.30.00.10.20.30.4Shins0.00.10.20.30.00.10.20.30.4Feet0.00.10.20.3Composite Motion Learning with Task Control

•

13

Fig. 9. Interactive control of switching between walking, crouching and running for location targeting while aiming.

Fig. 10. Learning performance on tasks of composite motion learning from
two single-clip reference motions, which are illustrated in Fig. 4. "MO"
stands for the proposed multi-objective learning framework detailed in
Section 4.3. Colored regions denote mean values ± a standard deviation
based on 10 trials.

the figure, it is hard for "w/o MO" to balance the learning of the two
reference motions. For example, in the ChestOpen+JumpingJack
task, as the upper-body (ChestOpen) imitation error goes down,
the lower-body (JumpingJack) error increases; in the Punch+Run
task, the policy almost gives up on learning how to run, focusing on
punching without too much success. In contrast, when leveraging
our MO framework either with or without PopArt, the imitation
errors of the upper and lower body show similar and stable trends,
keep decreasing as the training goes on. Additionally, the introduc-
tion of PopArt typically facilitates better training, allowing for faster
convergence speed, lower imitation error, and more robust training
achieving similar performance across different trials.

Figure 11 shows the performance of our MO approach with and
without exploiting the proposed incremental learning scheme. We
also provide comparisons with the "w/o MO" baseline. The tested
tasks have four objectives, as described in Section 6.5: two imitation
objectives for the upper and lower body respectively, one Target
Location task for the locomotion and one Aiming task. In the cases

Fig. 11. Learning performance on three composite tasks where each task
combines learning from two partial motions while accomplishing two goal
objectives. Multi-objective learning in an incremental manner leads to
sample-efficient training allowing for high-fidelity composite motion syn-
thesis with goal-directed control. Colored regions denote mean values ±
one standard deviation based on 10 trials.

using incremental learning, we employed a pre-trained, locomo-
tion policy as the meta one. Consistent with the previous ablation
study, we can see that the "w/o MO" baseline struggles to balance
the different objective terms. Here, the character quickly achieves a
high reward for the goal-directed Aiming task (3rd row) but fails
to complete other objectives, and in particular to account for the
motion style provided by the imitation reward terms. For exam-
ple, the controlled character holds the toy pistol in an unnatural

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

0.00.51.01.52.0×1070.20.40.60.81.0NormalizedImitationErrorChestOpen+JumpingJackChestOpen0.00.51.01.52.0×1070.60.81.0JumpingJack+WalkInPlaceJumpingJack0.00.51.01.52.0×1070.40.60.81.0Punch+RunPunch0.00.51.01.52.0#ofSamples×1070.20.40.60.81.0JumpingJack0.00.51.01.52.0#ofSamples×1070.60.81.0WalkInPlace0.00.51.01.52.0#ofSamples×1070.40.60.81.0RunMO+PopArt(Ours)w/oPopArtw/oMO012345×1070.00.20.40.60.81.0NormalizedImitationErrorAiming+CrouchToyPistolHolding012345×1070.00.20.40.60.81.0Aiming+WalkToyPistolHolding012345×1070.00.20.40.60.81.0Aiming+RunToyPistolHolding012345×1070.00.20.40.60.81.0Crouch012345×1070.00.20.40.60.81.0Walk012345×1070.00.20.40.60.81.0Run012345×1070.00.20.40.60.81.0TaskRewardAiming012345×1070.00.20.40.60.81.0Aiming012345×1070.00.20.40.60.81.0Aiming012345#ofSamples×1070.00.20.40.60.81.0Locomotion012345#ofSamples×1070.00.20.40.60.81.0Locomotion012345#ofSamples×1070.00.20.40.60.81.0LocomotionwithIncr.Learningw/oIncr.Learningw/oMO14

• Xu, P. et al

way compared to the demonstrations in the provided reference mo-
tions as indicated by the high imitation error (1st row). While such
issues are successfully resolved by our proposed MO framework,
learning in a non-incremental way leads to sample inefficient train-
ing as compared to learning by leveraging a meta policy. Besides
slow speed of convergence, non-incremental training can be time
consuming for challenging multi-objective tasks. For example, in
the Aiming+Run task, while the case with incremental learning
only needs 1.5 hours to finish the training by using about 20 mil-
lion samples, the non-incremental cases need about 20 more hours
for training and will consume about 300 million more samples to
achieve a similar performance.

7 LIMITATIONS AND FUTURE WORK
We present a technique for training composite-motion controllers
using a multi-objective learning framework that is capable of com-
bining multiple reference examples and task goals to control a
physically-simulated character. We demonstrate that our approach
can generalize to a large number of examples based on the availabil-
ity of reference data. Likewise, we show its ability to accomplish
simultaneous goal-driven tasks such as aiming at specific targets
and moving to a target location with different locomotion styles.
Furthermore, we can interactively control such character’s actions,
pushing the boundary of what is capable for physics-based charac-
ters to date.

Of course, there is still more to explore in this space. Our system
is currently not well-equipped to handle behaviors which include
multiple phases, as the imitation is not phase-locked in any fashion
and our discriminators do not distinguish between different stages of
an activity. Exploring the potential to add a state machine with state
transitions could aid in this capacity [Starke et al. 2019]. Another
shortcoming of the approach presently is that we do not account
for variation across the humans that recorded the motion clips. This
implies that we are introducing bias in the imitation process that
may degrade the final quality of the animation. As is, the system
is able to make adjustments automatically as needed based on the
physical characteristics of the behavior but it cannot distinguish
errors that are more stylistic.

In its current form, our system can not create new composite ac-
tivities without performing additional training. A possible direction
for future work is aimed at sidestepping this limitation to directly
combine preexisting policies and greatly improve the scalability
of trained controllers. That is, to train two (or more) policies in-
dependently and combine them at runtime to create a composite
motion. Finally, in human motion, composite behaviors go beyond
an anticipated split, e.g. the lower and upper body, which is one of
the modest underlying assumptions in our current implementation.
Instead, humans may enlist body parts and release them fluidly. For
example, a well-trained martial artist changes the use of appendages
quickly in fighting sequences. We wish to explore this direction in
future investigations and believe that our proposed multi-objective
learning framework can provide the foundation for such future
endeavors.

Although we employed an upper and lower body split in most
of our experiments, there is nothing tied to this body decoupling

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

Fig. 12. Failure case study. Top: The character’s body is bisected into a left
and right group, imitating walking and jumping respectively. Bottom: Juggle
while running.

scheme except that it is a practical general choice for deploying
the limbs of the whole body. Currently, as long as the subtasks are
compatible, our system is capable of combining motions along other
body splits. For instance, in the Juggling+TargetLocation example
discussed in Section 6.4, the trained policy controls the arms for
juggling and the rest of the body for walking. Our approach may fail
if, for example, the lower limbs are separated due to the requirements
of physical balance. As an example, in Fig. 12, we show a failure
case where the body is bisected into a left/right split and asked to
imitate walking and jumping motions respectively. Such a composite
motion is not well-defined, even for humans. We can see that though
not falling down, the simulated character cannot imitate the two
motions accurately, and instead performs an in-between motion
where the character neither jumps up nor walks in an expected
fashion.

In Fig. 12, we also show another failure case where running refer-
ence motions with an average speed of around 3.5?/? are provided
for the Juggling+TargetLocotion task. With the difficulty of juggling
while moving at this higher speed, this example is significantly more
challenging than the one shown in Fig. 5d. Even though we are able
to synthesize the composite motions, the simulated character cannot
juggle the balls successfully under these conditions. Currently, our
approach cannot identify if a composite motion is compatible on its
own, and instead, it relies on a human to combine behaviors with
some domain knowledge about the affinity of the mixing and the
feasibility of associated goal-directed tasks. Automating this would
be a great direction for future work.

ACKNOWLEDGMENTS
This work was supported by the National Science Foundation un-
der Grant No. IIS-2047632 and by Roblox. We would like to thank
Rokoko 2 for providing mocap data for this project.

2https://www.rokoko.com

REFERENCES
Yeuhi Abe, Marco Da Silva, and Jovan Popovi?. 2007. Multiobjective control with fric-
tional contacts. In ACM SIGGRAPH/Eurographics Symposium on Computer Animation.
249–258.

Eduardo Alvarado, Damien Rohmer, and Marie-Paule Cani. 2022. Generating Upper-
Body Motion for Real-Time Characters Making their Way through Dynamic Envi-
ronments. Computer Graphics Forum 41, 8 (2022).

Kevin Bergamin, Simon Clavet, Daniel Holden, and James Richard Forbes. 2019. DReCon:
data-driven responsive control of physics-based characters. ACM Transactions On
Graphics 38, 6 (2019), 1–11.

Nuttapong Chentanez, Matthias Müller, Miles Macklin, Viktor Makoviychuk, and Stefan
Jeschke. 2018. Physics-Based motion capture imitation with deep reinforcement
learning. In ACM SIGGRAPH Conference on Motion, Interaction and Games.

Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. 2014. Empir-
ical evaluation of gated recurrent neural networks on sequence modeling. arXiv
preprint arXiv:1412.3555 (2014).

Alexander Clegg, Wenhao Yu, Jie Tan, C Karen Liu, and Greg Turk. 2018. Learning to
dress: Synthesizing human dressing motion via deep reinforcement learning. ACM
Transactions on Graphics (TOG) 37, 6 (2018), 1–10.

Stelian Coros, Philippe Beaudoin, and Michiel van de Panne. 2010. Generalized biped

walking control. ACM Transactions on Graphics 29, 4 (2010), 130.

Danilo Borges da Silva, Rubens Fernandes Nunes, Creto Augusto Vidal, Joaquim B
Cavalcante-Neto, Paul G Kry, and Victor B Zordan. 2017. Tunable robustness:
An artificial contact strategy with virtual actuator control for balance. Computer
Graphics Forum 36, 8 (2017), 499–510.

Marco Da Silva, Yeuhi Abe, and Jovan Popovi?. 2008. Simulation of human motion
data using short-horizon model-predictive control. Computer Graphics Forum 27, 2
(2008), 371–380.

Martin De Lasa and Aaron Hertzmann. 2009. Prioritized optimization for task-space
control. In IEEE/RSJ International Conference on Intelligent Robots and Systems. 5755–
5762.

Martin De Lasa, Igor Mordatch, and Aaron Hertzmann. 2010. Feature-based locomotion

controllers. ACM Transactions on Graphics 29, 4 (2010), 1–10.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron C
Courville. 2017. Improved training of wasserstein gans. Advances in Neural Infor-
mation Processing Systems 30 (2017).

Perttu Hämäläinen, Joose Rajamäki, and C Karen Liu. 2015. Online control of simulated
humanoids using particle belief propagation. ACM Transactions on Graphics 34, 4
(2015), 1–13.

Tatsuya Harada, Sou Taoka, Taketoshi Mori, and Tomomasa Sato. 2004. Quantitative
evaluation method for pose and motion similarity based on human perception. In
IEEE/RAS International Conference on Humanoid Robots, Vol. 1. 494–512.

Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. 2020. Robust

motion in-betweening. ACM Transactions on Graphics 39, 4 (2020).

Jonathan Ho and Stefano Ermon. 2016. Generative adversarial imitation learning.

Advances in Neural Information Processing Systems 29 (2016).

Deok-Kyeong Jang, Soomin Park, and Sung-Hee Lee. 2022. Motion Puzzle: Arbitrary
Motion Style Transfer by Body Part. ACM Transactions on Graphics 41, 3 (2022).
Won-Seob Jang, Won-Kyu Lee, In-Kwon Lee, and Jehee Lee. 2008. Enriching a motion
database by analogous combination of partial human motions. The Visual Computer
24, 4 (2008), 271–280.

Andrej Karpathy and Michiel Van De Panne. 2012. Curriculum learning for motor skills.
In Canadian Conference on Advances in Artificial Intelligence. Springer, 325–330.
Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization.

arXiv preprint arXiv:1412.6980 (2014).

Taesoo Kwon and Jessica K Hodgins. 2010. Control systems for human running using an
inverted pendulum model and a reference motion capture sequence. In Proceedings
of the 2010 ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 129–
138.

Taesoo Kwon and Jessica K Hodgins. 2017. Momentum-mapped inverted pendulum
models for controlling dynamic human motions. ACM Transactions on Graphics 36,
1 (2017), 1–14.

Seunghwan Lee, Phil Sik Chang, and Jehee Lee. 2022. Deep Compliant Control. In ACM
SIGGRAPH 2022 Conference Proceedings. Association for Computing Machinery.
Seyoung Lee, Sunmin Lee, Yongwoo Lee, and Jehee Lee. 2021. Learning a family of
motor skills from a single motion clip. ACM Transactions on Graphics 40, 4 (2021),
1–13.

Seunghwan Lee, Moonseok Park, Kyoungmin Lee, and Jehee Lee. 2019. Scalable muscle-
actuated human simulation and control. ACM Transactions on Graphics 38, 4 (2019),
1–13.

Yoonsang Lee, Sungeun Kim, and Jehee Lee. 2010. Data-driven biped control. ACM

Transactions on Graphics 29, 4 (2010), 129.

Jae Hyun Lim and Jong Chul Ye. 2017. Geometric GAN. arXiv preprint arXiv:1705.02894

(2017).

Hung Yu Ling, Fabio Zinno, George Cheng, and Michiel Van De Panne. 2020. Character
controllers using motion vaes. ACM Transactions on Graphics 39, 4 (2020), 40–1.

Composite Motion Learning with Task Control

•

15

Libin Liu and Jessica Hodgins. 2018. Learning basketball dribbling skills using trajectory
optimization and deep reinforcement learning. ACM Transactions on Graphics 37, 4
(2018), 1–14.

Libin Liu, Michiel van de Panne, and KangKang Yin. 2016. Guided learning of control
graphs for physics-based characters. ACM Transactions on Graphics 35, 3 (2016),
1–14.

Libin Liu, KangKang Yin, and Baining Guo. 2015. Improving sampling-based motion

control. Computer Graphics Forum 34, 2 (2015), 415–423.

Libin Liu, KangKang Yin, Michiel van de Panne, and Baining Guo. 2012. Terrain runner:
control, parameterization, composition, and planning for highly dynamic motions.
ACM Transactions on Graphics 31, 6 (2012), 154–1.

Libin Liu, KangKang Yin, Michiel van de Panne, Tianjia Shao, and Weiwei Xu. 2010.
Sampling-based contact-rich motion control. In ACM SIGGRAPH 2010 papers. 1–10.
Adriano Macchietto, Victor Zordan, and Christian R Shelton. 2009. Momentum control

for balance. In ACM SIGGRAPH 2009 papers. 1–8.

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles
Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, et al. 2021.
Isaac Gym: High performance GPU-based physics simulation for robot learning.
arXiv preprint arXiv:2108.10470 (2021).

Josh Merel, Leonard Hasenclever, Alexandre Galashov, Arun Ahuja, Vu Pham, Greg
Wayne, Yee Whye Teh, and Nicolas Heess. 2019. Neural Probabilistic Motor Primi-
tives for Humanoid Control. In International Conference on Learning Representations.
Josh Merel, Yuval Tassa, Dhruva TB, Sriram Srinivasan, Jay Lemmon, Ziyu Wang, Greg
Wayne, and Nicolas Heess. 2017. Learning human behaviors from motion capture
by adversarial imitation. arXiv preprint arXiv:1707.02201 (2017).

Josh Merel, Saran Tunyasuvunakool, Arun Ahuja, Yuval Tassa, Leonard Hasenclever,
Vu Pham, Tom Erez, Greg Wayne, and Nicolas Heess. 2020. Catch & Carry: reusable
neural controllers for vision-guided whole-body tasks. ACM Transactions on Graphics
39, 4 (2020), 39–1.

Igor Mordatch and Emo Todorov. 2014. Combining the benefits of function approxima-

tion and trajectory optimization. In Robotics: Science and Systems, Vol. 4.

Igor Mordatch, Emanuel Todorov, and Zoran Popovi?. 2012. Discovery of complex
behaviors through contact-invariant optimization. ACM Transactions on Graphics
31, 4 (2012), 1–8.

Uldarico Muico, Yongjoon Lee, Jovan Popovi?, and Zoran Popovi?. 2009. Contact-aware
nonlinear control of dynamic characters. In ACM SIGGRAPH 2009 papers. 1–9.
Ofir Nachum, Michael Ahn, Hugo Ponte, Shixiang Gu, and Vikash Kumar. 2019. Multi-
agent manipulation via locomotion using hierarchical sim2real. arXiv preprint
arXiv:1908.05224 (2019).

Soohwan Park, Hoseok Ryu, Seyoung Lee, Sunmin Lee, and Jehee Lee. 2019. Learn-
ing predict-and-simulate policies from unorganized human motion data. ACM
Transactions on Graphics 38, 6 (2019), 1–11.

Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne. 2018. Deepmimic:
Example-guided deep reinforcement learning of physics-based character skills. ACM
Transactions on Graphics 37, 4 (2018), 1–14.

Xue Bin Peng, Glen Berseth, KangKang Yin, and Michiel van de Panne. 2017. Deeploco:
Dynamic locomotion skills using hierarchical deep reinforcement learning. ACM
Transactions on Graphics 36, 4 (2017), 1–13.

Xue Bin Peng, Michael Chang, Grace Zhang, Pieter Abbeel, and Sergey Levine. 2019.
MCP: Learning Composable Hierarchical Control with Multiplicative Compositional
Policies. Advances in Neural Information Processing Systems 32 (2019), 3686–3697.
Xue Bin Peng, Yunrong Guo, Lina Halper, Sergey Levine, and Sanja Fidler. 2022. ASE:
Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Char-
acters. ACM Transactions on Graphics 41, 4 (2022).

Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, and Angjoo Kanazawa. 2021.
AMP: Adversarial motion priors for stylized physics-based character control. ACM
Transactions on Graphics 40, 4 (2021).

Avinash Ranganath, Pei Xu, Ioannis Karamouzas, and Victor Zordan. 2019. Low dimen-
sional motor skill learning using coactivation. In ACM SIGGRAPH Conference on
Motion, Interaction and Games. 1–10.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017.
Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).
Asako Soga, Yuho Yazaki, Bin Umino, and Motoko Hirayama. 2016. Body-part motion
synthesis system for contemporary dance creation. In ACM SIGGRAPH 2016 Posters.
1–2.

Kwang Won Sok, Manmyung Kim, and Jehee Lee. 2007. Simulating biped behaviors

from human motion data. In ACM SIGGRAPH 2007 papers. 107–es.

Sebastian Starke, He Zhang, Taku Komura, and Jun Saito. 2019. Neural State Machine
for Character-Scene Interactions. ACM Transactions on Graphics 38, 6, Article 209
(2019).

Sebastian Starke, Yiwei Zhao, Fabio Zinno, and Taku Komura. 2021. Neural animation
layering for synthesizing martial arts movements. ACM Transactions on Graphics
40, 4 (2021), 1–16.

Jie Tan, Karen Liu, and Greg Turk. 2011. Stable proportional-derivative controllers.

IEEE Computer Graphics and Applications 31, 4 (2011), 34–44.

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

Jeff KT Tang, Howard Leung, Taku Komura, and Hubert PH Shum. 2008. Emulating
human perception of motion similarity. Computer Animation and Virtual Worlds 19,
3-4 (2008), 211–221.

Yuval Tassa, Tom Erez, and Emanuel Todorov. 2012. Synthesis and stabilization of
complex behaviors through online trajectory optimization. In IEEE/RSJ International
Conference on Intelligent Robots and Systems. 4906–4913.

Yuval Tassa, Nicolas Mansard, and Emo Todorov. 2014. Control-limited differential
dynamic programming. In IEEE International Conference on Robotics and Automation.
1168–1175.

Hado P van Hasselt, Arthur Guez, Matteo Hessel, Volodymyr Mnih, and David Silver.
2016. Learning values across many orders of magnitude. Advances in Neural
Information Processing Systems 29 (2016).

Kevin Wampler, Zoran Popovi?, and Jovan Popovi?. 2014. Generalizing locomotion
style to new animals with inverse optimal regression. ACM Transactions on Graphics
33, 4 (2014), 1–11.

Tingwu Wang, Yunrong Guo, Maria Shugrina, and Sanja Fidler. 2020. Unicon: Universal
neural controller for physics-based character motion. arXiv preprint arXiv:2011.15119
(2020).

Jungdam Won, Deepak Gopinath, and Jessica Hodgins. 2020. A scalable approach to
control diverse behaviors for physically simulated characters. ACM Transactions on
Graphics 39, 4 (2020), 33–1.

Jungdam Won, Deepak Gopinath, and Jessica Hodgins. 2021. Control strategies for
physically simulated characters performing two-player competitive sports. ACM
Transactions on Graphics 40, 4 (2021), 1–11.

Jungdam Won, Deepak Gopinath, and Jessica Hodgins. 2022. Physics-based character
controllers using conditional VAEs. ACM Transactions on Graphics 41, 4 (2022),
1–12.

Jungdam Won, Jungnam Park, and Jehee Lee. 2018. Aerobatics control of flying creatures
via self-regulated learning. ACM Transactions on Graphics 37, 6 (2018), 1–10.
Chun-Chih Wu and Victor Zordan. 2010. Goal-directed stepping with momentum
control. In ACM SIGGRAPH/Eurographics Symposium on Computer Animation. 113–
118.

Zhaoming Xie, Hung Yu Ling, Nam Hee Kim, and Michiel van de Panne. 2020. ALL-
STEPS: Curriculum-Driven Learning of Stepping Stone Skills. Computer Graphics
Forum 39 (2020), 213–224.

Pei Xu and Ioannis Karamouzas. 2021. A GAN-Like Approach for Physics-Based
Imitation Learning and Interactive Character Control. Proceedings of the ACM on
Computer Graphics and Interactive Techniques 4, 3 (2021).

Zeshi Yang and Zhiqi Yin. 2021. Efficient hyperparameter optimization for physics-
based character animation. Proceedings of the ACM on Computer Graphics and
Interactive Techniques 4, 1 (2021), 1–19.

Yuho Yazaki, Asako Soga, Bin Umino, and Motoko Hirayama. 2015. Automatic compo-
sition by body-part motion synthesis for supporting dance creation. In International
Conference on Cyberworlds. IEEE, 200–203.

Yuting Ye and C Karen Liu. 2010a. Optimal feedback control for character animation

using an abstract model. In ACM SIGGRAPH 2010 papers. 1–9.

Yuting Ye and C Karen Liu. 2010b. Synthesis of responsive motion using a dynamic

model. In Computer Graphics Forum, Vol. 29. 555–562.

KangKang Yin, Kevin Loken, and Michiel van de Panne. 2007. Simbicon: Simple biped

locomotion control. ACM Transactions on Graphics 26, 3 (2007), 105–es.

Chao Yu, Akash Velu, Eugene Vinitsky, Yu Wang, Alexandre Bayen, and Yi Wu. 2021.
The surprising effectiveness of PPO in cooperative, multi-agent games. arXiv preprint
arXiv:2103.01955 (2021).

Wenhao Yu, Greg Turk, and C Karen Liu. 2018. Learning symmetric and low-energy

locomotion. ACM Transactions on Graphics 37, 4 (2018), 1–12.

Victor Zordan, David Brown, Adriano Macchietto, and KangKang Yin. 2014. Con-
trol of rotational dynamics for ground and aerial behavior. IEEE Transactions on
Visualization and Computer Graphics 20, 10 (2014), 1356–1366.

Victor Zordan and Jessica K Hodgins. 2002. Motion capture-driven simulations that
hit and react. In ACM SIGGRAPH/Eurographics Symposium on Computer Animation.
89–96.

A STATE AND ACTION REPRESENTATION
Given the definition in Eq. 4, we have the character state vector
s? ? R(?+1)×?link×13, which includes all body links’ positions, orien-
tations, and linear and angular velocities of the simulated character
in the last ? + 1 frame from ? ?? to ?. To ignore the global coordinate,
we assume that the ground height is 0, and all the body links’ states
are localized based on the position and heading direction of the
character’s root link (pelvis) at the last frame ?. Similarly, if a goal
state g? is provided, we localize the position and direction state in g?
using the same coordinate system with s? . During our experiments,

Composite Motion Learning with Task Control

•

1

if multiple goal-directed tasks are involved, we simply concatenate
goal states from all the tasks together as the representation of g? .
We refer to the Appendix in the supplementary material for the
representation of g? in our designed goal-directed tasks.

The action a? is a set of target postures fed into the PD servo.
Therefore, we have a? ? R?dof where ?dof is the total degrees of
freedom (DoF) in the character model. a? is assumed to be nor-
malized by the valid movement range of each DoF but without
upper and lower bounds applied. The observation space o?
? for dis-
criminators is similar to s? . However, we keep only body links’
positions and orientations, and the discriminators rely on the pose
trajectory of o? to ensure that the visual velocities between two
frames are consistent with the reference motions. As such, we have
? ? R(?? +2)×? ?
o?
link×7 where ?? + 2 is the number of observed frames
as defined in Eq. 3. o?
? is localized depending on its characteristics.
For lower body parts, their motions often involve the character’s
spatial movement. Therefore, we follow the definition of s? , and
use a local coordinate defined by the root pose at the last observed
frame. For upper-body motions, however, we typically care more
about the body parts’ local poses related to a specific parent body
link. Therefore, we use a framewisely defined local system based
on the parent link’s pose such that the global-space displacement
and rotation controlled by the lower body are ignored. In our imple-
mentation, for upper-body motions, we choose pelvis as the parent
link; and for arm only motions, we choose torso as the parent.
The observation sampled from the reference motions, i.e. ˜o?

? , is
defined the same as o?
? . However, instead of performing sampling at
a fixed frame rate identical to the control policy’s working frequency
(30Hz in Fig. 2), we do sampling with dynamic interval ?? = ??
where ? = 1/30? is the time interval between two frames during
simulation and ? ? Uniform(0.8, 1.2). In such a way, we scale the
reference motion temporally within a small range, for better com-
bining motions from multiple reference sources with inconsistent
pace. To keep the motion stable, ?? differs among multiple times of
sampling but is identical for the ?? + 2 frames of one sample.

B TASK ENVIRONMENT SETUP
B.1 Task: Target Heading
The goal-directed reward is defined as

?? = ? (cid:164)x

root
? +1 /|| (cid:164)x? ||, g? ?,

(12)

where (cid:164)xroot
is the horizontal displacement of the character’s root
? +1
link from the frame ? to ? + 1. The goal state g? ? R2 is a unit
vector representing the target heading direction, which is randomly
sampled every 30 frames (1?).

B.2 Task: Target Location
The goal-directed reward is defined as

(cid:40)

?? =

exp(?3|| (cid:164)xroot
1

? +1 /? ? v?

? ||2/||v?

? ||2)

if ||x? +1 ? pgoal|| > ?
otherwise,

(13)
where ? = 0.5 is the goal radius of the target location, ? = 1/30?
is the time interval between two frames, (cid:164)xroot
? +1 /? denotes the hori-
zontal velocity of the character’s root link from the frame ? to ? + 1,

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

2

• Xu, P. et al

? is the target velocity with a preferred speed and a direction

and v?
toward the target goal location.

?

The goal state g? ? R4 includes a 2D unit vector representing
the direction to the target location, the horizontal distance from
the character to the goal, i.e. ||xroot
? pgoal||, and the preferred
speed ||v?
? ||. The preferred speed is sampled from [1, 1.5] in the
unit of ?/? for crouching and walking motions, and from [1, 3]
for running. The goal direction is sampled from [0, 2?). A timer
variable is sampled from [3, 5] in the unit of ? for crouching and
walking motions, and from [2, 3] for running. We use these three
goal variables to obtain the target location. As such, we can perform
speed control during the location targeting.

B.3 Task: Aiming
The goal-directed reward is defined as

(cid:40)

?? =

exp(?2||dforearm
Clip(?dforearm
?

?

? g? ||2)

if aiming is activated

, uref?, 0, 0.8)/0.8 otherwise

(14)
where dforearm
? R3 is a unit vector representing the direction of the
?
right forearm from the elbow to the hand, and uref is a unit vector
representing the up axis of the world space. In our implementation,
the toy pistol is fixed on the right hand, which is linked to the right
forearm with a fixed joint. Therefore, we use the direction of the
right forearm as the aiming direction. When the aiming action is not
activated, we use the 2nd reward term to encourage the character
to lift its arm and hold the gun up without aiming anything.

The goal state g? ? R3 is a unit vector representing the target
aiming direction. We let g? = 0 if the aiming action is not activated.
When combined with the target location task, aiming is deactivated
if the character is close to the goal, i.e. ||x? ? pgoal|| ? ?. g? is
sampled with an elevation angle in range of [0, ?/6] and azimuth
angle in [0, ?/4].

B.4 Task: Tennis Swing
The goal-directed reward is defined as

1.2 + ||vout||/10
? pose
?
? pose
?

+ 0.5 exp(?0.1?2

fall)

if ball was hit and ?fall = 0
if ball was hit but ?fall > 0
otherwise

(15)

?? =

?????
???
?

where

,

(16)

||2).

ball
? p
?

ball
? p
?

|| ? 1, 0)2),

+ 0.5? racket
?

= 0.2? shoulder
?

racket
= exp(?5||p
?

shoulder
= exp(? max(||p
?

is position of the character’s right shoulder and pracket

? pose
?
? shoulder
?
? racket
?
pshoulder
is
?
the position of the racket. To emulate the tennis court, we consider
a valid ball falling region with dimension 12? × 11?, which is 6?
ahead of the initial position of the tennis ball along the x-axis. ?fall is
the distance from the ball’s falling point to this region. We let ?fall =
0 if the ball will fall or fell in the target region. ?fall is estimated by
a simple projectile model based on the linear velocity of the ball
without considering any friction or air resistance, but updated at
every simulation step in order to get an accurate estimation. ||vout||

?

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.

?

is the outgoing speed of the tennis ball when it was hit. The purpose
of using ? shoulder
is to encourage the character to approach the
tennis ball but not necessarily when the distance is less than 1?
such that the character can have enough space to swing the racket,
rather than keeping moving close to the ball.

The goal state g? ? R4 includes a 3D vector representing the po-
sition of the ball pball
, and a scalar identifying the heading direction
of the character’s root link. The heading direction in g? is used to
identify the direction of x-axis toward which the ball is expected to
be hit. We let pball

= 0 when constructing g? if the ball was hit.

?

?

Juggling

B.5
The goal-directed reward is defined as

?? = 0.5? hand,left
?
and ? hand,right
?

+ 0.5? hand,right
?

(17)

where ? hand,left
are defined identically but evaluate
?
the performance of the left hand and right hand respectively. For
each hand-related reward, we define

? hand
?

=

(cid:40)? throw
?
0.1? height
?

+ 0.9? distance
?
where ? is the time interval between two trials of ball throwing and

if ? mod ? = 0
otherwise

(18)

? throw
?
? height
?
? distance
?

= exp(?5(? ball
?
= exp(?20(?ball ? ?hand
= 0.9 exp(?20?2

/? ball ? 1)2),
)2),
? ) + 0.2 exp(??2

?

? ).

(19)

As stated in Section 6.4, we employ an automatic catch-and-throw
mechanism where a ball is considered caught by a hand and is fixed
to that hand if it is close enough, and will be detached (thrown)
automatically at a fixed time interval ? between two trials of throw-
ing. The target ball for a hand is decided using a cascade juggling
pattern. In the reward function, ? throw
measures the performance
?
of ball throwing and is computed only at the frame where a ball
is thrown. ? ball
is the vertical velocity of the thrown ball and ? ball
is the preferred vertical thrown velocity. The preferred velocity is
obtained by assuming that the thrown ball will be caught at the
same height where it is thrown and at a dwell time ?? before the
next time the catching hand performs a thrown. In our experiment,
we set ? = 2/3? (20 frames) with a preferred dwell time ?d = 0.4?
(12 frames) and set the number of balls ?ball = 3. Given the gravity
? = 9.81?/?2, this leads to a preferred velocity

?

? ball = 0.5?(

?

2

?ball ? ?d) = 2.94?/?.

(20)

?

The height-related reward term ? height
measures the error between
?
the hand’s vertical position (?hand
) and the target ball’s height when
it was thrown (?ball). It encourages the control policy to throw and
catch a ball at the same height. We let ? height
= 1 if the target
ball was caught by the hand already. The distance-related reward
? distance
measures the distance error between the hand and the
?
target ball. We estimate the ball’s vertical movement trajectory
using a simple projectile model taking into account only the ball’s
vertical linear velocity and gravity. The distance ?? is defined as the
distance between the hand and the target ball if the hand is above the

?

Table 3. Hyperparameters

Parameter

policy network learning rate
critic network learning rate
discriminator learning rate
reward discount factor (?)
GAE discount factor (?)
surrogate clip range (?)
gradient penalty coefficient (??? )
number of PPO workers (simulation instances)
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
512
4096
256
5
8192
512

estimated trajectory, i.e. when the hand is unable to catch the ball at
the current hand height, or just the horizontal distance otherwise.
As such, ? distance
ignores the vertical ball-hand distance if the hand
is able to catch the ball at its current height, and thus prevents the
hand from aggressively moving toward the ball vertically.

?

Composite Motion Learning with Task Control

•

3

The goal state g? ? R19 includes the three balls’ states (position
and linear velocity) and a timer variable counting the time left before
the next throwing of the ball by one hand. The ball states are in the
order of the left-hand target ball, the right-hand target ball, and the
other ball. For a caught target ball, we let the corresponding state
be zero.

C HYPERPARAMETERS
The hyperparameters used for policy training is listed in Table 3.
Half of the samples for discriminator training are from the simu-
lated character and half are sampled from the reference motions.
The character state horizon ? + 1 is chosen as 4, and the discrim-
inator observation horizon ?? + 2 is 3 for aiming motions and 5
for other motions. The objective weight ?? in Eq. 9 is 0.5 shared
equally by all goal-related objectives. In the Juggling with Target
Location task, given the difficulty of ball catching, the juggling task
is assigned a weight of 0.6, the locomotion task has a weight of
0.1, and the imitation tasks account for the remaining weight with
a ratio of 1 : 4 for juggling and walking motion imitation. In the
Aiming+Locomotion task, the upper-body motion of aiming has a
weight of 0.2 and the lower-body motion has a weight of 0.3. On
the other tests, besides the weights taken by the goal-related ob-
jectives, the remaining weight is shared equally by the imitation
objectives.

ACM Trans. Graph., Vol. 42, No. 4, Article . Publication date: August 2023.


