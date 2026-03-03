# Composite Motion Learning with Task Control

This is the official implementation for

- _*Composite Motion Learning with Task Control*_
[[arXiv](https://arxiv.org/abs/2305.03286)]
[[Youtube](https://youtu.be/mcRAxwoTh3E)]
([SIGGRAPH'23](https://s2023.siggraph.org/presentation/?id=papers_763&sess=sess118), [TOG](https://dl.acm.org/doi/abs/10.1145/3592447))

- _*A GAN-Like Approach for Physics-Based Imitation Learning and Interactive Character Control*_
[[arXiv](https://arxiv.org/abs/2105.10066)]
[[Youtube](https://www.youtube.com/watch?v=VHMyvDD3B_o)]
([SCA'21](https://www.youtube.com/watch?v=vPzpCarkm74), [PACMCGIT](https://dl.acm.org/doi/abs/10.1145/3480148))

We also refer to the extended implementation

- _*AdaptNet: Policy Adaptation for Physics-Based Character Control*_
[[arXiv](http://arxiv.org/abs/2310.00239)]
[[Youtube](https://youtu.be/WxmJSCNFb28)]
[[webpage](https://pei-xu.github.io/AdaptNet)]
[[code](https://github.com/xupei0610/AdaptNet)]
([SIGGRAPH Asia'23](https://asia.siggraph.org/2023/presentation/?id=papers_543&sess=sess120), [TOG](https://dl.acm.org/doi/10.1145/3618375))

- _*Synchronize Dual Hands for Physics-Based Dexterous Guitar Playing*_
[[arXiv](https://arxiv.org/abs/2409.16629)]
[[Youtube](https://www.youtube.com/watch?v=r_y0P2pIeF8&list=PLLfEynalFz6j0X5Kiut0U3GLRxt3Oz_oa)]
[[webpage](https://pei-xu.github.io/guitar)]
[[code](https://github.com/xupei0610/guitar)]
([SIGGRAPH Asia'24](https://asia.siggraph.org/2024/presentation/?id=papers_1155&sess=sess150))

- _*FürElise: Capturing and Physically Synthesizing Hand Motion of Piano Performance*_
[[arXiv](https://arxiv.org/abs/2410.05791)]
[[webpage](https://for-elise.github.io/)]
([SIGGRAPH Asia'24](https://asia.siggraph.org/2024/presentation/?id=papers_1250&sess=sess129))


![](doc/teaser_tennis.png)

![](doc/teaser_juggling.png)

![](doc/teaser_aiming.png)

![](doc/teaser_fight.png)

## Code Usage - Overview

This conversion replaces IsaacGym with MuJoCo for physics simulation while maintaining the same reinforcement learning algorithms (PPO) and model architectures. The key changes include:

1. **Physics Engine**: Replaced IsaacGym with MuJoCo
2. **Environment Interface**: Uses Gymnasium API instead of IsaacGym's custom API
3. **Simulation**: CPU-based simulation (MuJoCo) instead of GPU-based (IsaacGym)
4. **Rendering**: Uses MuJoCo's built-in renderer

### Dependencies
- Pytorch 1.12
- Mujuco

We recommend to install all the requirements through Conda by

    $ conda create --name <env> --file requirements.txt -c pytorch -c conda-forge

Download Mujuco from the [official site](https://mujoco.org/) and install it via pip.

### Policy Training

    $ python main.py <configure_file> --ckpt <checkpoint_dir>

We provide our configure files in `config` folder for reference. To reproduce the examples shown in the paper, 

e.g. `diff walks, please run the training by (will create sub folder automatically based on config file name)

    $ python main.py config/iccgan/jaunty_walk.py --ckpt checkpoints    # Standalone Run
    $ python main.py config/iccgan/limp_walk.py --ckpt checkpoints      # Phase Input - looped motion improvement
    $ python main.py config/iccgan/joyful_walk.py --ckpt checkpoints    # Symmetry Loss - mirror clip mimicability

e.g. `Juggling+Walk`, please run the training by

    $ python main.py config/juggling+locomotion_walk.py --ckpt checkpoints/juggling+locomotion_walk

The training results (model and log) will be generated in the `current_folder/checkpoints/juggling+locomotion_walk` folder.

The training can be done on a single GPU. Use `--device` option to specify the device used for training (default: 0). All our results were obtained using machines equipped with Nvidia V100 or A100 GPU. 

After Trainning, you can view the run's logs made by tensorboard summary writter via cmd like:

    $ tensorboard --logdir=checkpoints/jaunty_walk


### Policy Evaluation

    $ python main.py <configure_file> --ckpt <checkpoint_dir> --test --render [optional]

- by default: without `--render` => render_mode = None
- if `--render` set *const* (or) `--render non`=> render_mode = Non-Interactive rgb_array simulation using cv2
- `--render int` => render_mode = Interactive mode with humanoid realtime simulation

We provide pretrained policy models in `pretrained` folder. To evaluate a pretrained policy, 

e.g. `jaunty_walk`, please run

    $ python main.py config/iccgan/jaunty_walk.py --ckpt pretrained/iccgan/jaunty_walk --test --render int
    
e.g. `Juggling+Walk`, please run

    $ python main.py config/juggling+locomotion_walk.py --ckpt pretrained/juggling+locomotion_walk --test --render


### Understanding the Scores in the Code

These scores are calculated using a **Hinge Loss** variation with a **Gradient Penalty (GP)** to ensure training stability:

* **`score_real` (Expert/Reference Motion):** This is the discriminator's output when viewing the expert motion capture data (`ref`). In the code, `loss_r = relu(1 - score_r).mean()`. A "good" `score_real` is typically positive and high (approaching or exceeding 1.0), meaning the discriminator correctly identifies the expert data as highly realistic.


* **`score_fake` (Generated/Policy Motion):** This is the score for the motion produced by your learning policy (`ob`). The code uses `loss_f = relu(1 + score_f).mean()`. A "good" `score_fake` for the discriminator is a very low/negative value (approaching -1.0), while for the **Generator (the policy)**, a "good" score is one that is high/positive, successfully "fooling" the discriminator into thinking the fake motion is real.

According to the research framework, good progress is marked by the following trends across epochs:

* **Convergence toward Zero (Equilibrium):** In a stable GAN training loop, the discriminator and policy reach a point where the discriminator can no longer easily distinguish between them. You should see `score_real` and `score_fake` begin to oscillate around a stable range rather than one side completely dominating.


* **Rising `score_fake` over Epochs:** Early in training, `score_fake` will be very low (negative) as the character's movement is erratic. As learning progresses, `score_fake` should increase toward the level of `score_real`, indicating the policy is producing more life-like, "expert-quality" motions.


* **Consistency across Decoupled Discriminators:** Since the paper uses **multiple discriminators** for different body parts (e.g., upper-body vs. lower-body), good progress is indicated when both sets of scores (`score_real/upper` and `score_real/lower`) show similar stability. If one discriminator has a near-zero loss while the other is very high, it suggests the model is failing to balance the composite motion.

* **Discriminator Collapse:** If `score_real` stays near 1.0 and `score_fake` stays near -1.0 indefinitely, the discriminator is too strong, and the policy is not learning anything new (vanishing gradients).


* **Policy Collapse (Mode Collapse):** If `score_fake` suddenly spikes to a very high value but the character's **Lifetime** or **Task Reward** (logged in your code) drops, the policy may have found a "cheat" to fool the discriminator without actually performing the task.

**Summary Table of Progress Indicators:**

| Metric | Start of Training | Good Progress | Bad Progress (Failure) |
| --- | --- | --- | --- |
| **`score_real`** | High (~1.0) | Stable High (>0) | Drops to 0 (Discriminator is confused) |
| **`score_fake`** | Very Low (<-1.0) | Rising toward `score_real` | Stays at -1.0 (Policy not learning) |
| **Lifetime** | Low | Increasing/Maximizing | Decreasing/Stagnant |

## Motion Data Copyright
We provide our motion data in `assets/motions`. 

The data labeled with `lafan1` are extracted from [Ubisoft LAFAN1 dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
The juggling motion is extracted from the demo provided by [FreeMoCap Project](https://github.com/freemocap/freemocap).
We cannot provide the tennis motions shown in the paper due to the commercial license.

## Citation

If you use the code or provided motions for your work, please consider citing our papers:

    @article{composite,
        author = {Xu, Pei and Shang, Xiumin and Zordan, Victor and Karamouzas, Ioannis},
        title = {Composite Motion Learning with Task Control},
        journal = {ACM Transactions on Graphics},
        publisher = {ACM New York, NY, USA},
        year = {2023},
        volume = {42},
        number = {4},
        doi = {10.1145/3592447},
        keywords = {physics-based control, character animation, motion synthesis, reinforcement learning, multi-objective learning, incremental learning, GAN}
    }

    @article{iccgan,
        author = {Xu, Pei and Karamouzas, Ioannis},
        title = {A GAN-Like Approach for Physics-Based Imitation Learning and Interactive Character Control},
        journal = {Proceedings of the ACM on Computer Graphics and Interactive Techniques},
        publisher = {ACM New York, NY, USA},
        year = {2021},
        volume = {4},
        number = {3},
        pages = {1--22},
        doi = {10.1145/3480148},
        keywords = {physics-based control, character animation, reinforcement learning, GAN}
    }
