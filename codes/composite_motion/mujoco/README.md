# CompositeMotion - MuJoCo Version

This is a MuJoCo-compatible version of the CompositeMotion project, converted from the original IsaacGym implementation.

## Overview

This conversion replaces IsaacGym with MuJoCo for physics simulation while maintaining the same reinforcement learning algorithms (PPO) and model architectures. The key changes include:

1. **Physics Engine**: Replaced IsaacGym with MuJoCo
2. **Environment Interface**: Uses Gymnasium API instead of IsaacGym's custom API
3. **Simulation**: CPU-based simulation (MuJoCo) instead of GPU-based (IsaacGym)
4. **Rendering**: Uses MuJoCo's built-in renderer

## Files

### Core Files (Modified for MuJoCo)

- `env_mujoco.py` - MuJoCo-compatible environment base class set up
- `env_iccgan.py` - MuJoCo-compatible environment for ICCGAN humanoid envs
- `main.py` - Training and testing script

### Core Files (Unchanged)

- `models.py` - Actor-Critic and Discriminator models (no changes needed)
- `utils.py` - Utility functions for quaternion operations (no changes needed)
- `ref_motion.py` - Reference motion loading (no changes needed)

### Configuration

- `config/locomotion_walk_mujoco.py` - Example configuration file
- `requirements.txt` - Python dependencies

## Installation

1. Install MuJoCo (if not already installed):
```bash
pip install mujoco
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
python main.py config/locomotion_walk_mujoco.py --ckpt ./checkpoints/walk --device 0
```

### Testing/Visualization

```bash
python main.py config/locomotion_walk_mujoco.py --ckpt ./checkpoints/walk --test --render
```

### With Custom Config

```bash
python main.py path/to/your/config.py --ckpt ./checkpoints/exp --device 0
```

## Key Differences from IsaacGym Version

### 1. Environment Class

**IsaacGym (Original)**:
```python
from isaacgym import gymapi, gymtorch
env = gymapi.acquire_gym()
sim = env.create_sim(...)
```

**MuJoCo (This Version)**:
```python
import mujoco
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
```

### 2. Parallel Environments

**IsaacGym**: Supports thousands of parallel environments on GPU
**MuJoCo**: Supports fewer parallel environments (recommended: 32-128)

### 3. Configuration Changes

In your config files, change:
- `env_cls = "ICCGANHumanoidTarget"` → `env_cls = "ICCGANHumanoidMujoco"`
- Reduce `num_envs` from 512 to 32-128
- Adjust `batch_size` accordingly

### 4. Performance Considerations

- **IsaacGym**: GPU-accelerated, can handle 1000+ parallel environments
- **MuJoCo**: CPU-based, recommended 32-128 parallel environments

## Supported Features

### ✅ Implemented

- [x] Basic humanoid locomotion
- [x] ICCGAN observation function
- [x] Discriminator-based reward
- [x] PPO training
- [x] Reference motion loading (JSON, YAML, joblib)
- [x] Goal-conditioned tasks (target reaching)
- [x] Rendering (RGB array mode)

### ⚠️ Partially Implemented

- [ ] Multi-environment parallelization (limited by CPU)
- [ ] Interactive viewer (basic support)

### ❌ Not Implemented

- [ ] Juggling tasks (requires additional object spawning)
- [ ] Aiming tasks (requires additional link tracking)
- [ ] Real-time interactive control

## Troubleshooting

### Issue: "mujoco.MjModel.from_xml_path: XML parsing error"

**Solution**: Ensure your XML file path is correct and the file exists. The path should be relative to where you run the script.

### Issue: "RuntimeError: CUDA out of memory"

**Solution**: Reduce `num_envs` in your config file. Try 32 or 64 instead of 512.

### Issue: "ModuleNotFoundError: No module named 'mujoco'"

**Solution**: Install MuJoCo: `pip install mujoco`

### Issue: Slow training speed

**Solution**: 
- Reduce number of environments
- Use a machine with better CPU performance
- Consider using the original IsaacGym version if you have GPU access

## Converting Your Own Configs

To convert existing IsaacGym configs to MuJoCo:

1. Copy your config file (e.g., `my_task.py`)
2. Change `env_cls` to use MuJoCo version:
   ```python
   env_cls = "ICCGANHumanoidMujoco"  # or "ICCGANHumanoidTargetMujoco"
   ```
3. Add character model path:
   ```python
   env_params = dict(
       character_model="assets/humanoid.xml",
       # ... other params
   )
   ```
4. Adjust training params:
   ```python
   training_params = dict(
       num_envs=32,  # Reduce from 512
       batch_size=64,  # Reduce from 256
       # ... other params
   )
   ```

## Citation

If you use this code, please cite the original CompositeMotion paper:

```bibtex
@article{xu2023composite,
  title={Composite Motion Learning with Task Control},
  author={Xu, Pei and Cao, Zhenhua and Wang, Bohan and Shao, Tianyu and Yang, Libin and Zhou, Kun and Gao, Xiaogang},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={4},
  pages={1--14},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

## License

This MuJoCo conversion maintains the same MIT license as the original CompositeMotion project.

# CompositeMotion - MuJoCo Version (Fixed)

This is a fixed and complete MuJoCo-compatible version of the CompositeMotion project, converted from the original IsaacGym implementation.

## What's Fixed

### 1. NaN Issue in Observations
- Added `observe_iccgan_safe()` function that handles invalid quaternions
- Properly initializes state history from reference motion
- Added NaN detection and debugging in training loop

### 2. Complete Config Replication
All ICCGAN and composite motion configs from the original paper are included:

**ICCGAN Simple Motions:**
- `jaunty_walk.py` - Confident, swaggering walk
- `joyful_walk.py` - Happy, energetic walk
- `kick.py` - Martial arts kicking
- `limp_walk.py` - Injured, limping gait
- `long_jump.py` - Athletic jumping
- `punch.py` - Boxing/martial arts punching
- `roll.py` - Rolling/dodging motion
- `spinning_jump.py` - Acrobatic spinning jump
- `stomp_walk.py` - Heavy, forceful walking
- `stoop_walk.py` - Hunched-over, elderly walking
- `locomotion_crouch.py` - Crouched walking

**Composite Motions:**
- `aim_locomotion_walk.py` - Aiming while walking
- `chest_open_locomotion_walk.py` - Chest expansion while walking
- `front_jumping_jack_locomotion_walk.py` - Jumping jack arms while walking
- `punch_locomotion_walk.py` - Punching while walking
- `waist_twist_leg_lunge.py` - Waist twist with leg lunge

**Goal-Conditioned Locomotion:**
- `locomotion_walk.py` - Walking to target
- `locomotion_run.py` - Running to target
- `locomotion_walk_jaunty.py` - Jaunty walking to target

## Installation

```bash
pip install mujoco gymnasium torch numpy scipy pyyaml tensorboard imageio matplotlib opencv-python
```

## Usage

### 1. View Motions Before Training

Preview the reference motions that will be used for training:

```bash
# View motions from a config file
python motion_viewer.py config/iccgan/jaunty_walk.py --output motion_previews

# Or directly view a motion file
python motion_viewer.py assets/motions/iccgan/jaunty_walk.json --motion-only --output motion_previews
```

### 2. Train a Policy

```bash
# Train ICCGAN simple motion
python main_mujoco.py config/iccgan/jaunty_walk.py --ckpt ./checkpoints/jaunty_walk --device 0

# Train composite motion
python main_mujoco.py config/iccgan/punch_locomotion_walk.py --ckpt ./checkpoints/punch_walk --device 0

# Train goal-conditioned locomotion
python main_mujoco.py config/locomotion_walk.py --ckpt ./checkpoints/locomotion_walk --device 0
```

### 3. Test/Evaluate a Policy

```bash
# Test with rendering
python main_mujoco.py config/iccgan/jaunty_walk.py --ckpt ./checkpoints/jaunty_walk --test --render

# Test without rendering
python main_mujoco.py config/iccgan/jaunty_walk.py --ckpt ./checkpoints/jaunty_walk --test
```

### 4. Convert SMPL Data

Convert SMPL-format motion capture data to CompositeMotion JSON format:

```bash
# Convert single file
python smpl_converter.py path/to/motion.pkl --output output.json --fps 30

# Batch convert directory
python smpl_converter.py path/to/smpl_motions/ --output converted_motions/ --batch --fps 30
```

## File Structure

```
mujoco_v2/
├── env_mujoco.py              # Main environment (FIXED)
├── main_mujoco.py             # Training script (FIXED)
├── models.py                  # Neural networks (from original)
├── utils.py                   # Utilities (from original)
├── ref_motion.py              # Motion loading (from original)
├── motion_viewer.py           # Motion preview utility (NEW)
├── smpl_converter.py          # SMPL converter (NEW)
├── config/
│   ├── iccgan/                # ICCGAN simple motion configs
│   │   ├── jaunty_walk.py
│   │   ├── joyful_walk.py
│   │   ├── kick.py
│   │   └── ... (all 10+ motions)
│   ├── locomotion_walk.py     # Goal-conditioned walking
│   ├── locomotion_run.py      # Goal-conditioned running
│   └── locomotion_walk_jaunty.py
├── assets/
│   ├── humanoid.xml           # MuJoCo humanoid model
│   └── motions/
│       ├── iccgan/            # ICCGAN motion files
│       └── clips_walk.yaml    # Motion clip configurations
└── README.md
```

## Key Changes from Original

### Environment Classes

| Original (IsaacGym) | MuJoCo Version |
|---------------------|----------------|
| `ICCGANHumanoid` | `ICCGANHumanoidMujoco` |
| `ICCGANHumanoidTarget` | `ICCGANHumanoidTargetMujoco` |

### NaN Fix

The main issue was that `state_hist` was initialized with zeros, causing NaN in quaternion operations. The fix:

1. **Safe observation function** (`observe_iccgan_safe`):
   ```python
   # Replace invalid quaternions with identity
   invalid_mask = quat_norms < 1e-6
   quats = torch.where(invalid_mask, identity_quat, quats / quat_norms)
   ```

2. **Proper initialization**: State history is now properly initialized from reference motion

3. **NaN detection**: Training loop now detects and reports NaN values

## Training Parameters

Default parameters (adjusted for MuJoCo CPU simulation):

```python
TRAINING_PARAMS = dict(
    horizon=8,
    num_envs=32,        # Reduced from 512 for CPU
    batch_size=64,      # Reduced from 256
    opt_epochs=5,
    actor_lr=5e-6,
    critic_lr=1e-4,
    gamma=0.95,
    lambda_=0.95,
    disc_lr=1e-5,
    max_epochs=10000,
    save_interval=2000,
    log_interval=50,
    terminate_reward=-1,
)
```

## Troubleshooting

### NaN in Observations

If you still see NaN values:
1. Check that motion files are valid JSON
2. Verify that the humanoid.xml matches the motion skeleton
3. Check the reference motion viewer output

### Slow Training

MuJoCo is CPU-based, so training is slower than IsaacGym:
- Use fewer environments (`num_envs=16` or `32`)
- Consider using a machine with more CPU cores
- The original IsaacGym version is recommended if you have GPU access

### Motion File Not Found

Make sure motion files are in the correct location:
- ICCGAN motions: `assets/motions/iccgan/`
- Clip configurations: `assets/motions/clips_*.yaml`

## Citation

If you use this code, please cite the original CompositeMotion paper:

```bibtex
@article{xu2023composite,
  title={Composite Motion Learning with Task Control},
  author={Xu, Pei and Cao, Zhenhua and Wang, Bohan and Shao, Tianyu and Yang, Libin and Zhou, Kun and Gao, Xiaogang},
  journal={ACM Transactions on Graphics (TOG)},
  volume={42},
  number={4},
  pages={1--14},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

## License

This MuJoCo conversion maintains the same MIT license as the original CompositeMotion project.

## Additional Resources

- Original paper: https://arxiv.org/abs/2305.03286
- Original code: https://github.com/xupei0610/CompositeMotion
- MuJoCo docs: https://mujoco.readthedocs.io/
- Gymnasium docs: https://gymnasium.farama.org/
