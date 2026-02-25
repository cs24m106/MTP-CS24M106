"""
Main training script for MuJoCo version of CompositeMotion
Converted from IsaacGym version
"""

import os, sys
import importlib.util
from collections import namedtuple

import env_iccgan as env
from models import ACModel, Discriminator
from helpers import test, train

import numpy as np
import random
from imageio import imwrite
import argparse

import torch
import warnings # ignore cuda warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str,
    help="Configure file used for training. Please refer to files in `config` folder.")
parser.add_argument("--ckpt", type=str, default=None,
    help="Checkpoint directory or file for training or evaluation.")
parser.add_argument("--test", action="store_true", default=False,
    help="Run visual evaluation.")
parser.add_argument("--seed", type=int, default=42,
    help="Random seed.")
parser.add_argument("--device", type=int, default=0,
    help="ID of the target GPU device for model running.")
parser.add_argument(
    "--render",
    choices=["non", "int"],
    nargs="?",
    const="non",  # Value when flag is present but no argument given (--render)
    default=None,  # Value when flag is absent (no --render)
    help=(
        "Rendering mode: "
        "omit flag for no rendering (default=None), "
        "'--render' alone for 'rgb_array' offscreen rendering, "
        "or '--render int' for interactive window. "
        "Choices: 'non' (offscreen), 'int' (interactive)."
    )
)
parser.add_argument("--verbose", action="store_true", default=False,
    help="To print debug statements")
settings = parser.parse_args()

# src code followup
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# our additionals
torch.use_deterministic_algorithms(True)
torch.set_printoptions(linewidth=120,  precision=6)  # fixed-point, no exponent, space in +ve vals
np.set_printoptions(linewidth=120, suppress=True, precision=6, sign=' ', floatmode='fixed') 
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # use for debuging without 

TRAINING_PARAMS = dict(
    horizon = 8,        # each PPO epoch (one buffer fill) takes 8 env steps.
    num_envs = 32,       # 32*8 = 256 --> single batch produced
    simulation_speed = 120, # physics simulation running at 120 hz as per paper
    batch_size = 256,
    opt_epochs = 5,
    actor_lr = 5e-6,
    critic_lr = 1e-4,
    gamma = 0.95,
    lambda_ = 0.95,
    disc_lr = 1e-5,
    max_epochs = 10000, # 10000 iterations / 8 per PPO epoch = 125 PPO epochs --> in each training loop
    save_interval = 500,
    log_interval = 10,
    terminate_reward = -5,    # FIX BUG5: was -1. With disc_reward≈-0.5/step, falling immediately (V=-1) was
                              # ALWAYS better than surviving 5+ steps (V=-3.6). Policy learned to fall faster.
                              # With -5: surviving 8 steps at -0.5/step = V≈-3.8 > -5 → survival is rewarded.
    control_mode="position",
    character_model=os.path.join("assets", "humanoid.xml"),
    # update xml for differ model simulations, but make sure no.of body parts are same
)


def safe_load_model(model, weights_file):
    print(f"Loading weights from {weights_file} ...")
    state = torch.load(weights_file, map_location=next(model.parameters()).device)
    load_match_success = True
    
    try:
        model.load_state_dict(state["model"])
        print("Model loaded (strict=True) successfully!")
    except RuntimeError as e:
        print("Strict load failed:", e)
        # Build a compatible state dict: only load keys with matching shapes
        current_state = model.state_dict()
        compat_state = {}
        missing, shape_mismatch, loaded = [], [], []
        for k, v in state["model"].items():
            if k not in current_state:
                missing.append(k)
            elif v.shape != current_state[k].shape:
                shape_mismatch.append(f"{k}: ckpt={v.shape} vs model={current_state[k].shape}")
            else:
                compat_state[k] = v
                loaded.append(k)
        current_state.update(compat_state)
        model.load_state_dict(current_state)
        print(f"Loaded {len(loaded)} matching keys.")
        print(f"Shape mismatches (skipped): {shape_mismatch}")
        print(f"Missing keys (skipped): {missing}")
        load_match_success = False
    
    print()
    return load_match_success

def load_latest_checkpoint(model, ckpt_dir):
    """Find the checkpoint with highest run number"""
    ckpt_files = [f for f in os.listdir(settings.ckpt) if f.startswith("ckpt-")]
    if not ckpt_files:
        return 0

    # Find largest epoch number
    epochs = [int(f.split("ckpt-")[-1]) for f in ckpt_files if f.split("ckpt-")[-1].isdigit()]
    prev_epochs = max(epochs)
    weights_file = os.path.join(settings.ckpt, f"ckpt-{prev_epochs}")
    model_state = torch.load(weights_file, map_location=next(model.parameters()).device)
    model.load_state_dict(model_state["model"])
    print(f"Resuming from prev ckpt-{prev_epochs} ... Load successful :-)")
    return prev_epochs

    # --- Robust checkpoint path logic ---
def get_ckpt_dir_and_weights_file(ckpt_path, config_path):
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    ckpt_path = os.path.normpath(ckpt_path)
    last_folder = os.path.basename(ckpt_path)
    if os.path.exists(ckpt_path):
        if os.path.isfile(ckpt_path):
            ckpt_dir = os.path.dirname(ckpt_path)
            weights_file = ckpt_path
        else:
            ckpt_dir = ckpt_path
            if last_folder != config_base:
                ckpt_dir = os.path.join(ckpt_dir, config_base)
            weights_file = os.path.join(ckpt_dir, "ckpt")
    else:
        if ("ckpt" in last_base): # given path to file that doesnt exist
            ckpt_dir = os.path.dirname(ckpt_path)
            weights_file = ckpt_path
        else: # treat as directory and apply config_base logic
            ckpt_dir = ckpt_path
            if last_folder != config_base:
                ckpt_dir = os.path.join(ckpt_dir, config_base)
            weights_file = os.path.join(ckpt_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir, weights_file


if __name__ == "__main__":
    # Load config
    if os.path.splitext(settings.config)[-1] in [".pkl", ".json", ".yaml"]:
        config = object()
        config.env_params = dict(motion_file=settings.config)
    else:
        spec = importlib.util.spec_from_file_location("config", settings.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)

    if hasattr(config, "training_params"):
        TRAINING_PARAMS.update(config.training_params)
    if not TRAINING_PARAMS["save_interval"]:
        TRAINING_PARAMS["save_interval"] = TRAINING_PARAMS["max_epochs"]
    
    print(f"\nTraining Params:\n {TRAINING_PARAMS}\n")
    training_params = namedtuple('x', TRAINING_PARAMS.keys())(*TRAINING_PARAMS.values())
    
    discriminators = {}
    if hasattr(config, "discriminators"):
        discriminators = {
            name: env.DiscriminatorConfig(**prop)
            for name, prop in config.discriminators.items()
        }

    if hasattr(config, "env_cls"):
        env_cls = getattr(env, config.env_cls)
    else:
        env_cls = env.ICCGANHumanoidMujoco
    
        
    render_mode = None
    if settings.render:
        render_mode = "rgb_array" # settings.render = 'non' --> non-interactive rgb_array
        if settings.render == 'int': #interactive human realtime
            render_mode = "human"

    print(f"Env(render:{settings.render}-{render_mode}): {env_cls} Params:\n{config.env_params}\n")

    if settings.test:
        num_envs = 1
    else:
        num_envs = training_params.num_envs

    # Create environment
    env = env_cls(
        num_envs,
        character_model=training_params.character_model,
        discriminators=discriminators,
        compute_device=settings.device,
        render_mode=render_mode,
        run_speed=training_params.simulation_speed,
        verbose = settings.verbose,
        **config.env_params
    )
    
    if settings.test:
        env.episode_length = 500000

    value_dim = len(env.discriminators) + env.rew_dim
    model = ACModel(env.state_dim, env.act_dim, env.goal_dim, value_dim)
    discriminators = torch.nn.ModuleDict({
        name: Discriminator(dim) for name, dim in env.disc_dim.items()
    })
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    model.to(device)
    discriminators.to(device)
    model.discriminators = discriminators

    settings.ckpt, weights_file = get_ckpt_dir_and_weights_file(settings.ckpt, settings.config)

    if settings.test:
        load_success = False
        if settings.ckpt is not None and os.path.exists(weights_file):
            load_success = safe_load_model(model, weights_file)
        if not load_success:
            print(f"WARNING! Unable to Load model (using random weights) from path(exists?{os.path.exists(weights_file)}) : {settings.ckpt}\n")
        test(env, model, settings.ckpt)
    else:
        import shutil
        from datetime import datetime
        #shutil.copy(settings.config, settings.ckpt) # uncomment to copy config onto checkpoints folder for ease of access
        with open(os.path.join(settings.ckpt, "cmds.txt"), "a") as f: # keep track of cmds
            f.write(f"{datetime.now()} \"{" ".join(sys.argv)}\"\n")
        train(env, model, settings.ckpt, training_params, init_epoch = load_latest_checkpoint(model, settings.ckpt))
