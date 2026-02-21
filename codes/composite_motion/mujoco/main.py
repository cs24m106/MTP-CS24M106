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
    num_envs = 1,       # only one env setup done for now (scaleablity need to do in future)
    batch_size = 256,
    opt_epochs = 5,
    actor_lr = 5e-6,
    critic_lr = 1e-4,
    gamma = 0.95,
    lambda_ = 0.95,
    disc_lr = 1e-5,
    max_epochs = 10000, # 10000 iterations / 8 per PPO epoch = 125 PPO epochs --> in each training loop
    save_interval = None,
    log_interval = 10,
    terminate_reward = -1,
    control_mode="position",
    character_model=os.path.join("assets", "humanoid.xml"),
    # update xml for differ model simulations, but make sure no.of body parts are same
)


def safe_load_model(model, weights_file):
    print(f"Loading weights from {weights_file} ...")
    state = torch.load(weights_file, map_location=next(model.parameters()).device)
    load_match_success = True

    # First try strict load (preferred) and give a helpful error if it fails
    try:
        model.load_state_dict(state["model"])
        print("Model loaded(strict=True) sucessfully!")
        # If norm is very small (< 0.1), weights might not have loaded
        first_param = next(model.parameters())
        print(f"Model check: device={first_param.device}, norm={first_param.norm().item():.4f}")
    except RuntimeError as e:
        # Show the error and try safe fallback
        print("Strict load failed (expected when architectures differ).")
        print("RuntimeError:", e)
        # Print env-derived dims for comparison
        print(f"\nCurrent env properties: ob_dim: {env.ob_dim}, ob_horizon: {env.ob_horizon}, state_dim: {env.state_dim},", end=" ")
        print(f"goal_dim: {env.goal_dim}, rew_dim: {env.rew_dim}, disc_dim: {getattr(env, 'disc_dim', None)},", end=" ")
        print(f"  discriminators keys: {list(env.discriminators.keys()) if hasattr(env, 'discriminators') else None}, model value_dim: {len(env.discriminators) + env.rew_dim}")
        print("\nAttempting load_state_dict(..., strict=False) to load matching keys and report mismatches ...", end= " ")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        print("load_state_dict(strict=False) completed.")
        print(">>> UNEXPECTED keys in current model (present in checkpoint but not used):", sorted(list(unexpected)))
        print(">>> MISSING keys in current model (these keys were not loaded because shapes differ or absent):", sorted(list(missing)))
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

    if os.path.exists(os.path.join(settings.ckpt)):
        # Set weights_file --> model save file, settings.ckpt --> folder containing the save file
        weights_file = None
        if os.path.isdir(settings.ckpt):
            weights_file = os.path.join(settings.ckpt, "ckpt")
        else:
            weights_file = settings.ckpt
            settings.ckpt = os.path.dirname(weights_file)
    else:
        if (os.path.splitext(settings.ckpt)[1]): # given path to file that doesnt exist
            weights_file = settings.ckpt
            settings.ckpt = os.path.dirname(weights_file)
        else:
            weights_file = os.path.join(settings.ckpt, "ckpt")
        os.makedirs(settings.ckpt, exist_ok=True) # make sure the checkpoint folder exists

    if settings.test:
        load_success = False
        if settings.ckpt is not None and os.path.exists(weights_file):
            load_success = safe_load_model(model, weights_file)
        if not load_success:
            print(f"WARNING! Unable to Load model (using random weights) from path(exists?{os.path.exists(weights_file)}) : {settings.ckpt}\n")
            os.makedirs(settings.ckpt, exist_ok=True)
        
        test(env, model, settings.ckpt)
    else:
        import shutil
        from datetime import datetime
        #shutil.copy(settings.config, settings.ckpt) # uncomment to copy config onto checkpoints folder for ease of access
        with open(os.path.join(settings.ckpt, "commands.txt"), "a") as f: # keep track of cmds
            f.write(f"{datetime.now()} \"{" ".join(sys.argv)}\"\n")
        train(env, model, settings.ckpt, training_params, init_epoch = load_latest_checkpoint(model, settings.ckpt))
