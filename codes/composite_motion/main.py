"""
Main training script for MuJoCo version of CompositeMotion
Converted from IsaacGym version
"""

import os, sys
import importlib.util
from collections import namedtuple

import .include.env_iccgan as env
from .include.models import ACModel, Discriminator
from .include.helpers import test, train

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

TRAINING_PARAMS = dict(     # env - represent params to set to environment (make sure to verify if applied in cmds.txt)
    horizon = 16,           # default = 8 [each PPO epoch (one buffer fill) takes 8 env steps] --> longer rollouts give much better GAE advantage estimates
    num_envs = 32,          # default = 512 (512*8 = 256*16 --> 16 batches produced) -- heavy 1ep = 20s
    simulation_speed = 120, # env-set: physics simulation running at 120 hz as per paper
    batch_size = 256,       # default = 256
    opt_epochs = 5,
    actor_lr = 5e-6,
    critic_lr = 1e-4,
    gamma = 0.95,
    lambda_ = 0.95,
    disc_lr = 1e-5,         # default = 1e-5, slow it when incresing horizon, to give policy time to respond
    log_interval = 10,
    control_mode="position",
    # env-set: update xml for differ model simulations, but make sure no.of body parts are same
    character_model=os.path.join("assets", "humanoid_posctrl.xml"),
    term_height = 0.15,     # default = 0.15 
    grace_steps = 3,        # default <= 1 (only for testing, adding in training will disrupt learning behaviour)
# ---- ---- ---- ---- ---- Best to update bottom params in config.py and top params here (for convienience) ---- ---- ---- ---- ----
    max_epochs = 10000,     # 10000 iterations / 8 per PPO epoch = 125 PPO epochs --> in each training loop
    save_interval = 500,
    terminate_reward = -1,  # default = -1, update in config file
    # --- (NEW) Symmetry Regularization: Bilateral symmetry loss ---
    sym_loss_coeff = 0.0,   # train-set: > 0 to enable. Recommended start: 0.005
)
'''
# has been moved to config files resp
ENV_PARAMS = dict(
    # --- NOTE: set cycles count > 1 for training, only is motion is loopable seamlessly (test always runs in 2 times this value)
    max_cycles = 1,         # env-set: hard reset episode length = max_cycles * max_motion_len * avg_fps.
    # --- (NEW) Phase Input: Phase-conditioned observations ---
    loop_phase_obs = False, # env-set: True when motion is looped for better results. phase_period --> gait cycle, best to calc from ref-motion
)
'''
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
            return ckpt_dir, weights_file
        else:
            ckpt_dir = ckpt_path
            if last_folder != config_base:
                ckpt_dir = os.path.join(ckpt_dir, config_base)
            
    else:
        if ("ckpt" in last_folder): # given path to file that doesnt exist
            ckpt_dir = os.path.dirname(ckpt_path)
            weights_file = ckpt_path
            return ckpt_dir, weights_file
        else: # treat as directory and apply config_base logic
            ckpt_dir = ckpt_path
            if last_folder != config_base:
                ckpt_dir = os.path.join(ckpt_dir, config_base)

    # --- Added for my convention's convinience ---
    # grp_dir = os.path.dirname(ckpt_dir)
    # grp_dir += '-'
    # if env.loop_phase_obs:
    #     grp_dir += 'l'
    # if training_params.sym_loss_coeff > 0.0:
    #     grp_dir += 's'
    # grp_dir += f'-h{training_params.horizon}'
    # ckpt_dir = os.path.join(grp_dir, os.path.basename(ckpt_dir))
    # --- remove if not needed ---
    os.makedirs(ckpt_dir, exist_ok=True)
    weights_file = os.path.join(ckpt_dir, "ckpt")
    return ckpt_dir, weights_file

def get_param_dict(obj):
    def is_parsable(val):
        # Allow only int, float, str, bool, None, or containers of these
        primitive = (int, float, str, bool)
        if isinstance(val, primitive):
            return True
        elif isinstance(val, (list, tuple)):
            return all(is_parsable(x) for x in val)
        elif isinstance(val, dict):
            return all(is_parsable(k) and is_parsable(v) for k, v in val.items())
        else:
            return False

    # Write all environment class parameters
    params = {}
    for attr in dir(obj):
        if not callable(getattr(obj, attr)) and not attr.startswith('_'):
            value = getattr(obj, attr)
            # Exclude torch.Tensor, np.ndarray, all-caps (default placeholder attributes)
            if isinstance(value, (torch.Tensor, np.ndarray)) or attr.isupper(): continue
            # Add all other primitives that json parsable
            if value is None or is_parsable(value):
                #print(attr, type(value))
                params[attr] = value
    
    #print(obj.__dict__.keys())
    return params

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
        env_cls = env.ICCGANHumanoid
    
        
    render_mode = None
    if settings.render:
        render_mode = "rgb_array" # settings.render = 'non' --> non-interactive rgb_array
        if settings.render == 'int': #interactive human realtime
            render_mode = "human"

    print(f"Env(render:{settings.render}-{render_mode}): {env_cls} Params:\n{config.env_params}")
    if settings.test:
        num_envs = 1        # one env will be rendering realtime to view
    else:
        num_envs = training_params.num_envs
    
    # if config file env params grace_steps given, its supposed based on motion difficulty type, dont override it
    grace_steps = config.env_params.pop('grace_steps', None) 
    if grace_steps is None:
        if settings.test:
            grace_steps = training_params.grace_steps
            print(f"Grace-Steps allowed for testing by training-params => {grace_steps}\n")
        else:
            grace_steps = 1     # allowing grace steps in training is not ideal
            print(f"Grace-Steps strict rule overwritten for training (reset to default) => {grace_steps}\n")
    else:
        if settings.test:
            grace_steps *= 2
        print(f"Grace-Steps overwritten by env-params {"(doubled for test) " if settings.test  else ""}=> {grace_steps}\n")

    # Create environment
    env = env_cls(
        num_envs,
        character_model = training_params.character_model,
        discriminators = discriminators,
        compute_device = settings.device,
        render_mode = render_mode,
        run_speed = training_params.simulation_speed,
        term_height = training_params.term_height,
        grace_steps = grace_steps,
        verbose = settings.verbose,
        #max_cycles = ENV_PARAMS.max_cycles,
        #loop_phase_obs = ENV_PARAMS.loop_phase_obs,
        **config.env_params
    )
    
    if settings.test:
        env.max_cycles *= 2 # double max_cycles for testing

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
            print(f"WARNING! Unable to Load model (using random weights) from path(exists?{os.path.exists(weights_file)}) : {os.path.abspath(weights_file)}\n")
        test(env, model, settings.ckpt)
    else:
        import shutil
        from datetime import datetime
        #shutil.copy(settings.config, settings.ckpt) # uncomment to copy config onto checkpoints folder for ease of access

        init_epoch = load_latest_checkpoint(model, settings.ckpt)
        TRAINING_PARAMS["init_epoch"] = init_epoch # additing to editable dict just for saving purposes
        env_params = get_param_dict(env) # keep track env setup as well
        with open(os.path.join(settings.ckpt, "cmds.txt"), "a") as f: # keep track of cmds
            f.write(f"{datetime.now()} \"{" ".join(sys.argv)}\"\n")
            f.write(f"Training Params: {TRAINING_PARAMS}\n")
            f.write(f"Environment Params: {env_params}\n\n")

        train(env, model, settings.ckpt, training_params, init_epoch)
