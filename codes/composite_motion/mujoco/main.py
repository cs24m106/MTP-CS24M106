"""
Main training script for MuJoCo version of CompositeMotion
Converted from IsaacGym version
"""

import os, sys, time
import importlib.util
from collections import namedtuple

import env_iccgan as env
from models import ACModel, Discriminator

import numpy as np
import random, cv2
from imageio import imwrite
import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
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


os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(settings.seed)
np.random.seed(settings.seed)
random.seed(settings.seed)
torch.manual_seed(settings.seed)
torch.cuda.manual_seed(settings.seed)
torch.cuda.manual_seed_all(settings.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True)
except:
    pass

TRAINING_PARAMS = dict(
    horizon = 8,
    num_envs = 512,
    batch_size = 256,
    opt_epochs = 5,
    actor_lr = 5e-6,
    critic_lr = 1e-4,
    gamma = 0.95,
    lambda_ = 0.95,
    disc_lr = 1e-5,
    max_epochs = 10000,
    save_interval = None,
    log_interval = 10,
    terminate_reward = -1,
    control_mode="position",
    character_model=os.path.join("assets", "humanoid_5xpd.xml"),
    # update xml for differ model simulations, but make sure no.of body parts are same
)

def check_exit(env):
    """Check for keyboard input for exit simulation (works for both render modes)"""
    if env.render_mode == "rgb_array":
        # For offscreen rendering with cv2 window
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q") or key == ord("Q"):
            return True
    elif env.render_mode == "human" and env.viewer is not None:
        # For interactive viewer, check if window was closed
        # MuJoCo passive viewer doesn't have direct keyboard access
        try:
            # But you can check if viewer is still alive -> If closed, this fails
            if not env.viewer.is_running():
                return True
        except:
            return True
    
    return False

def test(env, model, save_dir, max_steps=10000):
    """Test the trained model"""
    model.eval()
    env.eval()
    env.reset()    
    steps = 0
    
    if env.render_mode is not None:
        save_folder = f"env_{env.render_mode}"
        os.makedirs(os.path.join(save_dir, save_folder), exist_ok=True)
    
    while steps < max_steps:
        with torch.no_grad():
            obs, info = env.reset_done()
            
            # Check for NaN in observations
            if torch.isnan(obs).any():
                nan_count = torch.isnan(obs).sum().item()
                print(f"ERROR: NaN detected in obs, count: {nan_count}")
                print(f"NaN locations: {torch.where(torch.isnan(obs))}")
                print(f"Obs sample:\n{obs[0]}")
                break
            
            seq_len = info["ob_seq_lens"]
            actions = model.act(obs, seq_len - 1)
            
            # Check for NaN in actions
            if torch.isnan(actions).any():
                print(f"ERROR: NaN detected in actions")
                break
        
        # DEBUG Problem
        if env.verbose and steps == 0:
            print(f"Observation stats:")
            print(f"  Shape: {obs.shape}")
            print(f"  Min: {obs.min():.4f}, Max: {obs.max():.4f}")
            print(f"  Mean: {obs.mean():.4f}, Std: {obs.std():.4f}")
            print(f"  NaN count: {torch.isnan(obs).sum()}")
            print(f"  First 10 obs: {obs[0, :10]}")
            print(f"\n[ACTION DEBUG]")
            print(f"  Raw actions: min={actions.min():.4f}, max={actions.max():.4f}, mean={actions.mean():.4f}")
            print(f"  Non-zero count: {(actions.abs() > 0.01).sum().item()}/{actions.numel()}")
            print(f"  First 10: {actions[0, :10]}\n")

        obs, rewards, terminated, truncated, info = env.step(actions.cpu().numpy())
        if rewards.size > 0:
            title = f"Step {steps}, reward: {rewards.mean():.4f}"
        else:
            title = f"Step {steps} (no task reward - discriminator-only mode)"
        
        # Render if enabled
        if env.render_mode is not None:
            frame = env.render()
            #print(f"frame {f"(dtype={frame.dtype})" if hasattr(frame, "dtype") else ""}: \n{frame}\n")
            
            # Offscreen frames returned for (rgb_array)
            if env.render_mode == "rgb_array" and isinstance(frame, np.ndarray):
                bgr = cv2.cvtColor(
                    np.ascontiguousarray(frame, dtype=np.uint8),  # Force contiguous uint8 array
                    cv2.COLOR_RGB2BGR  # Convert RGB → BGR (OpenCV's native format)
                )
                
                # Draw a black outline then white text for readability
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (10, 25)  # x,y
                cv2.putText(bgr, title, org, font, 0.6, (0, 0, 0), thickness=3, lineType=cv2.LINE_AA)
                cv2.putText(bgr, title, org, font, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

                # Show window (uncomment to save)
                cv2.imshow(save_folder, bgr)
                #cv2.imwrite(os.path.join(save_dir, save_folder, f"frame_{steps:05d}.png"), bgr)
        
            # 1 ms wait keeps window responsive; ESC or 'q' to quit test early
            key = cv2.waitKey(1) & 0xFF  # compare using Unicode code, i.e. use ord()
            if check_exit(env):
                print(f"User requested exit!")
                break
        
        # Print progress every 100 steps
        if steps % 100 == 0:
            #render_file = os.path.join(save_dir, save_folder, f"frame_{steps:05d}.png")
            print(title)
            #print(f"... check save_path (exists?{os.path.exists(render_file)}) : {render_file}")
        
        steps += 1
    
    print(f"Test completed: {steps} steps")


def train(env, model, ckpt_dir, training_params):
    """Train the model using PPO"""
    if ckpt_dir is not None:
        logger = SummaryWriter(ckpt_dir)
    else:
        logger = None

    optimizer = torch.optim.Adam([
        {"params": model.actor.parameters(), "lr": training_params.actor_lr},
        {"params": model.critic.parameters(), "lr": training_params.critic_lr}
    ])
    ac_parameters = list(model.actor.parameters()) + list(model.critic.parameters())
    disc_optimizer = {name: torch.optim.Adam(disc.parameters(), training_params.disc_lr) 
                      for name, disc in model.discriminators.items()}

    buffer = dict(
        s=[], a=[], v=[], lp=[], v_=[], not_done=[], terminate=[],
        ob_seq_len=[]
    )
    multi_critics = env.reward_weights is not None and env.reward_weights.size(-1) > 1
    if multi_critics:
        buffer["reward_weights"] = []
    has_goal_reward = env.rew_dim > 0
    if has_goal_reward:
        buffer["r"] = []

    buffer_disc = {
        name: dict(fake=[], real=[]) for name in env.discriminators.keys()
    }
    real_losses, fake_losses = {n: [] for n in buffer_disc.keys()}, {n: [] for n in buffer_disc.keys()}

    BATCH_SIZE = training_params.batch_size
    HORIZON = training_params.horizon
    GAMMA = training_params.gamma
    GAMMA_LAMBDA = training_params.gamma * training_params.lambda_
    OPT_EPOCHS = training_params.opt_epochs
    LOG_INTERVAL = training_params.log_interval

    epoch = 0
    model.eval()
    env.train()
    env.reset()
    
    tic = time.time()
    total_steps = 0
    
    while total_steps < training_params.max_epochs * training_params.num_envs:
        with torch.no_grad():
            obs, info = env.reset_done()
            
            # Check for NaN
            if torch.isnan(obs).any():
                print(f"ERROR: NaN in observations at step {total_steps}")
                break
            
            seq_len = info["ob_seq_lens"]
            reward_weights = info["reward_weights"]
            actions, values, log_probs = model.act(obs, seq_len - 1, stochastic=True)
            
            obs_next, rews, terminated, truncated, info = env.step(actions.cpu().numpy())
            
            log_probs = log_probs.sum(-1, keepdim=True)
            dones = torch.from_numpy(np.logical_or(terminated, truncated)).to(env.device)
            not_done = (~dones).unsqueeze_(-1)
            terminate = info["terminate"]
            
            if env.discriminators:
                fakes = info["disc_obs"]
                reals = info["disc_obs_expert"]

            values_ = model.evaluate(torch.from_numpy(obs_next).to(env.device), seq_len)

        buffer["s"].append(obs)
        buffer["a"].append(actions)
        buffer["v"].append(values)
        buffer["lp"].append(log_probs)
        buffer["v_"].append(values_)
        buffer["not_done"].append(not_done)
        buffer["terminate"].append(terminate)
        buffer["ob_seq_len"].append(seq_len)
        
        if has_goal_reward:
            buffer["r"].append(torch.from_numpy(rews).to(env.device))
        if multi_critics:
            buffer["reward_weights"].append(reward_weights)
        if env.discriminators:
            for name, fake in fakes.items():
                buffer_disc[name]["fake"].append(fake)
                buffer_disc[name]["real"].append(reals[name])

        total_steps += training_params.num_envs

        if len(buffer["s"]) == HORIZON:
            disc_data = []
            ob_seq_lens = torch.cat(buffer["ob_seq_len"]).to(env.device)
            ob_seq_end_frames = ob_seq_lens - 1
            
            if env.discriminators:
                with torch.no_grad():
                    for name, data in buffer_disc.items():
                        disc = model.discriminators[name]
                        fake = torch.cat(data["fake"]).to(env.device)
                        real_ = torch.cat(data["real"]).to(env.device)
                        end_frame = ob_seq_lens

                        length = torch.arange(fake.size(1), 
                            dtype=end_frame.dtype, device=env.device
                        ).unsqueeze_(0)
                        mask = length <= end_frame.unsqueeze(1)
                        mask_ = length >= fake.size(1) - 1 - end_frame.unsqueeze(1)

                        real = torch.zeros_like(real_, device=env.device)
                        real[mask] = real_[mask_]
                        disc.ob_normalizer.update(fake[mask])
                        disc.ob_normalizer.update(real[mask])
                        ob = disc.ob_normalizer(fake)
                        ref = disc.ob_normalizer(real)
                        disc_data.append((name, disc, ref, ob, end_frame))

                model.train()
                n_samples = 0
                for name, disc, ref, ob, seq_end_frame_ in disc_data:
                    real_loss = real_losses[name]
                    fake_loss = fake_losses[name]
                    opt = disc_optimizer[name]
                    
                    if len(ref) != n_samples:
                        n_samples = len(ref)
                        idx = torch.randperm(n_samples, device=env.device)
                    
                    for batch in range(n_samples // BATCH_SIZE):
                        sample = idx[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]

                        score_r = disc(r, seq_end_frame, normalize=False)
                        score_f = disc(f, seq_end_frame, normalize=False)

                        loss_r = torch.nn.functional.relu(1 - score_r).mean()
                        loss_f = torch.nn.functional.relu(1 + score_f).mean()

                        with torch.no_grad():
                            alpha = torch.rand(r.size(0), dtype=r.dtype, device=r.device)
                            alpha = alpha.view(-1, *([1] * (r.ndim - 1)))
                            interp = alpha * r + (1 - alpha) * f
                        
                        interp.requires_grad = True
                        with torch.backends.cudnn.flags(enabled=False):
                            score_interp = disc(interp, seq_end_frame, normalize=False)
                        
                        grad = torch.autograd.grad(
                            score_interp, interp, torch.ones_like(score_interp),
                            retain_graph=True, create_graph=True, only_inputs=True
                        )[0]
                        gp = grad.reshape(grad.size(0), -1).norm(2, dim=1).sub(1).square().mean()
                        l = loss_f + loss_r + 10 * gp
                        l.backward()
                        opt.step()
                        opt.zero_grad()

                        real_loss.append(score_r.mean().item())
                        fake_loss.append(score_f.mean().item())

            model.eval()
            with torch.no_grad():
                terminate = torch.cat(buffer["terminate"])
                if multi_critics:
                    reward_weights = torch.cat(buffer["reward_weights"])
                    rewards = torch.empty_like(reward_weights)
                else:
                    reward_weights = None
                    rewards = None
                
                for name, disc, _, ob, seq_end_frame in disc_data:
                    r = (disc(ob, seq_end_frame, normalize=False).clamp_(-1, 1)
                         .mean(-1, keepdim=True))
                    if rewards is None:
                        rewards = r
                    else:
                        rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                
                if has_goal_reward:
                    rewards_task = torch.cat(buffer["r"])
                    if rewards is None:
                        rewards = rewards_task
                    else:
                        rewards[:, -rewards_task.size(-1):] = rewards_task
                else:
                    rewards_task = None
                
                rewards[terminate] = training_params.terminate_reward

                values = torch.cat(buffer["v"])
                values_ = torch.cat(buffer["v_"])
                
                if model.value_normalizer is not None:
                    values = model.value_normalizer(values, unnorm=True)
                    values_ = model.value_normalizer(values_, unnorm=True)
                
                values_[terminate] = 0
                rewards = rewards.view(HORIZON, -1, rewards.size(-1))
                values = values.view(HORIZON, -1, values.size(-1))
                values_ = values_.view(HORIZON, -1, values.size(-1))

                not_done = buffer["not_done"]
                advantages = (rewards - values).add_(values_, alpha=GAMMA)
                for t in reversed(range(HORIZON - 1)):
                    advantages[t].add_(advantages[t + 1] * not_done[t], alpha=GAMMA_LAMBDA)

                advantages = advantages.view(-1, advantages.size(-1))
                returns = advantages + values.view(-1, advantages.size(-1))

                log_probs = torch.cat(buffer["lp"])
                actions = torch.cat(buffer["a"])
                states = torch.cat(buffer["s"])
                ob_seq_lens = torch.cat(buffer["ob_seq_len"])
                ob_seq_end_frames = ob_seq_lens - 1

                sigma, mu = torch.std_mean(advantages, dim=0, unbiased=True)
                advantages = (advantages - mu) / (sigma + 1e-8)

                length = torch.arange(env.ob_horizon, 
                    dtype=ob_seq_lens.dtype, device=ob_seq_lens.device)
                mask = length.unsqueeze_(0) < ob_seq_lens.unsqueeze(1)
                states_raw = model.observe(states, norm=False)[0]
                model.ob_normalizer.update(states_raw[mask])
                
                if model.value_normalizer is not None:
                    model.value_normalizer.update(returns)
                    returns = model.value_normalizer(returns)
                
                if multi_critics:
                    advantages = advantages.mul_(reward_weights)

            n_samples = advantages.size(0)
            epoch += 1
            model.train()
            policy_loss, value_loss = [], []
            
            for _ in range(OPT_EPOCHS):
                idx = torch.randperm(n_samples)
                for batch in range(n_samples // BATCH_SIZE):
                    sample = idx[BATCH_SIZE * batch: BATCH_SIZE * (batch + 1)]
                    s = states[sample]
                    a = actions[sample]
                    lp = log_probs[sample]
                    adv = advantages[sample]
                    v_t = returns[sample]
                    end_frame = ob_seq_end_frames[sample]

                    pi_, v_ = model(s, end_frame)
                    lp_ = pi_.log_prob(a).sum(-1, keepdim=True)

                    ratio = torch.exp(lp_ - lp)
                    clipped_ratio = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                    pg_loss = -torch.min(adv * ratio, adv * clipped_ratio).sum(-1).mean()
                    vf_loss = (v_ - v_t).square().mean()

                    loss = pg_loss + 0.5 * vf_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ac_parameters, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    policy_loss.append(pg_loss.item())
                    value_loss.append(vf_loss.item())
            
            model.eval()
            for v in buffer.values():
                v.clear()
            for buf in buffer_disc.values():
                for v in buf.values():
                    v.clear()

            if epoch % LOG_INTERVAL == 0 or epoch == 1:
                lifetime = env.lifetime.to(torch.float32).mean().item()
                policy_loss, value_loss = np.mean(policy_loss), np.mean(value_loss)
                
                if multi_critics:
                    rewards = rewards.view(*reward_weights.shape)
                    r = rewards.mean(0).cpu().tolist()
                else:
                    r = rewards.view(-1, rewards.size(-1)).mean(0).cpu().tolist()
                
                if rewards_task is not None:
                    rewards_task = rewards_task.mean(0).cpu().tolist()
                
                print("Epoch: {:4d}, Loss: {:.4f}/{:.4f}, Reward: {}, Lifetime: {:.4f} -- {:.4f}s".format(
                    epoch, policy_loss, value_loss, "/".join(list(map("{:.4f}".format, r))), lifetime, time.time() - tic
                ))
                
                if logger is not None:
                    logger.add_scalar("train/lifetime", lifetime, epoch)
                    logger.add_scalar("train/reward", np.mean(r), epoch)
                    logger.add_scalar("train/loss_policy", policy_loss, epoch)
                    logger.add_scalar("train/loss_value", value_loss, epoch)
                    
                    for name, r_loss in real_losses.items():
                        if r_loss:
                            logger.add_scalar("score_real/{}".format(name), sum(r_loss) / len(r_loss), epoch)
                    for name, f_loss in fake_losses.items():
                        if f_loss:
                            logger.add_scalar("score_fake/{}".format(name), sum(f_loss) / len(f_loss), epoch)
                    
                    if rewards_task is not None:
                        for i in range(len(rewards_task)):
                            logger.add_scalar("train/task_reward_{}".format(i), rewards_task[i], epoch)
                
                for v in real_losses.values():
                    v.clear()
                for v in fake_losses.values():
                    v.clear()

            if ckpt_dir is not None:
                state = None
                if epoch % 50 == 0:
                    state = dict(model=model.state_dict())
                    torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                
                if epoch % training_params.save_interval == 0:
                    if state is None:
                        state = dict(model=model.state_dict())
                    torch.save(state, os.path.join(ckpt_dir, "ckpt-{}".format(epoch)))
                
                if epoch >= training_params.max_epochs:
                    break
            
            tic = time.time()

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
        if settings.ckpt:
            if os.path.isfile(settings.ckpt) or os.path.exists(os.path.join(settings.ckpt, "ckpt")):
                raise ValueError("Checkpoint folder {} exists. Add `--test` option to run test with an existing checkpoint file".format(settings.ckpt))
            import shutil, sys
            os.makedirs(settings.ckpt, exist_ok=True)
            shutil.copy(settings.config, settings.ckpt)
            with open(os.path.join(settings.ckpt, "command_{}.txt".format(time.time())), "w") as f:
                f.write(" ".join(sys.argv))

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

    # Set weights_file --> model save file, settings.ckpt --> folder containing the save file
    weights_file = None
    if os.path.isdir(settings.ckpt):
        weights_file = os.path.join(settings.ckpt, "ckpt")
    else:
        weights_file = settings.ckpt
        settings.ckpt = os.path.dirname(weights_file)

    if settings.test:
        load_success = False
        if settings.ckpt is not None and os.path.exists(weights_file):
            load_success = safe_load_model(model, weights_file)
        if not load_success:
            print(f"WARNING! Unable to Load model (using random weights) from path(exists?{os.path.exists(weights_file)}) : {settings.ckpt}\n")
            os.makedirs(settings.ckpt, exist_ok=True)
        
        test(env, model, settings.ckpt)
    else:
        train(env, model, settings.ckpt, training_params)
