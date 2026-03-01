import os, sys, time, multiprocessing
import numpy as np
import cv2, csv
import torch, matplotlib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' #  Turn off oneDNN custom operations in tensorflow
from torch.utils.tensorboard import SummaryWriter # set env before import to ignore warnings from tensorboard
matplotlib.use('TkAgg') # Ensure GUI backend
import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)
from training_dashboard import TrainingDashboard

# Custom formatting for loss and reward: space for positive, minus for negative
def fmt_signed(val):
    return f" {val:7.4f}" if val >= 0 else f"{val:8.4f}"
def fmt_float(val):
    int_part = int(val)
    frac_part = abs(val - int_part)
    # Format: 4 spaces for integer, dot, 4 for fraction (no leading zero)
    return f"{int_part:4d}"+f"{frac_part:.4f}".replace("0.", ".")

def check_exit(env):
    """Check for keyboard input for exit simulation (works for both render modes)"""
    if env.render_mode == "rgb_array":
        # For offscreen rendering with cv2 window
        # 1 ms wait keeps window responsive; ESC or 'q' to quit test early
        key = cv2.waitKey(1) & 0xFF  # compare using Unicode code, i.e. use ord()
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

# Start dashboard in a separate process
def dashboard_worker(csv_path, window_size=None):
    #time.sleep(5) # to start delayed
    dash = TrainingDashboard(csv_path, window_size)
    dash.run()
#---

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

                # Show via resizable window (save periodic frames)
                #cv2.namedWindow(save_folder, cv2.WINDOW_NORMAL)
                cv2.imshow(save_folder, bgr)
                if steps % 100 == 0:
                    cv2.imwrite(os.path.join(save_dir, save_folder, f"frame_{steps:05d}.png"), bgr)
        
            if check_exit(env):
                print(f"User requested exit!")
                break
        
        # Print progress every 100 steps
        if steps % 100 == 0:
            #render_file = os.path.join(save_dir, save_folder, f"frame_{steps:05d}.png")
            print(title)
            #print(f"... check save_path (exists?{os.path.exists(render_file)}) : {render_file}")
        
        steps += 1
    
    env.close()
    print(f"Test completed: {steps} steps")
# ---

def train(env, model, ckpt_dir, training_params, init_epoch=0):
    """Train the model using PPO"""
    if ckpt_dir is not None:
        # Default log_dir is 'ckpt_dir/CURRENT_DATETIME_HOSTNAME'
        logger = SummaryWriter(ckpt_dir)
        
        # --- CSV Logger Setup ---
        csv_log_path = os.path.join(ckpt_dir, "training_metrics.csv")
        
        # Define CSV columns based on metrics being tracked
        csv_columns = ["epoch", "lifetime", "reward_mean", "policy_loss", "value_loss"]
        
        # Add discriminator score columns if present
        if env.discriminators:
            for name in env.discriminators.keys():
                csv_columns.append(f"score_real_{name}")
                csv_columns.append(f"score_fake_{name}")
        
        # Add discriminator reward columns if present
        if env.discriminators:
            for name in env.discriminators.keys():
                csv_columns.append(f"disc_reward_{name}")
        
        # Add task reward columns if present
        if env.rew_dim > 0:
            for i in range(env.rew_dim):
                csv_columns.append(f"task_reward_{i}")
        
        # Handle resume: if init_epoch > 0, truncate CSV to that epoch
        if init_epoch > 0 and os.path.exists(csv_log_path):
            # Read existing CSV and keep only rows with epoch < init_epoch
            existing_rows = []
            with open(csv_log_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row['epoch']) <= init_epoch:
                        existing_rows.append(row)
            
            # Rewrite CSV with kept rows
            with open(csv_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerows(existing_rows)
            csv_mode = 'a'
        else:
            csv_mode = 'w'
        
        # Initialize CSV file with header if needed
        if not os.path.exists(csv_log_path) or csv_mode == 'w':
            with open(csv_log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_columns)
                writer.writeheader()

        # live dashboard to view training metrics on last save_interval*2 window
        dashboard_proc = multiprocessing.Process(target=dashboard_worker, args=(csv_log_path, training_params.save_interval *2))
        dashboard_proc.start()
    else:
        logger = None
        csv_log_path = None
        csv_columns = None

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
    # for each epoch --> num_steps =  num_env * horizon; 
    # remove horizon if u want max_epochs to be affiliated with (max_steps per env) as per src code
    steps_per_epoch = training_params.num_envs * HORIZON
    
    print(f"$ Starting Training Module for epochs set: {init_epoch+1} to {init_epoch+training_params.max_epochs} with steps_per_epoch: {steps_per_epoch} ...")
    if env.verbose:
        print(f"  ob_horizon={env.ob_horizon}  disc.ob_horizon(s)={[d.ob_horizon for d in env.discriminators.values()]}")
        print(f"  NOTE: end_frame = ob_seq_lens-1 ensures GRU reads last VALID frame (not zero-padded slot)")
        
    epoch = 0
    model.eval()
    env.train()
    env.reset()    
    tic = time.time()
    total_steps = 0

    while total_steps < training_params.max_epochs * steps_per_epoch:
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

        #if env.verbose: print(f"[Training Counter] --> epochs:{epoch} --> total_steps:{total_steps}")
        total_steps += training_params.num_envs

        if len(buffer["s"]) == HORIZON: # wait till buffer caches horizon number of steps/obs (kinda like dfs with max depth=horizon)
            disc_data = []
            ob_seq_lens = torch.cat(buffer["ob_seq_len"]).to(env.device)
            ob_seq_end_frames = ob_seq_lens - 1
            
            if env.discriminators:
                with torch.no_grad():
                    for name, data in buffer_disc.items():
                        disc = model.discriminators[name]
                        fake = torch.cat(data["fake"]).to(env.device)
                        real_ = torch.cat(data["real"]).to(env.device)
                        # FIX: end_frame must be ob_seq_lens - 1, NOT ob_seq_lens
                        end_frame = ob_seq_end_frames   # ← was ob_seq_lens (off-by-one BUG)

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

                        # ---- DEBUG: verify the fix by checking what the GRU will actually read ----
                        if env.verbose and (epoch % LOG_INTERVAL == 0 or epoch <= 2):
                            # Sample a few instances to inspect
                            n_check = min(8, fake.size(0))
                            ef_check = end_frame[:n_check]
                            # Grab the actual vectors the GRU will output at (before normalisation)
                            fake_at_ef = fake[:n_check][
                                torch.arange(n_check, device=env.device),
                                ef_check.clamp(max=fake.size(1)-1)
                            ]
                            real_at_ef = real[:n_check][
                                torch.arange(n_check, device=env.device),
                                ef_check.clamp(max=real.size(1)-1)
                            ]
                            fake_norm = fake_at_ef.norm(dim=-1).mean().item()
                            real_norm = real_at_ef.norm(dim=-1).mean().item()
                            ef_mean   = ef_check.float().mean().item()
                            ef_min    = ef_check.min().item()
                            ef_max    = ef_check.max().item()
                            print(f"  [DISC-DBG][{name}] end_frame: mean={ef_mean:.1f} [{ef_min}-{ef_max}]"
                                  f" | fake@ef_norm={fake_norm:.4f} | real@ef_norm={real_norm:.4f}"
                                  f" | seq_dim={fake.size(1)}")
                            if fake_norm < 1e-3:
                                print(f"  [DISC-DBG][{name}] ⚠️  fake@end_frame is near-ZERO — reading zero-padded slot!")
                            else:
                                print(f"  [DISC-DBG][{name}] ✓  fake@end_frame has real content (norm={fake_norm:.4f})")

                model.train()
                n_samples = 0
                for name, disc, ref, ob, seq_end_frame_ in disc_data:
                    real_loss = real_losses[name]
                    fake_loss = fake_losses[name]
                    opt = disc_optimizer[name]
                    
                    if len(ref) != n_samples:
                        n_samples = len(ref)
                        idx = torch.randperm(n_samples, device=env.device)
                    
                    # Clamp batch size to available samples (e.g. n_envs=1 → only 8 samples)
                    eff_disc_batch = min(BATCH_SIZE, n_samples)
                    if eff_disc_batch == 0:
                        continue
                    
                    for batch in range(n_samples // eff_disc_batch):
                        sample = idx[batch * eff_disc_batch:(batch + 1) * eff_disc_batch]
                        r = ref[sample]
                        f = ob[sample]
                        seq_end_frame = seq_end_frame_[sample]

                        score_r = disc(r, seq_end_frame, normalize=False)
                        score_f = disc(f, seq_end_frame, normalize=False)

                        loss_r = torch.nn.functional.relu(1 - score_r).mean()
                        loss_f = torch.nn.functional.relu(1 + score_f).mean()

                        if env.verbose and (epoch % LOG_INTERVAL == 0 and batch == 0):
                            print(f"\n--- [DEBUG] Discriminator Feature Norms ({name}) ---")
                            
                            # Flatten batch and sequence dimensions: [Batch*Seq, Features]
                            r_flat = r.reshape(-1, r.shape[-1])
                            f_flat = f.reshape(-1, f.shape[-1])
                            
                            # Calculate global per-feature mean absolute difference (1D tensor now)
                            diff = torch.abs(r_flat - f_flat).mean(dim=0)
                            
                            # Print the top 5 features where the discriminator finds the biggest gap
                            top_diffs, top_idx = torch.topk(diff, 5)
                            for i in range(5):
                                idx = top_idx[i].item()
                                print(f" Feature {idx:3d}: Real_Avg={r_flat[:, idx].mean():.4f}, Fake_Avg={f_flat[:, idx].mean():.4f}, Diff={top_diffs[i]:.4f}")
                            print("---------------------------------------------------\n")

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
                
                # --- Store individual discriminator rewards for logging ---
                disc_rewards = {}
                for name, disc, _, ob, seq_end_frame in disc_data:
                    r = (disc(ob, seq_end_frame, normalize=False).clamp_(-1, 1)
                         .mean(-1, keepdim=True))
                    disc_rewards[name] = r  # Store for logging
                    if rewards is None:
                        rewards = r
                    else:
                        rewards[:, env.discriminators[name].id] = r.squeeze_(-1)
                    
                    # ---- DEBUG: show discriminator score distributions ----
                    if env.verbose and (epoch % LOG_INTERVAL == 0 or epoch <= 2):
                        r_flat = r.view(-1)
                        n_pos  = (r_flat > 0).sum().item()
                        n_neg  = (r_flat < 0).sum().item()
                        print(f"  [DISC-SCORE][{name}] reward: mean={r_flat.mean():.4f}"
                              f"  std={r_flat.std():.4f}"
                              f"  pos={n_pos}({100*n_pos/len(r_flat):.0f}%)"
                              f"  neg={n_neg}({100*n_neg/len(r_flat):.0f}%)")
                
                # ---- DEBUG: physics / lifetime health ----
                if env.verbose and (epoch % LOG_INTERVAL == 0 or epoch <= 2):
                    n_terminated = terminate.sum().item()
                    n_total      = terminate.numel()
                    root_h_mean  = env.root_tensor[:, 2].mean().item()
                    root_h_min   = env.root_tensor[:, 2].min().item()
                    print(f"  [PHYS-DBG] terminate={n_terminated}/{n_total}"
                          f"  root_h: mean={root_h_mean:.3f}  min={root_h_min:.3f}"
                          f"  lifetime_mean={env.lifetime.float().mean():.1f}")
                
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

                # ---- DEBUG: advantage health (should span a reasonable range, not all same sign) ----
                if env.verbose and (epoch % LOG_INTERVAL == 0 or epoch <= 2):
                    adv_flat = advantages.view(-1)
                    print(f"  [ADV-DBG]  advantages: mean={adv_flat.mean():.4f}"
                          f"  std={adv_flat.std():.4f}"
                          f"  min={adv_flat.min():.4f}  max={adv_flat.max():.4f}"
                          f"  pos%={(adv_flat>0).float().mean()*100:.0f}%")

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
            update_epoch = init_epoch + epoch
            model.train()
            policy_loss, value_loss = [], []
            
            # Clamp batch size to available samples (e.g. n_envs=1, horizon=8 → 8 samples total)
            eff_batch = min(BATCH_SIZE, n_samples)
            if eff_batch == 0:
                print(f"[WARNING] Epoch {update_epoch}: no samples for PPO update (n_samples={n_samples}), skipping.")
            else:
                for _ in range(OPT_EPOCHS):
                    idx = torch.randperm(n_samples)
                    for batch in range(n_samples // eff_batch):
                        sample = idx[eff_batch * batch: eff_batch * (batch + 1)]
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

                        # NaN guard: skip this batch if loss is invalid
                        if torch.isnan(pg_loss) or torch.isnan(vf_loss) or \
                           torch.isinf(pg_loss) or torch.isinf(vf_loss):
                            print(f"[WARNING] Epoch {update_epoch} batch {batch}: "
                                  f"NaN/Inf in loss (pg={pg_loss.item():.4f}, vf={vf_loss.item():.4f}), skipping.")
                            optimizer.zero_grad()
                            continue

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

            if epoch % LOG_INTERVAL == 0 or (init_epoch == 0 and epoch == 1):
                lifetime = env.lifetime.to(torch.float32).mean().item()
                policy_loss = np.mean(policy_loss) if policy_loss else float('nan')
                value_loss  = np.mean(value_loss)  if value_loss  else float('nan')
                
                if multi_critics:
                    rewards = rewards.view(*reward_weights.shape)
                    r = rewards.mean(0).cpu().tolist()
                else:
                    r = rewards.view(-1, rewards.size(-1)).mean(0).cpu().tolist()
                
                if rewards_task is not None:
                    rewards_task = rewards_task.mean(0).cpu().tolist()
                
                # --- Compute individual discriminator reward means for logging ---
                disc_reward_means = {}
                if env.discriminators:
                    for name, disc_r in disc_rewards.items():
                        disc_reward_means[name] = disc_r.mean().item()
                
                if env.verbose: print()
                print("Epoch: {:4d}, Loss: policy={} / value={}, Reward: {}, Lifetime: {} -- {:.4f}s{}".format(
                    update_epoch,
                    fmt_signed(policy_loss),
                    fmt_signed(value_loss),
                    "/".join([fmt_signed(x) for x in r]), # stringify rewards
                    fmt_float(lifetime),
                    time.time() - tic,
                    " @ckpt-save!" if epoch % training_params.save_interval == 0 else ""
                ))
                
                if logger is not None:
                    logger.add_scalar("train/lifetime", lifetime, update_epoch)
                    logger.add_scalar("train/reward", np.mean(r), update_epoch)
                    logger.add_scalar("train/loss_policy", policy_loss, update_epoch)
                    logger.add_scalar("train/loss_value", value_loss, update_epoch)
                    
                    for name, r_loss in real_losses.items():
                        if r_loss:
                            logger.add_scalar("score_real/{}".format(name), sum(r_loss) / len(r_loss), update_epoch)
                    for name, f_loss in fake_losses.items():
                        if f_loss:
                            logger.add_scalar("score_fake/{}".format(name), sum(f_loss) / len(f_loss), update_epoch)
                    
                    # --- Log individual discriminator rewards to TensorBoard ---
                    if env.discriminators:
                        for name, disc_r_mean in disc_reward_means.items():
                            logger.add_scalar("disc_reward/{}".format(name), disc_r_mean, update_epoch)
                    
                    # --- Log individual task rewards to TensorBoard ---
                    if rewards_task is not None:
                        for i in range(len(rewards_task)):
                            logger.add_scalar("train/task_reward_{}".format(i), rewards_task[i], update_epoch)
                
                # --- CSV Logger: Write metrics to CSV file ---
                if csv_log_path is not None:
                    csv_row = {
                        "epoch": update_epoch,
                        "lifetime": lifetime,
                        "reward_mean": np.mean(r),
                        "policy_loss": policy_loss,
                        "value_loss": value_loss
                    }
                    
                    # Add discriminator scores
                    if env.discriminators:
                        for name in env.discriminators.keys():
                            r_loss = real_losses.get(name, [])
                            f_loss = fake_losses.get(name, [])
                            csv_row[f"score_real_{name}"] = sum(r_loss) / len(r_loss) if r_loss else float('nan')
                            csv_row[f"score_fake_{name}"] = sum(f_loss) / len(f_loss) if f_loss else float('nan')
                    
                    # Add discriminator rewards
                    if env.discriminators:
                        for name in env.discriminators.keys():
                            csv_row[f"disc_reward_{name}"] = disc_reward_means.get(name, float('nan'))
                    
                    # Add task rewards
                    if rewards_task is not None:
                        for i in range(len(rewards_task)):
                            csv_row[f"task_reward_{i}"] = rewards_task[i]
                    
                    # Append to CSV file
                    with open(csv_log_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_columns)
                        writer.writerow(csv_row)
                
                for v in real_losses.values():
                    v.clear()
                for v in fake_losses.values():
                    v.clear()

            if ckpt_dir is not None:
                state = None
                if epoch % (training_params.save_interval/10) == 0: # overwrite latest model frequently
                    state = dict(model=model.state_dict())
                    torch.save(state, os.path.join(ckpt_dir, "ckpt"))
                
                if epoch % training_params.save_interval == 0: # keep seperate model save file at save interval
                    if state is None:
                        state = dict(model=model.state_dict()) # if loaded prev model, save with epoch increment
                    torch.save(state, os.path.join(ckpt_dir, "ckpt-{}".format(update_epoch)))
                
                if epoch >= training_params.max_epochs:
                    break
            
            tic = time.time()
    if logger:
        logger.flush() # make sure that all pending events have been written to disk.
        logger.close()
# ---