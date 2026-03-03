"""
MuJoCo-compatible environment for CompositeMotion
Converted from IsaacGym version
"""
from typing import Dict, Tuple, Optional, List
import torch, os
import numpy as np

from ref_motion import ReferenceMotion
from env_mujoco import MujocoEnv, DiscriminatorConfig

class ICCGANHumanoid(MujocoEnv):
    """ICCGAN Humanoid environment for MuJoCo"""
    
    CHARACTER_MODEL = os.path.join("assets", "humanoid.xml")
    CONTACTABLE_LINKS = ["right_foot", "left_foot"]
    UP_AXIS = 2
    
    GOAL_DIM = 0
    GOAL_REWARD_WEIGHT = None
    ENABLE_GOAL_TIMER = False
    GOAL_TENSOR_DIM = None
    
    OB_HORIZON = 4
    KEY_LINKS = None  # All links
    PARENT_LINK = None  # root link
    TERMINATION_HEIGHT_THRESHOLD = 0.15  # Default: 0.15 (meters) --> no need to reduce as we already grace step impleted
    TERMINATION_GRACE_STEPS = 1          # ADDED: Require N consecutive frames below threshold

    def __init__(self, *args,
        motion_file: str,
        discriminators: Dict[str, DiscriminatorConfig],
        **kwargs
    ):
        self.contactable_links_names = kwargs.get("contactable_links", self.CONTACTABLE_LINKS)
        self.goal_reward_weight = kwargs.get("goal_reward_weight", self.GOAL_REWARD_WEIGHT)
        self.enable_goal_timer = kwargs.get("enable_goal_timer", self.ENABLE_GOAL_TIMER)
        self.goal_tensor_dim = kwargs.get("goal_tensor_dim", self.GOAL_TENSOR_DIM)
        self.ob_horizon = kwargs.get("ob_horizon", self.OB_HORIZON)
        self.key_links_names = kwargs.get("key_links", self.KEY_LINKS)
        self.parent_link_name = kwargs.get("parent_link", self.PARENT_LINK)
        self.term_height = kwargs.get("term_height", self.TERMINATION_HEIGHT_THRESHOLD)
        self.grace_steps  = kwargs.get("grace_steps", self.TERMINATION_GRACE_STEPS)       # no.of consecutive frames required
        # Phase-conditioned observations
        self.use_phase_obs = kwargs.pop("use_phase_obs", False)
        self.phase_period  = float(kwargs.pop("phase_period", 1.0))
        
        # initialize max_ob_horizon first itself, so base class can setup_state_spaces() correctly
        self.max_ob_horizon = self.ob_horizon + 1
        for config in discriminators.values(): # iterate through discriminator configs
            if config.ob_horizon is None:
                config.ob_horizon = self.ob_horizon + 1 # update configs with empty ob_horizon
            self.max_ob_horizon = max(self.max_ob_horizon, config.ob_horizon)
        
        # Setup state history --> will be done create_tensors(), that will be called from super.__init__
        super().__init__(*args, **kwargs)
        
        n_envs = self.n_envs
        n_links = self.model.nbody - 1 # Exclude world body 
        n_dofs = self.model.nv
        
        # Setup contactable links
        if self.contactable_links_names is None:
            self.contactable_links = None
        else:
            contact = np.full((n_envs, n_links), float(self.term_height))
            if not isinstance(self.contactable_links_names, dict):
                contactable_links_dict = {link: -10000 for link in self.contactable_links_names}
            else:
                contactable_links_dict = self.contactable_links_names
            
            for link_name, h in contactable_links_dict.items():
                if link_name in self.body_names:
                    lid = self.body_names.index(link_name)  # Index in body_names (includes world)
                    # CHANGED: Adjust index to exclude world body
                    if lid > 0:  # Skip world body
                        contact[:, lid - 1] = h  # Subtract 1 to get link_tensor index
                    else:
                        print(f"[Warning] Skipping world body in contactable_links")
                else:
                    print(f"[Warning] Unrecognized contactable link {link_name}")
            
            self.contactable_links = torch.tensor(contact, dtype=torch.float32, device=self.device)
        
        # Setup reward weights
        reward_weights = None
        if self.goal_reward_weight is not None:
            reward_weights = torch.empty((n_envs, self.rew_dim), dtype=torch.float32, device=self.device)
            if not hasattr(self.goal_reward_weight, "__len__"):
                self.goal_reward_weight = [self.goal_reward_weight]
            assert self.rew_dim == len(self.goal_reward_weight)
            for i, w in enumerate(self.goal_reward_weight):
                reward_weights[:, i] = w
        
        # Setup discriminators
        n_comp = len(discriminators) + self.rew_dim
        if n_comp > 1:
            self.reward_weights = torch.zeros((n_envs, n_comp), dtype=torch.float32, device=self.device)
            weights = [disc.weight for _, disc in discriminators.items() if disc.weight is not None]
            total_weights = sum(weights) if weights else 0
            assert total_weights <= 1, "Discriminator weights must not be greater than 1."
            n_unassigned = len(discriminators) - len(weights)
            rem = 1 - total_weights
            for disc in discriminators.values():
                if disc.weight is None:
                    disc.weight = rem / n_unassigned
                elif n_unassigned == 0:
                    disc.weight /= total_weights
        else:
            self.reward_weights = None
        
        self.discriminators = dict()
        # validate discriminator configs and update with links
        for i, (id, config) in enumerate(discriminators.items()):
            # Setup key links
            if config.key_links is None:
                key_links = None
            else:
                key_links = []
                for link_name in config.key_links:
                    if link_name in self.body_names:
                        lid = self.body_names.index(link_name)
                        key_links.append(lid)
                    else:
                        print(f"[Warning] Unrecognized key link {link_name}")
                key_links = sorted(key_links) if key_links else None
            
            # Setup parent link
            if config.parent_link is None:
                parent_link = None
            else:
                if config.parent_link in self.body_names:
                    parent_link = self.body_names.index(config.parent_link)
                else:
                    print(f"[Warning] Unrecognized parent link {config.parent_link}")
                    parent_link = None
            
            config.parent_link = parent_link
            config.key_links = key_links
            
            if config.motion_file is None:
                config.motion_file = motion_file
            config.id = i
            config.name = id
            self.discriminators[id] = config
            
            if self.reward_weights is not None:
                self.reward_weights[:, i] = config.weight
        
        if self.reward_weights is None:
            self.reward_weights = torch.ones((n_envs, 1), dtype=torch.float32, device=self.device)
        elif self.rew_dim > 0 and reward_weights is not None:
            if self.rew_dim > 1:
                self.reward_weights *= (1 - reward_weights.sum(dim=-1, keepdim=True))
            else:
                self.reward_weights *= (1 - reward_weights)
            self.reward_weights[:, -self.rew_dim:] = reward_weights
        
        self.info["ob_seq_lens"] = torch.zeros_like(self.lifetime)  # dummy result
        self.goal_dim = self.GOAL_DIM + (2 if self.use_phase_obs else 0) # phase encoding adds 2 dims to the goal slice (sin/cos of phase)
        self.state_dim = (self.ob_dim - self.goal_dim) // self.ob_horizon
        
        if self.discriminators:
            self.info["disc_obs"] = self.observe_disc(self.state_hist)  # dummy result
            self.info["disc_obs_expert"] = self.info["disc_obs"]  # dummy result
            self.disc_dim = {
                name: ob.size(-1)
                for name, ob in self.info["disc_obs"].items()
            }
        else:
            self.disc_dim = {}
        
        # Load reference motion
        self.build_motion_lib(motion_file)
        if "phase_period" not in kwargs and self.ref_motion.period > 0: # overwrite if not passed as training args
            print(f"Ref-Motion :: Phase Input overwrites phase_period: {self.phase_period} -> {self.ref_motion.period}\n")
            self.phase_period = self.ref_motion.period
            
        self.sampling_workers = []
        self.real_samples = []
    
    def build_motion_lib(self, motion_file):
        """Build reference motion library"""
        self.ref_motion = ReferenceMotion(
            motion_file=motion_file, 
            character_model=self.character_model, 
            device=self.device
        )
        # @overwrite prev calc in base class for refernce motion data
        if self.ref_motion.fps > 0: 
            print(f"Build Ref-Motion overwrites fps: {self.fps} -> {self.ref_motion.fps}")
            self.fps = self.ref_motion.fps
            self.frameskip = int(self.run_speed/self.fps)
            self.step_time = 1.0 / self.fps

    def reset_envs(self, env_ids):  # overwrite for extra handling
        super().reset_envs(env_ids)          # calls init_state → sets env_motion_times
        self.state_hist[:, env_ids] = 0.0
        self.terminate_counter[env_ids] = 0 
        # Phase Input: seed phase from the motion time sampled in init_state
        if self.use_phase_obs:
            self.env_phase[env_ids] = (
                self.env_motion_times[env_ids] % self.phase_period
            ) / self.phase_period

    def reset_done(self):
        """Reset done environments and return observations with info"""
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def step(self, actions):
        """Step environment and update discriminator observations"""
        obs, rews, terminated, truncated, info = super().step(actions)
        if self.discriminators and self.training:
            info["disc_obs"] = self.observe_disc(self.state_hist)
            info["disc_obs_expert"] = self.fetch_real_samples()
        return obs, rews, terminated, truncated, info
    
    def create_tensors(self):
        """Create tensors with character-specific info"""
        super().create_tensors()
        
        # n_links should be model.nbody - 1 (excluding world body)
        n_links = self.model.nbody -1
        n_dofs = self.model.nv
        
        # Character-specific tensors
        self.root_pos = self.root_tensor[:, :3]
        self.root_orient = self.root_tensor[:, 3:7]
        self.root_lin_vel = self.root_tensor[:, 7:10]
        self.root_ang_vel = self.root_tensor[:, 10:13]
        self.char_root_tensor = self.root_tensor
        
        self.link_pos = self.link_tensor[:, :, :3]
        self.link_orient = self.link_tensor[:, :, 3:7]
        self.link_lin_vel = self.link_tensor[:, :, 7:10]
        self.link_ang_vel = self.link_tensor[:, :, 10:13]
        self.char_link_tensor = self.link_tensor
        
        self.joint_pos = self.joint_tensor[:, :, 0]
        self.joint_vel = self.joint_tensor[:, :, 1]
        self.char_joint_tensor = self.joint_tensor
        
        self.char_contact_force_tensor = self.contact_force_tensor
        # Grace counter for termination (tracks consecutive low-height frames)
        self.terminate_counter = torch.zeros((self.n_envs,), dtype=torch.int32, device=self.device)

        # Phase-conditioned observation tensors
        if self.use_phase_obs:
            # env_phase ∈ [0, 1): normalised position within one motion cycle
            self.env_phase = torch.zeros((self.n_envs,), dtype=torch.float32, device=self.device)
            # env_motion_times: start time sampled from ref motion, used to seed phase on reset
            self.env_motion_times = torch.zeros((self.n_envs,), dtype=torch.float32, device=self.device)

        # Setup state history (NOTE: self.max_ob_horizon must be initilized before)
        self.state_hist = torch.zeros((self.max_ob_horizon, self.n_envs, n_links * 13),
            dtype=torch.float32, device=self.device)
        
        # Setup key links
        if self.key_links_names is None:
            self.key_links = list(range(n_links))
        else:
            self.key_links = []
            for link_name in self.key_links_names:
                if link_name in self.body_names:
                    lid = self.body_names.index(link_name)        
                    if lid > 0:  # Adjust for world body being at index 0 --> Skip if it's world body
                        self.key_links.append(lid - 1)  # Subtract 1 to account for world body offset
        
        # Setup parent link
        if self.parent_link_name is None:
            self.parent_link = None
        else:
            if self.parent_link_name in self.body_names:
                self.parent_link = self.body_names.index(self.parent_link_name)
            else:
                self.parent_link = None
        
        # Goal tensor
        if self.goal_tensor_dim:
            try:
                self.goal_tensor = [
                    torch.zeros((self.n_envs, dim), dtype=torch.float32, device=self.device)
                    for dim in self.goal_tensor_dim
                ]
            except TypeError:
                self.goal_tensor = torch.zeros((self.n_envs, self.goal_tensor_dim), 
                    dtype=torch.float32, device=self.device)
        else:
            self.goal_tensor = None
        
        self.goal_timer = torch.zeros((self.n_envs,), dtype=torch.int32, device=self.device) \
            if self.enable_goal_timer else None
    
    def init_state(self, env_ids):
        """Initialize state from reference motion"""
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        
        # Adjust for ground height if needed
        ground_height = self.ground_height(ref_link_tensor[:, 0, :3], env_ids)
        if ground_height is not None:
            ref_link_tensor[:, :, 2] += ground_height.unsqueeze(1)
        
        # Phase Input: store sampled motion times so reset_envs can seed phase from them
        if self.use_phase_obs:
            self.env_motion_times[env_ids] = torch.from_numpy(
                np.array(motion_times, dtype=np.float32)
            ).to(self.device)
        
        return ref_link_tensor, ref_joint_tensor
    
    def ground_height(self, p, env_ids=None):
        """Get ground height at position"""
        return None
    
    def observe(self, env_ids=None):
        """Observe with ICCGAN observation function"""
        self.ob_seq_lens = torch.clamp(self.lifetime + 1, max=self.ob_horizon)
        n_envs = self.n_envs
        
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[:-1] = self.state_hist[1:].clone()
            self.state_hist[-1] = self.char_link_tensor.view(n_envs, -1)
            env_ids = None
            # Phase Input: advance phase for ALL envs each simulation step (full-env call only)
            # Subset calls (reset envs) already have correct phase set in reset_envs()
            if self.use_phase_obs:
                self.env_phase = (self.env_phase + self.step_time / self.phase_period) % 1.0
        else:
            n_envs = len(env_ids)
            self.state_hist[:-1, env_ids] = self.state_hist[1:, env_ids].clone()
            self.state_hist[-1, env_ids] = self.char_link_tensor[env_ids].view(n_envs, -1)
        
        if self.verbose and self.simulation_step % 100 == 0:
            print(f"\n[Step {self.simulation_step}] OBS check:")
            print(f"  Shape: {self.state_hist.shape}")
            print(f"  Range: [{self.state_hist.min():.4f}, {self.state_hist.max():.4f}]")
            print(f"  NaN/Inf: {torch.isnan(self.state_hist).any()} / {torch.isinf(self.state_hist).any()}")
        return self._observe_iccgan(env_ids)
    
    def _observe_iccgan(self, env_ids=None):
        """Internal ICCGAN observation"""
        if env_ids is None:
            ground_height = self.ground_height(self.state_hist[-1, :, :3])
            obs = observe_iccgan_safe(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link,
                ground_height=ground_height
            ).flatten(start_dim=1)
            # Phase Input: append (sin, cos) phase encoding — bounded [-1,1], no normalisation needed
            if self.use_phase_obs:
                phase_enc = torch.stack([
                    torch.sin(2.0 * np.pi * self.env_phase),
                    torch.cos(2.0 * np.pi * self.env_phase),
                ], dim=-1)
                obs = torch.cat([obs, phase_enc], dim=-1)
        else:
            ground_height = self.ground_height(self.state_hist[-1, env_ids, :3], env_ids)
            obs = observe_iccgan_safe(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.key_links, self.parent_link,
                ground_height=ground_height
            ).flatten(start_dim=1)
            # Phase Input: append phase encoding for subset
            if self.use_phase_obs:
                phase_enc = torch.stack([
                    torch.sin(2.0 * np.pi * self.env_phase[env_ids]),
                    torch.cos(2.0 * np.pi * self.env_phase[env_ids]),
                ], dim=-1)
                obs = torch.cat([obs, phase_enc], dim=-1)
        return obs
    
    def observe_disc(self, state):
        """Observe for discriminator"""
        # FIX (secondary bug): self.info["ob_seq_lens"] is set in reset_done() and is NOT
        # updated when step() calls observe() internally. By the time observe_disc() runs
        # (inside step(), after observe()), self.ob_seq_lens has been refreshed but
        # self.info["ob_seq_lens"] still holds the value from the *previous* reset_done call.
        # Fix: read self.ob_seq_lens directly — it is always current.
        seq_len = self.ob_seq_lens   # ← was self.info["ob_seq_lens"] (stale reference)
        res = dict()

        if self.verbose and self.simulation_step % 100 == 0:
            print(f"\n[Step {self.simulation_step}] observe_disc() check:")

        if torch.is_tensor(state):
            # Fake — build discriminator observations from current state history
            for id, disc in self.discriminators.items():
                ob = observe_iccgan_safe(
                    state[-disc.ob_horizon:], seq_len,
                    disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos
                )
                res[id] = ob
                # ---- DEBUG: check that we're not feeding all-zeros into the disc ----
                if self.verbose and self.simulation_step % 100 == 0:
                    # ob shape: [n_envs, disc.ob_horizon, features]
                    # Valid frames are packed at positions [0 .. seq_len-1].
                    # The discriminator will read the GRU output at position (seq_len-1),
                    # so check that position is non-zero.
                    ef = (seq_len - 1).clamp(max=ob.size(1)-1)
                    norm_at_ef = ob[torch.arange(ob.size(0), device=ob.device), ef].norm(dim=-1).mean().item()
                    print(f"    [OBS-DBG][{id}] disc ob@(seq_len-1): norm={norm_at_ef:.4f}"
                          f"  seq_len=[{seq_len.min()}-{seq_len.max()}]  ob_shape={tuple(ob.shape)}")
            return res
        else:
            # Real — state is a dict of pre-built tensors (from fetch_real_samples)
            seq_len_ = dict()
            for disc_name, s in state.items():
                disc = self.discriminators[disc_name]
                res[disc_name] = observe_iccgan_safe(
                    s[-disc.ob_horizon:], seq_len,
                    disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos
                )
                seq_len_[disc_name] = seq_len
            return res, seq_len_
    
    def fetch_real_samples(self):
        """Fetch real samples from reference motion"""
        if not self.real_samples:
            # Generate samples directly without multiprocessing for simplicity
            obs_list = []
            for _ in range(128):
                dt = self.step_time
                ob_horizon = max(disc.ob_horizon for disc in self.discriminators.values())
                
                motion_ids, motion_times0 = self.ref_motion.sample(self.n_envs, truncate_time=dt*(ob_horizon-1))
                motion_ids = np.tile(motion_ids, ob_horizon)
                motion_times = np.concatenate([motion_times0 + dt*i for i in range(ob_horizon)])
                
                link_tensor = self.ref_motion.state(motion_ids, motion_times, with_joint_tensor=False)
                samples = link_tensor.view(ob_horizon, self.n_envs, -1)
                
                # Create discriminator observations
                disc_obs = {}
                for name, disc in self.discriminators.items():
                    ob = observe_iccgan_safe(samples[-disc.ob_horizon:], None, disc.key_links, disc.parent_link,
                        include_velocity=False, local_pos=disc.local_pos)
                    disc_obs[name] = ob.cpu()
                
                obs_list.append(disc_obs)
            
            self.real_samples = obs_list
        
        return self.real_samples.pop()
    
    def termination_check(self):
        """Check termination conditions"""
        if self.contactable_links is None:
            return torch.zeros_like(self.done)
        
        # Check contact forces — requires contact_force_tensor to be current (updated in _sync_state_from_mujoco)
        contacted = torch.any(self.char_contact_force_tensor.abs() > 1., dim=-1)
        
        # Check height
        ground_height = self.ground_height(self.char_root_tensor[:, :3])
        if ground_height is None:
            low_threshold = self.contactable_links
        else:
            low_threshold = self.contactable_links + ground_height.unsqueeze(1)
        
        too_low = self.link_pos[..., self.UP_AXIS] < low_threshold

        # --- grace logic: require N consecutive frames of (contacted & too_low) before terminating ---
        # cond_per_link: shape [n_envs, n_links]
        cond_per_link = torch.logical_and(contacted, too_low)

        # cond_env is True if any link in that env meets the cond
        cond_env = torch.any(cond_per_link, dim=-1)  # shape [n_envs]

        # increment where cond_env True, reset where False
        # make sure dtype matches terminate_counter
        incr = cond_env.to(self.terminate_counter.dtype)
        # increment
        self.terminate_counter = self.terminate_counter + incr
        # reset counters where condition false
        self.terminate_counter *= incr

        # terminate when counter >= grace AND lifetime > 1 (preserves earlier lifetime check)
        terminate = (self.terminate_counter >= int(self.grace_steps)) & (self.lifetime > 1)

        # ---- DEBUG: show why/how often we terminate ----
        if self.verbose and self.simulation_step % 100 == 0:
            n_contact  = contacted.sum().item()
            n_too_low  = too_low.any(-1).sum().item()
            n_term     = terminate.sum().item()
            root_h_min = self.link_pos[:, 0, self.UP_AXIS].min().item()  # pelvis
            cf_max     = self.char_contact_force_tensor.abs().max().item()
            print(f"  [TERM-DBG] step={self.simulation_step:5d} | "
                f"contacted={n_contact}/{self.n_envs}  too_low={n_too_low}/{self.n_envs}  "
                f"terminate={n_term} | root_h_min={root_h_min:.3f}  cf_max={cf_max:.2f}")
        
        return terminate
    
    def reward(self):
        """Optional feature for ICCGAN - use default for pretrained weights of composite_motion"""
        # defualt 0 task rewards (rew_dim = 0) i.e. trained only on 1 discriminator
        return super().reward()


class ICCGANHumanoidTarget(ICCGANHumanoid):
    """ICCGAN Humanoid with target reaching - for locomotion tasks"""
    
    GOAL_DIM = 4    # direction(2) + distance(1) + speed(1), per paper appendix B.2
    GOAL_REWARD_WEIGHT = [0.5]
    ENABLE_GOAL_TIMER = True
    GOAL_TENSOR_DIM = 2
    
    def __init__(self, *args,
        goal_radius: float = 0.5,
        sp_lower_bound: float = 1.2,
        sp_upper_bound: float = 1.5,
        goal_timer_range: Tuple[int, int] = (90, 150),
        goal_sp_mean: float = 1.0,
        goal_sp_std: float = 0.25,
        goal_sp_min: float = 0.0,
        goal_sp_max: float = 1.25,
        **kwargs
    ):
        self.goal_radius = goal_radius
        self.sp_lower_bound = sp_lower_bound
        self.sp_upper_bound = sp_upper_bound
        self.goal_timer_range = goal_timer_range
        self.goal_sp_mean = goal_sp_mean
        self.goal_sp_std = goal_sp_std
        self.goal_sp_min = goal_sp_min
        self.goal_sp_max = goal_sp_max
        
        # Extract n_envs and compute_device from args/kwargs to initialize buffers early
        n_envs = kwargs.get('n_envs', args[0] if len(args) > 0 else 1)
        compute_device = kwargs.get('compute_device', 0)
        device = torch.device(f"cuda:{compute_device}" if torch.cuda.is_available() and compute_device >= 0 else "cpu")
        
        # Initialize goal positions
        self.goal_pos = torch.zeros((n_envs, 2), dtype=torch.float32, device=device)
        self.goal_speed = torch.ones((n_envs,), dtype=torch.float32, device=device)

        # need above params for reward fns to be initialized priorly before setup_state_spaces()
        super().__init__(*args, **kwargs)
        self._reset_goals(torch.arange(n_envs, device=device)) # other env params will now be set by super.init

    
    def _reset_goals(self, env_ids):
        """Reset goal positions for given environments"""
        n = len(env_ids)
        
        # Sample random goal positions in a circle
        angles = torch.rand(n, device=self.device) * 2 * np.pi
        distances = torch.rand(n, device=self.device) * 5.0 + 2.0  # 2-7 meters
        
        self.goal_pos[env_ids, 0] = torch.cos(angles) * distances
        self.goal_pos[env_ids, 1] = torch.sin(angles) * distances
        
        # Sample goal speed
        self.goal_speed[env_ids] = torch.clamp(
            torch.randn(n, device=self.device) * self.goal_sp_std + self.goal_sp_mean,
            self.goal_sp_min, self.goal_sp_max
        )
        
        # Reset goal timer
        if self.goal_timer is not None:
            self.goal_timer[env_ids] = torch.randint(
                self.goal_timer_range[0], self.goal_timer_range[1], 
                (n,), dtype=self.goal_timer.dtype, device=self.device
            )
    
    def reset_envs(self, env_ids):
        """Reset environments and their goals"""
        super().reset_envs(env_ids)
        self._reset_goals(env_ids)
    
    def step(self, actions):
        """Step and update goal timer"""
        obs, rews, terminated, truncated, info = super().step(actions)
        
        # Update goal timer
        if self.goal_timer is not None:
            self.goal_timer -= 1
            reset_envs = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(reset_envs) > 0:
                self._reset_goals(reset_envs)
        
        return obs, rews, terminated, truncated, info
    
    def observe(self, env_ids=None):
        """Observe with goal information"""
        base_obs = super().observe(env_ids) # calls ICCGANHumanoid.observe()
        
        # Compute goal-relative observation
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        
        # Goal position relative to root
        root_pos = self.root_tensor[env_ids, :2]  # x, y position
        goal_rel = self.goal_pos[env_ids] - root_pos          # raw (dx, dy)
        dist = goal_rel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        goal_dir = goal_rel / dist                             # unit direction (2D)
        speed = self.goal_speed[env_ids].unsqueeze(-1)         # (n, 1)
        
        # concat into goal vector: [dir_x, dir_y, dist, speed] = 4D
        goal_vec = torch.cat([goal_dir, dist, speed], dim=-1)
        return torch.cat([base_obs, goal_vec], dim=-1)
    
    def reward(self):
        """Compute goal-reaching reward"""
        # Distance to goal
        root_pos = self.root_tensor[:, :2]
        dist = torch.norm(self.goal_pos - root_pos, dim=-1, keepdim=True)
        
        # Reward for being close to goal
        reward = (dist < self.goal_radius).float()
        
        # Small penalty for distance
        reward -= 0.01 * dist
        
        return reward


from utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized

def observe_iccgan_safe(state_hist: torch.Tensor, seq_len: Optional[torch.Tensor]=None,
    key_links: Optional[List[int]]=None, parent_link: Optional[int]=None,
    include_velocity: bool=True, local_pos: Optional[bool]=None, ground_height:Optional[torch.Tensor]=None
):
    """Safe ICCGAN observation function (same as original) with NaN handling"""
    UP_AXIS = 2
    n_hist = state_hist.size(0)
    n_inst = state_hist.size(1)

    link_tensor = state_hist.view(n_hist, n_inst, -1, 13)
    
    # Handle NaN/invalid quaternions by replacing with identity
    # A valid quaternion should have unit norm
    quats = link_tensor[..., 3:7]
    quat_norms = torch.norm(quats, dim=-1, keepdim=True)
    
    # Replace zero/invalid quaternions with identity [0,0,0,1]
    invalid_mask = quat_norms < 1e-6
    identity_quat = torch.tensor([0., 0., 0., 1.], dtype=quats.dtype, device=quats.device)
    quats = torch.where(invalid_mask, identity_quat.view(1, 1, 1, 4), quats / (quat_norms + 1e-8))
    link_tensor = link_tensor.clone()
    link_tensor[..., 3:7] = quats
    
    if key_links is None:
        link_pos, link_orient = link_tensor[...,:3], link_tensor[...,3:7]
    else:
        link_pos, link_orient = link_tensor[:,:,key_links,:3], link_tensor[:,:,key_links,3:7]

    if parent_link is None:
        root_tensor = state_hist[..., :13]
        if local_pos is True:
            origin = root_tensor[:,:, :3]
            orient = root_tensor[:,:,3:7]
        else:
            origin = root_tensor[-1,:, :3]
            orient = root_tensor[-1,:,3:7]

        heading = heading_zup(orient)
        up_dir = torch.zeros_like(origin)
        up_dir[..., UP_AXIS] = 1
        orient_inv = axang2quat(up_dir, -heading)
        orient_inv = orient_inv.view(-1, n_inst, 1, 4)

        origin = origin.clone()
        if ground_height is None:
            origin[..., UP_AXIS] = 0
        else:
            origin[..., UP_AXIS] = ground_height
        origin.unsqueeze_(-2)
    else:
        if local_pos is True or local_pos is None:
            origin = link_tensor[:,:, parent_link, :3]
            orient = link_tensor[:,:, parent_link,3:7]
        else:
            origin = link_tensor[-1,:, parent_link, :3]
            orient = link_tensor[-1,:, parent_link,3:7]
        orient_inv = quatconj(orient)
        orient_inv = orient_inv.view(-1, n_inst, 1, 4)
        origin = origin.unsqueeze(-2)

    ob_link_pos = link_pos - origin
    ob_link_pos = rotatepoint(orient_inv, ob_link_pos)
    ob_link_orient = quatmultiply(orient_inv, link_orient)

    if include_velocity:
        if key_links is None:
            link_lin_vel, link_ang_vel = link_tensor[...,7:10], link_tensor[...,10:13]
        else:
            link_lin_vel, link_ang_vel = link_tensor[:,:,key_links,7:10], link_tensor[:,:,key_links,10:13]
        ob_link_lin_vel = rotatepoint(orient_inv, link_lin_vel)
        ob_link_ang_vel = rotatepoint(orient_inv, link_ang_vel)
        ob = torch.cat((ob_link_pos, ob_link_orient,
            ob_link_lin_vel, ob_link_ang_vel), -1)
    else:
        ob = torch.cat((ob_link_pos, ob_link_orient), -1)
    
    ob = ob.view(n_hist, n_inst, -1)

    ob1 = ob.permute(1, 0, 2)
    if seq_len is None: 
        return ob1

    ob2 = torch.zeros_like(ob1)
    arange = torch.arange(n_hist, dtype=seq_len.dtype, device=seq_len.device).unsqueeze_(0)
    seq_len_ = seq_len.unsqueeze(1)
    mask1 = arange > (n_hist-1) - seq_len_
    mask2 = arange < seq_len_
    ob2[mask2] = ob1[mask1]
    return ob2