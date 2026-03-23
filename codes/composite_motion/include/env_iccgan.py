"""
MuJoCo-compatible environment for CompositeMotion
Converted from IsaacGym version
"""
from typing import Dict, Tuple, Optional, List
import torch, os
import numpy as np

from .ref_motion import ReferenceMotion
from .env_mujoco import MujocoEnv, DiscriminatorConfig

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
        self.loop_phase_obs = kwargs.pop("loop_phase_obs", False)
        self.phase_period  = float(kwargs.pop("phase_period", 1.0)) # best not to be set by training params
        # Cycle motion if termination not triggered & to set episode length based on how many motion cycles before hard reset.
        self.max_cycles  = int(kwargs.pop("max_cycles", 1)) # NOTE: keep > 1 if the motion is loopable, if not keep def: 1
        
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
        self.goal_dim = self.GOAL_DIM + (2 if self.loop_phase_obs else 0) # phase encoding adds 2 dims to the goal slice (sin/cos of phase)
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
        if "phase_period" not in kwargs and hasattr(self.ref_motion, 'motion_length'): # overwrite if not passed as training args
            #print(f"Ref-Motion :: Phase Input overwrites phase_period: {self.phase_period} -> {self.ref_motion.period} (max clip len in secs)\n")
            self.phase_period = sum(self.ref_motion.motion_length) / len(self.ref_motion.motion_length) # avg motion len from ref-motion clips
        
        # longest clip length converted to control steps.
        self.steps_per_cycle = max(1 * self.fps, round(self.phase_period * self.fps)) # (keep min range for 1s i.e. 30 steps for 30 fps motion)
        self.episode_length = self.max_cycles * self.steps_per_cycle
        print("\n[ICCGAN-Humanoid Init] Update Episode params:")
        print(f"  Motion cycle: {self.phase_period:.3f}s (avg) = {self.steps_per_cycle} steps (aggregate).")
        print(f"  Max Episode Length = {self.steps_per_cycle} x {self.max_cycles} = {self.episode_length} steps.\n")
        
        self.sampling_workers = []
        self.real_samples = []

    # ---------------------------------------------------------------------------------------------
       
    def build_motion_lib(self, motion_file):
        """Build reference motion library"""
        self.ref_motion = ReferenceMotion(
            motion_file=motion_file, 
            character_model=self.character_model, 
            device=self.device
        )
        # @overwrite prev calc in base class for refernce motion data
        if self.ref_motion.fps: 
            avg_fps = sum(self.ref_motion.fps) / len(self.ref_motion.fps)
            print(f"Build Ref-Motion overwrites fps: {self.fps} -> {avg_fps}") 
            self.fps = avg_fps # sets avg fps from ref-motion
            self.frameskip = int(self.run_speed/self.fps)
            self.step_time = 1.0 / self.fps
    
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
        self.terminate_counter = torch.zeros((self.n_envs,), dtype=torch.float32, device=self.device)

        # Phase-conditioned observation tensors
        if self.loop_phase_obs:
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
    
    # ---------------------------------------------------------------------------------------------

    def reset_envs(self, env_ids):  # overwrite for extra handling
        super().reset_envs(env_ids)          # calls init_state → sets env_motion_times
        self.state_hist[:, env_ids] = 0.0
        self.terminate_counter[env_ids] = 0.0
        # Phase Input: seed phase from the motion time sampled in init_state
        if self.loop_phase_obs:
            self.env_phase[env_ids] = (
                self.env_motion_times[env_ids] % self.phase_period
            ) / self.phase_period

    def reset_done(self):
        """Reset done environments and return observations with info"""
        obs, info = super().reset_done()
        info["ob_seq_lens"] = self.ob_seq_lens
        info["reward_weights"] = self.reward_weights
        return obs, info
    
    def _cycle_reset(self, env_ids):
        """Soft cycle reset: preserve all physics state, only clear GRU history.

        Called when an env completes one full motion cycle without falling.
        Joint positions, velocities, root pose and root velocity are intentionally
        kept as-is so physics continues seamlessly into the next cycle.
        We only clear state_hist so the GRU doesn't see stale cross-cycle frames,
        and reset lightweight counters that track within-cycle state.

        No need teleport Root world-XY back to origin: the discriminator's
        observe_iccgan() already works in a root-relative frame (horizontal
        origin zeroed, only heading used), so absolute XY drift is invisible to
        imitation loss and harmless for physics stability.
        """
        # Clear GRU history — ob_seq_lens (= lifetime % motion_ep_len + 1 = 1 at cycle
        # start) already tells the GRU to read only the freshest slot, so this is safe.
        self.state_hist[:, env_ids] = 0.0
        # Reset per-cycle counters
        self.terminate_counter[env_ids] = 0.0
        if self.loop_phase_obs:
            self.env_phase[env_ids] = 0.0

    def init_state(self, env_ids):
        """Initialize state from reference motion"""
        motion_ids, motion_times = self.ref_motion.sample(len(env_ids))
        ref_link_tensor, ref_joint_tensor = self.ref_motion.state(motion_ids, motion_times)
        
        # Adjust for ground height if needed
        ground_height = self.ground_height(ref_link_tensor[:, 0, :3], env_ids)
        if ground_height is not None:
            ref_link_tensor[:, :, 2] += ground_height.unsqueeze(1)
        
        # Phase Input: store sampled motion times so reset_envs can seed phase from them
        if self.loop_phase_obs:
            self.env_motion_times[env_ids] = torch.from_numpy(
                np.array(motion_times, dtype=np.float32)
            ).to(self.device)
        
        return ref_link_tensor, ref_joint_tensor
    
    # ---------------------------------------------------------------------------------------------
        
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
                    ob = observe_iccgan(samples[-disc.ob_horizon:], None, disc.key_links, disc.parent_link,
                        include_velocity=False, local_pos=disc.local_pos)
                    disc_obs[name] = ob.cpu()
                
                obs_list.append(disc_obs)
            
            self.real_samples = obs_list
        
        return self.real_samples.pop()
    
    def step(self, actions):
        """Step environment and update discriminator observations"""
        obs, rews, terminated, truncated, info = super().step(actions)
        if self.discriminators and self.training:
            info["disc_obs"] = self.observe_disc(self.state_hist)
            info["disc_obs_expert"] = self.fetch_real_samples()

        # --- Motion cycle detection: reset only when given motion is for sure is loopable (start frame & end frame of motion should be seamless) ---
        # When an env finishes one full motion clip (lifetime is a nonzero multiple of motion_ep_len) 
        # without having already been terminated/truncated, perform a soft cycle reset so the motion loops seamlessly.
        # self.obs and self.done are already set by super().step(); we must NOT touch
        # self.done here — only clear history for envs that are still alive.
        at_cycle_end = (self.lifetime % self.steps_per_cycle == 0) & (self.lifetime > 0)
        cycle_env_ids = torch.nonzero(at_cycle_end & ~self.done).view(-1) # done or not --> set by super by checking termination conditions
        if len(cycle_env_ids) > 0:
            self._cycle_reset(cycle_env_ids)
        if self.verbose:
            print(f"\n[STEP-CYCLE] {len(cycle_env_ids)} envs completed a motion cycle at lifetime={self.lifetime[cycle_env_ids].cpu().numpy()}\n")

        return obs, rews, terminated, truncated, info
    
    def ground_height(self, p, env_ids=None):
        """Get ground height at position"""
        return None
    
    # ---------------------------------------------------------------------------------------------
    
    def observe(self, env_ids=None):
        """Observe with ICCGAN observation function"""
        if hasattr(self,'steps_per_cycle'): # handle case when super init calls before param is assigned
            lifetime_in_cycle = self.lifetime % self.steps_per_cycle
        else:
            lifetime_in_cycle = self.lifetime
        self.ob_seq_lens = torch.clamp(lifetime_in_cycle + 1, max=self.ob_horizon)
        n_envs = self.n_envs
        
        if env_ids is None or len(env_ids) == n_envs:
            self.state_hist[:-1] = self.state_hist[1:].clone()
            self.state_hist[-1] = self.char_link_tensor.view(n_envs, -1)
            env_ids = None
            # Phase Input: advance phase for ALL envs each simulation step (full-env call only)
            # Subset calls (reset envs) already have correct phase set in reset_envs()
            if self.loop_phase_obs:
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
            obs = observe_iccgan(
                self.state_hist[-self.ob_horizon:], self.ob_seq_lens, self.key_links, self.parent_link,
                ground_height=ground_height
            ).flatten(start_dim=1)
            # Phase Input: append (sin, cos) phase encoding — bounded [-1,1], no normalisation needed
            if self.loop_phase_obs:
                phase_enc = torch.stack([
                    torch.sin(2.0 * np.pi * self.env_phase),
                    torch.cos(2.0 * np.pi * self.env_phase),
                ], dim=-1)
                obs = torch.cat([obs, phase_enc], dim=-1)
        else:
            ground_height = self.ground_height(self.state_hist[-1, env_ids, :3], env_ids)
            obs = observe_iccgan(
                self.state_hist[-self.ob_horizon:][:, env_ids], self.ob_seq_lens[env_ids],
                self.key_links, self.parent_link,
                ground_height=ground_height
            ).flatten(start_dim=1)
            # Phase Input: append phase encoding for subset
            if self.loop_phase_obs:
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
                ob = observe_iccgan(
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
                res[disc_name] = observe_iccgan(
                    s[-disc.ob_horizon:], seq_len,
                    disc.key_links, disc.parent_link,
                    include_velocity=False, local_pos=disc.local_pos
                )
                seq_len_[disc_name] = seq_len
            return res, seq_len_

    def termination_check(self):
        """Check termination conditions.
        Two independent paths both contribute to termination:

        PATH A  — contact + height (original logic, catches torso-on-ground):
            Any non-foot link is both (a) receiving a contact force > 1 and
            (b) below its height threshold.  Requires N consecutive frames
            (grace_steps=1 def training) to fire, which smooths out brief ground-brush events.

        PATH B  — root height only (NEW, catches the lying/sliding loophole):
            If the root (pelvis) z-position drops below `term_height` for (N*2)
            consecutive frames we terminate regardless of contact forces.
            --> 2 times original so that training, dont waste too much time on it
            --> def training grace_step is supposed to be strict, i.e. 1, just like src code
            grace steps was initially added as a feature while testing, later extend to motion based
            where roll action require a bit of grace time to actually see if the character rise up again.
        
        New Termination Counter Logic:
            Prevents accumulation - Counter won't grow indefinitely over long episodes
            Allows recovery - Character can fall briefly and get up without terminating
            Recent-weighted - Only the last few frames matter (like a sliding window)
            No separate counters - Still using single shared counter for both paths
        """
        if self.contactable_links is None:
            return torch.zeros_like(self.done)

        # --- PATH A: contact-force + per-link height ---
        contacted = torch.any(self.char_contact_force_tensor.abs() > 1., dim=-1)  # [n_envs, n_links]

        ground_height = self.ground_height(self.char_root_tensor[:, :3])
        if ground_height is None:
            low_threshold = self.contactable_links
        else:
            low_threshold = self.contactable_links + ground_height.unsqueeze(1)

        too_low = self.link_pos[..., self.UP_AXIS] < low_threshold           # [n_envs, n_links]
        # any of the joints have force > 1 & current height from ground < threshold
        cond_env_A = torch.any(torch.logical_and(contacted, too_low), dim=-1) # [n_envs]

        # --- PATH B: root/pelvis height alone ---
        root_z = self.root_pos[:, self.UP_AXIS]                               # [n_envs]
        root_floor = 0.0 if ground_height is None else ground_height
        cond_env_B = (root_z - root_floor) < self.term_height                 # [n_envs]

        # --- Weighted grace counter (shared across both paths) ---
        # Path A: +1.0 per frame, Path B (when A fails): +0.5 per frame
        # Neither: -1.0 per frame (allows recovery, prevents accumulation)
        increment = torch.where(
            cond_env_A,                    # If Path A satisfied
            torch.ones_like(self.terminate_counter, dtype=torch.float32),
            torch.where(
                cond_env_B,                # Else if Path B satisfied
                torch.full_like(self.terminate_counter, 0.5),
                torch.full_like(self.terminate_counter, -1.0)  # Recovery: decrement
            )
        )
        
        # Accumulate increment with clamping to prevent negative values
        self.terminate_counter = torch.clamp(self.terminate_counter + increment, min=0.0)

        terminate = (self.terminate_counter >= int(self.grace_steps)) & (self.lifetime > 1)

        if self.verbose and self.simulation_step % 100 == 0:
            print(f"  [TERM-DBG] step={self.simulation_step:5d} | "
                  f"pathA={cond_env_A.sum().item()}/{self.n_envs}  "
                  f"pathB={cond_env_B.sum().item()}/{self.n_envs}  "
                  f"terminate={terminate.sum().item()} | "
                  f"root_h_min={root_z.min().item():.3f}  "
                  f"cf_max={self.char_contact_force_tensor.abs().max().item():.2f}")

        return terminate
    
    def reward(self):
        """Optional feature for ICCGAN - use default for pretrained weights of composite_motion"""
        # defualt 0 task rewards (rew_dim = 0) i.e. trained only on 1 discriminator
        return super().reward()

# =================================================================================================

class ICCGANHumanoidTarget(ICCGANHumanoid):
    """ICCGAN Humanoid with target reaching — ported from IsaacGym src.

    goal_tensor layout : [x, y, z]  (3-D world position; z kept for compat, always 0)
    Goal observation   : [dir_x, dir_y, speed, dist] — 4D, root-heading–relative
    Reward             : velocity-matching reward (same formula as src repo)
    Extra termination  : too_far — if agent drifts > 3 m beyond initial dist to goal
    Visualization      : goal marker drawn every render() call (sphere + radius ring)
    """

    GOAL_DIM = 4                    # dir(2) + speed(1) + dist(1)  — paper App B.2
    GOAL_REWARD_WEIGHT = [0.5]
    ENABLE_GOAL_TIMER = True
    GOAL_TENSOR_DIM = 3             # (x, y, z) world position; z unused (kept for src compat)

    GOAL_RADIUS    = 0.5
    SP_LOWER_BOUND = 1.2
    SP_UPPER_BOUND = 1.5
    GOAL_TIMER_RANGE = (90, 150)
    GOAL_SP_MEAN   = 1.0
    GOAL_SP_STD    = 0.25
    GOAL_SP_MIN    = 0.0
    GOAL_SP_MAX    = 1.25
    SHARP_TURN_RATE = 1             # probability of large direction change (1 = always random)

    def __init__(self, *args,
        goal_radius:      float           = GOAL_RADIUS,
        sp_lower_bound:   float           = SP_LOWER_BOUND,
        sp_upper_bound:   float           = SP_UPPER_BOUND,
        goal_timer_range: Tuple[int, int] = GOAL_TIMER_RANGE,
        goal_sp_mean:     float           = GOAL_SP_MEAN,
        goal_sp_std:      float           = GOAL_SP_STD,
        goal_sp_min:      float           = GOAL_SP_MIN,
        goal_sp_max:      float           = GOAL_SP_MAX,
        sharp_turn_rate:  float           = SHARP_TURN_RATE,
        **kwargs
    ):
        self.goal_radius     = goal_radius
        self.sp_lower_bound  = sp_lower_bound
        self.sp_upper_bound  = sp_upper_bound
        self.goal_timer_range = goal_timer_range
        self.goal_sp_mean    = goal_sp_mean
        self.goal_sp_std     = goal_sp_std
        self.goal_sp_min     = goal_sp_min
        self.goal_sp_max     = goal_sp_max
        self.sharp_turn_rate = sharp_turn_rate

        # super().__init__ will call create_tensors() → goal_tensor allocated there
        # (GOAL_TENSOR_DIM = 3, ENABLE_GOAL_TIMER = True handled by ICCGANHumanoid)
        super().__init__(*args, **kwargs)

        # init_dist tracks the distance to the goal at the moment it was spawned.
        # Used by termination_check() to detect too_far condition.
        self.init_dist = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)

        # First goal assignment (after super init so root_pos is valid)
        all_ids = torch.arange(self.n_envs, device=self.device)
        self.reset_goal(all_ids)

    # ====== Goal management ---------------------------------------------
    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        """Sample a new navigation goal for the given envs.

        Mirrors src ICCGANHumanoidTarget.reset_goal() exactly:
          • large_angle  — uniformly random heading (sharp turn)
          • small_angle  — current heading ± 60° (gentle turn)
          • sharp_turn_rate controls the mix
          • goal distance = sampled_speed × timer × step_time
          • goal stored as absolute world (x, y) in goal_tensor[:,0:2]
        """
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer  is None: goal_timer  = self.goal_timer

        n         = len(env_ids)
        all_envs  = (n == self.n_envs)
        root_orient = self.root_orient if all_envs else self.root_orient[env_ids]

        # --- Direction sampling (replicates src exactly) ---
        small_turn  = torch.rand(n, device=self.device) > self.sharp_turn_rate
        large_angle = torch.rand(n, dtype=torch.float32, device=self.device).mul_(2 * np.pi)
        small_angle = torch.rand(n, dtype=torch.float32, device=self.device).sub_(0.5).mul_(2 * (np.pi / 3))
        heading     = heading_zup(root_orient)
        small_angle = small_angle + heading
        theta       = torch.where(small_turn, small_angle, large_angle)

        # --- Speed & timer sampling ---
        timer = torch.randint(
            self.goal_timer_range[0], self.goal_timer_range[1],
            (n,), dtype=goal_timer.dtype, device=self.device
        )
        if self.goal_sp_min == self.goal_sp_max:
            vel = torch.full((n,), self.goal_sp_min, dtype=torch.float32, device=self.device)
        elif self.goal_sp_std == 0:
            vel = torch.full((n,), self.goal_sp_mean, dtype=torch.float32, device=self.device)
        else:
            vel = torch.nn.init.trunc_normal_(
                torch.empty(n, dtype=torch.float32, device=self.device),
                mean=self.goal_sp_mean, std=self.goal_sp_std,
                a=self.goal_sp_min, b=self.goal_sp_max
            )

        dist = vel * timer.float() * self.step_time
        dx   = dist * torch.cos(theta)
        dy   = dist * torch.sin(theta)
        root_pos = self.root_pos if all_envs else self.root_pos[env_ids]

        if all_envs:
            self.init_dist           = dist
            goal_timer.copy_(timer)
            goal_tensor[:, 0] = root_pos[:, 0] + dx
            goal_tensor[:, 1] = root_pos[:, 1] + dy
            goal_tensor[:, 2] = 0.0
        else:
            self.init_dist[env_ids]   = dist
            goal_timer[env_ids]        = timer
            goal_tensor[env_ids, 0]   = root_pos[:, 0] + dx
            goal_tensor[env_ids, 1]   = root_pos[:, 1] + dy
            goal_tensor[env_ids, 2]   = 0.0

    # ====== Reset hooks -------------------------------------------------
    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_goal(env_ids)

    # Step: goal timer countdown → re-sample on expiry -------------------
    def step(self, actions):
        obs, rews, terminated, truncated, info = super().step(actions)
        if self.goal_timer is not None:
            self.goal_timer -= 1
            expired = torch.nonzero(self.goal_timer <= 0).view(-1)
            if len(expired) > 0:
                self.reset_goal(expired)
        return obs, rews, terminated, truncated, info

    # ====== Observation: root-heading–relative goal direction + speed + dist
    # Note: state_hist is still updated inside ICCGANHumanoid.observe().
    # We must call that update before calling observe_iccgan_target.
    # Override the full observe flow to keep the hist update:
    def _observe_iccgan(self, env_ids=None):
        """Override: append goal obs after state-hist ICCGAN obs."""
        # base body obs (handles phase too)
        base_obs = super()._observe_iccgan(env_ids)

        # goal obs in heading-relative frame
        if env_ids is None:
            sh  = self.state_hist[-self.ob_horizon:]
            gt  = self.goal_tensor
            tmr = self.goal_timer
            seq = self.ob_seq_lens
        else:
            sh  = self.state_hist[-self.ob_horizon:][:, env_ids]
            gt  = self.goal_tensor[env_ids]
            tmr = self.goal_timer[env_ids]
            seq = self.ob_seq_lens[env_ids]

        root_pos    = sh[-1, :, :3]
        root_orient = sh[-1, :, 3:7]

        dp = gt[:, :3] - root_pos
        x, y = dp[:, 0], dp[:, 1]
        heading_inv = -heading_zup(root_orient)
        c, s = torch.cos(heading_inv), torch.sin(heading_inv)
        x, y = c * x - s * y, s * x + c * y

        dist = (x * x + y * y).sqrt()
        sp   = dist * (self.fps / tmr.float().clamp(min=1))

        too_close = dist < 1e-5
        x    = torch.where(too_close, x, x / dist.clamp(min=1e-6))
        y    = torch.where(too_close, y, y / dist.clamp(min=1e-6))
        sp.clamp_(max=self.sp_upper_bound)
        dist_obs = (dist / 3.0).clamp_(max=1.5)

        goal_obs = torch.stack([x, y, sp, dist_obs], dim=-1)   # [n, 4]
        return torch.cat([base_obs, goal_obs], dim=-1)

    # ====== Reward: velocity-matching (src formula) ---------------------
    def reward(self, goal_tensor=None, goal_timer=None):
        """Velocity-matching reward identical to the IsaacGym src."""
        if goal_tensor is None: goal_tensor = self.goal_tensor
        if goal_timer  is None: goal_timer  = self.goal_timer

        p  = self.root_pos                          # current root pos [n, 3]
        p_ = self.state_hist[-1][:, :3]            # prev root pos (last state hist entry)

        # Desired velocity vector towards goal
        dp_  = goal_tensor[:, :3] - p_
        dp_[:, self.UP_AXIS] = 0
        dist_ = torch.linalg.norm(dp_, ord=2, dim=-1)
        v_    = dp_ / (goal_timer.float().unsqueeze(-1) * self.step_time).clamp(min=1e-6)

        v_mag = torch.linalg.norm(v_, ord=2, dim=-1)
        sp_   = (dist_ / self.step_time).clamp(max=v_mag.clamp(min=self.sp_lower_bound, max=self.sp_upper_bound))
        v_    = v_ * (sp_ / v_mag.clamp(min=1e-6)).unsqueeze(-1)

        # Actual velocity
        dp = p - p_
        dp[:, self.UP_AXIS] = 0
        v  = dp / self.step_time

        # Gaussian-like velocity-matching reward
        r = (v - v_).pow(2).sum(dim=-1).mul(-3.0 / sp_.pow(2).clamp(min=1e-6)).exp()

        # Near-goal override: reward = 1 when within goal_radius
        dp_now = goal_tensor[:, :3] - p
        dp_now[:, self.UP_AXIS] = 0
        dist_now = torch.linalg.norm(dp_now, ord=2, dim=-1)
        self.near = dist_now < self.goal_radius
        r[self.near] = 1.0

        # Accelerate goal respawn when agent is near (viewer mode only)
        if self.render_mode is not None and self.goal_timer is not None:
            self.goal_timer[self.near] = self.goal_timer[self.near].clamp(max=20)

        return r.unsqueeze(-1)

    # ====== Extra termination: too_far ----------------------------------
    def termination_check(self, goal_tensor=None):
        """Fall check (from super) + too-far-from-goal check."""
        if goal_tensor is None: goal_tensor = self.goal_tensor

        fall = super().termination_check()

        dp   = goal_tensor[:, :3] - self.root_pos
        dp[:, self.UP_AXIS] = 0
        dist = dp.pow(2).sum(dim=-1).sqrt()
        too_far = (dist - self.init_dist) > 3.0

        return torch.logical_or(fall, too_far)

    # ====== Render: draw goal marker (sphere + radius ring) non-physics overlay
    def _add_goal_geoms_to_scene(self, scn):
        """Add goal-position visualization geoms to an MjvScene.
        Draws for env-0 only (single character when render_mode is active).
        Two elements:
          • a red sphere at (goal_x, goal_y, 0.05) — the target point
          • a blue ring of line-segment approximation of the goal_radius circle
        Uses mujoco.mjv_initGeom which is available in mujoco-python >= 3.x.
        Falls back silently if the API is unavailable or scene is full.
        """
        try:
            gx, gy = self.goal_tensor[0, 0].item(), self.goal_tensor[0, 1].item()
            # Red target sphere
            self._mjv_make_sphere(scn,
                [gx, gy, 0.05], radius=0.12,
                rgba=[1.0, 0.15, 0.15, 0.85])
            # Blue goal-radius ring (N-gon of capsules at z=0.02)
            N = 16
            r = float(self.goal_radius)
            angles = [2 * np.pi * i / N for i in range(N + 1)]
            pts = [[gx + r*np.cos(a), gy + r*np.sin(a), 0.02] for a in angles]
            for i in range(N):
                self._mjv_make_capsule(scn, pts[i], pts[i+1],
                    radius=0.02, rgba=[0.2, 0.4, 1.0, 0.9])
        except Exception:
            pass  # silently skip if MuJoCo API unavailable or scene full

    def render(self):
        """Render with non-physics goal overlay (env-0 only).
 
        rgb_array path
        --------------
        1. renderer.update_scene() fills the renderer's internal MjvScene.
        2. We inject extra geoms directly into renderer.scene (public attr, mujoco>=3).
        3. renderer.render() renders that (now augmented) scene — returns numpy array.
 
        human path
        ----------
        Passive viewer composites viewer.user_scn on top of the physics scene
        at every sync() call.  We reset and repopulate user_scn inside the
        viewer lock just before calling super().render() (which calls sync()).
        """
        import mujoco as _mj
 
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = _mj.Renderer(self.model, width=960, height=720)
            cam_id = _mj.mj_name2id(self.model, _mj.mjtObj.mjOBJ_CAMERA, "track")
            cam_arg = "track" if cam_id >= 0 else None
            self.renderer.update_scene(self.data, camera=cam_arg)
            # renderer.scene is the MjvScene that update_scene just populated —
            # inject our custom geoms into it before rendering
            try:
                self._add_goal_geoms_to_scene(self.renderer.scene)
            except AttributeError:
                pass  # mujoco < 3.x: scene attr may not exist; skip overlay
            return self.renderer.render()
 
        elif self.render_mode == "human" and self.viewer is not None:
            # Populate user_scn; viewer composites it at sync() time
            try:
                with self.viewer.lock():
                    self.viewer.user_scn.ngeom = 0
                    self._add_goal_geoms_to_scene(self.viewer.user_scn)
            except Exception:
                pass
            # fall through to super().render() which calls viewer.sync()
 
        return super().render()

# =================================================================================================

class ICCGANHumanoidTargetAiming(ICCGANHumanoidTarget):
    """
    Scope for further improvements: (extra goals based on shoot projectiles)
    
    No projectile in the src: the task is purely arm-alignment, NOT shooting.
    We add optional visual-only "shot beams" when alignment is very good
    (aiming_rew > SHOT_THRESHOLD). These are non-physics overlays rendered into
    the MuJoCo scene; they have no effect on rewards or physics.
 
    Projectile physics discussion
    ==============================
    Two design options (src uses neither; this is our extension):
 
    Option A — Gravity-free (laser / instantaneous):
      Draw a straight capsule from hand → target.  Simple, zero cost.
      Works because the reference motion implies the character "fires" the moment
      the arm is aligned, with negligible travel time vs. env step rate.
 
    Option B — Ballistic (arrow / ball, affected by gravity):
      Position: p(t) = p0 + v0·t + ½·g·t²  (g ≈ [0,0,-9.81])
      Render as a moving sphere updated each step until it hits the ground or
      a predetermined time-of-flight.
    
    Current implementation uses Option A (straight beam) as the default because 
        (1) step_time resolution is coarse for realistic ballistics, 
        (2) the env is imitation-learning 
        so what matters is arm alignment, not projectile impact.
    """
    GOAL_DIM        = 4 + 3        # 4-D nav obs  +  3-D aiming direction obs (desired forearm orientation *relative to the direction from root to target*)
    GOAL_TENSOR_DIM = 3 + 4        # 3-D nav target  +  4-D aiming quaternion
    GOAL_REWARD_WEIGHT = [0.25, 0.25]  # two reward components, equal weight
 
    # Links used for aiming reward/visualisation (MuJoCo body names)
    AIMING_START_LINK_NAME = "right_lower_arm"   # forearm origin
    AIMING_END_LINK_NAME   = "right_hand"        # forearm tip (gun barrel end)
 
    # Visualisation
    SHOT_THRESHOLD     = 0.85      # aiming_rew above this → flash shot beam
    SHOT_BEAM_FRAMES   = 5         # how many env steps the shot beam stays visible

    def __init__(self, *args, **kwargs):
        # These will be properly resolved in create_tensors() after super().__init__
        self.aiming_start_link = None
        self.aiming_end_link   = None
        # Active shot-beam visuals: list of {pos, dir, ttl} dicts (env-0 only)
        self._shot_beams: list = []
        super().__init__(*args, **kwargs)


    # ------------------------------------------------------------------
    # Tensor setup
    # ------------------------------------------------------------------
    def create_tensors(self):
        super().create_tensors()
        n_envs = self.n_envs
 
        # Resolve forearm / hand body indices (adjusted for world body at 0)
        def _link_idx(name):
            if name in self.body_names:
                idx = self.body_names.index(name)
                return idx - 1 if idx > 0 else None  # -1 for world-body offset
            print(f"[ICCGANHumanoidTargetAiming] WARNING: body '{name}' not found; aiming disabled.")
            return None
 
        self.aiming_start_link = _link_idx(self.AIMING_START_LINK_NAME)
        self.aiming_end_link   = _link_idx(self.AIMING_END_LINK_NAME)
 
        # x_dir: unit vector along world-x axis (used as reference for quaternion math)
        self.x_dir = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
        self.x_dir[:, 0] = 1.0
        # reverse_rotation: quaternion for 180° around z (handles target directly behind)
        self.reverse_rotation = torch.zeros((n_envs, 4), dtype=torch.float32, device=self.device)
        self.reverse_rotation[:, self.UP_AXIS] = 1.0   # (0,0,1,0) — 180° around z
 
    # ------------------------------------------------------------------
    # Goal sampling: nav goal via parent, then fresh aiming quaternion
    # ------------------------------------------------------------------
    def reset_goal(self, env_ids, goal_tensor=None, goal_timer=None):
        # Delegate navigation goal to parent (writes into [:, :3] slice view)
        super().reset_goal(env_ids,
            goal_tensor=self.goal_tensor[:, :3] if goal_tensor is None else goal_tensor,
            goal_timer=goal_timer)
        self._reset_aiming_goal(env_ids)
 
    def _reset_aiming_goal(self, env_ids):
        """Sample a random aiming quaternion for each env_id.
 
        Mirrors src reset_aiming_goal():
          elevation ∈ [−π/6, 0]  (gun tilted slightly down to up)
          azimuth   ∈ [0, π/4]   (gun pointed slightly off-axis)
        Stored as xyzw quaternion in goal_tensor[:, 3:7].
        """
        n = len(env_ids)
        # src samples half-angles directly
        elev = torch.rand(n, dtype=torch.float32, device=self.device).mul_(-np.pi / 6) / 2
        azim = torch.rand(n, dtype=torch.float32, device=self.device).mul_(np.pi  / 4) / 2
 
        cp, sp = torch.cos(elev), torch.sin(elev)   # pitch (elevation)
        cy, sy = torch.cos(azim), torch.sin(azim)   # yaw   (azimuth)
 
        # Quaternion (roll=0): w=cp·cy, x=−sp·sy, y=sp·cy, z=cp·sy
        w =  cp * cy
        x = -sp * sy
        y =  sp * cy
        z =  cp * sy
 
        all_envs = (n == self.n_envs)
        if all_envs:
            self.goal_tensor[:, 3] = x
            self.goal_tensor[:, 4] = y
            self.goal_tensor[:, 5] = z
            self.goal_tensor[:, 6] = w
        else:
            self.goal_tensor[env_ids, 3] = x
            self.goal_tensor[env_ids, 4] = y
            self.goal_tensor[env_ids, 5] = z
            self.goal_tensor[env_ids, 6] = w
 
    # ------------------------------------------------------------------
    # Observation: body obs + 4-D nav goal + 3-D aiming direction
    # ------------------------------------------------------------------
    def _observe_iccgan(self, env_ids=None):
        """Extends parent's _observe_iccgan with a 3-D aiming direction obs."""
        # 4-D nav obs appended by parent
        base_obs = super()._observe_iccgan(env_ids)   # [n, body_dim + 4]
 
        # Aiming direction in root-heading frame (3 dims)
        if env_ids is None:
            sh  = self.state_hist[-self.ob_horizon:]
            gt  = self.goal_tensor
        else:
            sh  = self.state_hist[-self.ob_horizon:][:, env_ids]
            gt  = self.goal_tensor[env_ids]
 
        aiming_ob = self._compute_aiming_obs(sh, gt)  # [n, 3]
        return torch.cat([base_obs, aiming_ob], dim=-1)
 
    def _compute_aiming_obs(self, sh, gt):
        """Compute the 3-D aiming direction observation (heading-relative).
 
        Matches observe_iccgan_target_aiming() in src env.py exactly.
        When the character is within goal_radius the aiming obs is zeroed
        (arm should be at rest, not aiming).
        """
        n = sh.size(1)
        root_pos    = sh[-1, :, :3]
        root_orient = sh[-1, :, 3:7]
 
        # Build orient_inv (heading-only rotation, around z)
        heading  = heading_zup(root_orient)
        up_dir   = torch.zeros_like(root_pos); up_dir[:, self.UP_AXIS] = 1.0
        orient_inv = axang2quat(up_dir, -heading)  # [n, 4]
 
        target_tensor  = gt[:, :3]
        aiming_tensor  = gt[:, 3:7]
 
        dp   = target_tensor - root_pos
        dp[:, self.UP_AXIS] = 0.0
        dist = torch.linalg.norm(dp, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
 
        x_dir = torch.zeros_like(dp); x_dir[:, 0] = 1.0
        target_dir = dp / dist
 
        # Quaternion that rotates x-axis to target_dir (on the ground plane)
        q = quatdiff_normalized(x_dir, target_dir)
        # Handle the degenerate "target is directly behind" case
        behind = (target_dir[:, :1] < -0.99999)
        rev    = self.reverse_rotation[:n] if n <= self.n_envs else \
                 torch.zeros_like(q); rev[:, self.UP_AXIS] = 1.0
        q = torch.where(behind, rev, q)
 
        # aiming_dir in world frame = rotate(q ⊗ aiming_quat, x_dir)
        aiming_dir = quatmultiply(q, aiming_tensor)        # compound rotation
        aiming_dir = quatmultiply(orient_inv, aiming_dir)  # into heading frame
        aiming_dir = rotatepoint(aiming_dir, x_dir)        # apply to x-axis → 3-vec [n,3]
 
        # Zero out when near goal (character is close → rest pose)
        near = (dist.squeeze(-1) < self.goal_radius)
        aiming_dir[near] = 0.0
 
        return aiming_dir   # [n, 3]
 
    # ------------------------------------------------------------------
    # Reward: nav reward (0.25) + aiming reward (0.25)
    # ------------------------------------------------------------------
    def reward(self, goal_tensor=None, goal_timer=None):
        """
        Two-component reward: navigation velocity-match + forearm alignment.
        Aiming reward (one scalar per env):
        • Compute aiming_dir = rotate(x_dir, q_target→dir * aiming_quat)
        • target_hand_pos  = fore_arm_pos + arm_len * aiming_dir
        • e = ||target_hand_pos - hand_pos|| / arm_len   (normalised alignment error)
        • aiming_rew = exp(-2·e) -1.0 when perfectly aligned
        • When near goal (within goal_radius), replaced by an "arm-raised" reward:
            rest_rew = clamp(fore_arm_dir_z / 0.8, 0, 1)
        """
        nav_rew = super().reward(
            goal_tensor=self.goal_tensor[:, :3],
            goal_timer=self.goal_timer
        )  # [n_envs, 1]
 
        aiming_rew = self._aiming_reward()   # [n_envs, 1]
 
        # Check if any env should "fire" (visual only, no physics effect)
        if self.render_mode is not None:
            self._check_and_add_shots(aiming_rew.squeeze(-1))
 
        return torch.cat([nav_rew, aiming_rew], dim=-1)   # [n_envs, 2]
 
    def _aiming_reward(self):
        """Forearm alignment reward identical to src ICCGANHumanoidTargetAiming.reward()."""
        if self.aiming_start_link is None or self.aiming_end_link is None:
            return torch.zeros((self.n_envs, 1), dtype=torch.float32, device=self.device)
 
        target_tensor = self.goal_tensor[:, :3]
        aiming_tensor = self.goal_tensor[:, 3:7]
 
        dp   = target_tensor - self.root_pos
        dp[:, self.UP_AXIS] = 0.0
        dist = torch.linalg.norm(dp, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
        target_dir = dp / dist
 
        q = quatdiff_normalized(self.x_dir, target_dir)
        behind = (target_dir[:, :1] < -0.99999)
        q = torch.where(behind, self.reverse_rotation, q)
 
        aiming_dir = rotatepoint(quatmultiply(q, aiming_tensor), self.x_dir)  # [n, 3]
 
        hand_pos      = self.link_pos[:, self.aiming_end_link]    # [n, 3]
        fore_arm_pos  = self.link_pos[:, self.aiming_start_link]  # [n, 3]
        fore_arm_vec  = hand_pos - fore_arm_pos
        arm_len       = torch.linalg.norm(fore_arm_vec, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
        fore_arm_dir  = fore_arm_vec / arm_len                    # normalised
 
        # How far does the actual hand deviate from the desired hand position?
        target_hand = fore_arm_pos + arm_len * aiming_dir
        e = torch.linalg.norm(target_hand - hand_pos, ord=2, dim=-1) / arm_len.squeeze(-1)
        aiming_rew = e.mul(-2.0).exp()   # 1.0 = perfect alignment
 
        # When near nav goal: reward for raising the arm instead (rest pose)
        rest_rew = fore_arm_dir[:, self.UP_AXIS].div(0.8).clamp(0.0, 1.0)
        aiming_rew = torch.where(self.near, rest_rew, aiming_rew)
 
        return aiming_rew.unsqueeze(-1)
 
    # ------------------------------------------------------------------
    # Termination: use only nav sub-goal (same as parent with :3 slice)
    # ------------------------------------------------------------------
    def termination_check(self, goal_tensor=None):
        return super().termination_check(goal_tensor=self.goal_tensor[:, :3])
 
    # ------------------------------------------------------------------
    # Visual-only shot beams (non-physics)
    # ------------------------------------------------------------------
    def _check_and_add_shots(self, aiming_rew_vec):
        """When env-0's aiming reward exceeds SHOT_THRESHOLD, add a shot beam."""
        if self.aiming_start_link is None:
            return
        rew0 = aiming_rew_vec[0].item()
        if rew0 > self.SHOT_THRESHOLD and not self.near[0].item():
            hand_pos = self.link_pos[0, self.aiming_end_link].cpu().numpy()
            target   = self.goal_tensor[0, :3].cpu().numpy()
 
            # Beam direction = hand → target (world space)
            diff = target - hand_pos
            dist = float(np.linalg.norm(diff))
            if dist > 1e-4:
                direction = diff / dist
                self._shot_beams.append({
                    "origin":    hand_pos.copy(),
                    "direction": direction,
                    "length":    dist,
                    "ttl":       self.SHOT_BEAM_FRAMES,
                })
        # Age existing beams
        self._shot_beams = [b for b in self._shot_beams if b["ttl"] > 0]
        for b in self._shot_beams:
            b["ttl"] -= 1
 
    def _add_goal_geoms_to_scene(self, scn):
        """Extend parent's goal visuals with aim-line and active shot beams."""
        # Navigation goal sphere + ring (from parent)
        super()._add_goal_geoms_to_scene(scn)
 
        if self.aiming_start_link is None:
            return
        if not self.near[0].item() if hasattr(self, 'near') else True:
            try:
                # Green aim-line: from forearm to estimated target point
                fore_arm_pos = self.link_pos[0, self.aiming_start_link].cpu().numpy()
                hand_pos     = self.link_pos[0, self.aiming_end_link].cpu().numpy()
                fore_arm_vec = hand_pos - fore_arm_pos
                arm_len      = float(np.linalg.norm(fore_arm_vec))
 
                if arm_len > 1e-4 and hasattr(self, 'goal_tensor'):
                    # Compute world-space aiming direction for env-0
                    target_tensor = self.goal_tensor[0:1, :3]
                    aiming_tensor = self.goal_tensor[0:1, 3:7]
                    dp  = target_tensor - self.root_pos[0:1]
                    dp[:, self.UP_AXIS] = 0.0
                    dist = dp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    x_dir      = self.x_dir[0:1]
                    target_dir = dp / dist
                    q = quatdiff_normalized(x_dir, target_dir)
                    behind = (target_dir[:, :1] < -0.99999)
                    q = torch.where(behind, self.reverse_rotation[0:1], q)
                    aiming_world = rotatepoint(quatmultiply(q, aiming_tensor), x_dir)
                    aim_end = fore_arm_pos + aiming_world[0].cpu().numpy() * (arm_len * 8)
 
                    # Draw green aim line (thin capsule)
                    self._mjv_make_capsule(scn,
                        fore_arm_pos, aim_end,
                        radius=0.015, rgba=[0.0, 1.0, 0.2, 0.7])
            except Exception:
                pass
 
        # Yellow shot beams (brief flash when fired)
        for beam in self._shot_beams:
            ttl_frac = beam["ttl"] / self.SHOT_BEAM_FRAMES
            alpha    = float(ttl_frac) * 0.9
            try:
                end_pt = beam["origin"] + beam["direction"] * beam["length"]
                self._mjv_make_capsule(scn,
                    beam["origin"], end_pt,
                    radius=0.025, rgba=[1.0, 0.85, 0.0, alpha])
            except Exception:
                pass
 
    def render(self):
        """Render with aiming overlay (inherits goal ring from parent render)."""
        # Tick shot beams every render call when NOT also calling step()
        # (during test, render is called every step so this is fine)
        return super().render()

# =================================================================================================

from .utils import heading_zup, axang2quat, rotatepoint, quatconj, quatmultiply, quatdiff_normalized


def observe_iccgan(state_hist: torch.Tensor, seq_len: Optional[torch.Tensor]=None,
    key_links: Optional[List[int]]=None, parent_link: Optional[int]=None,
    include_velocity: bool=True, local_pos: Optional[bool]=None, ground_height:Optional[torch.Tensor]=None
) -> torch.Tensor: 
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
    

def observe_iccgan_target(
    state_hist: torch.Tensor,
    seq_len: Optional[torch.Tensor],
    key_links: Optional[List[int]],
    parent_link: Optional[int],
    goal_tensor: torch.Tensor,
    timer: torch.Tensor,
    sp_upper_bound: float,
    fps: float,
    ground_height: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ICCGAN observation extended with a 4-D heading-relative goal vector.

    Matches the IsaacGym src ``observe_iccgan_target`` exactly, adapted to
    our MuJoCo tensor conventions.

    Goal observation layout (last 4 dims appended after body obs):
      • dir_x, dir_y : unit goal direction in root-heading frame
      • speed        : required speed = dist * fps / timer  (clipped to sp_upper_bound)
      • dist_norm    : dist / 3, clipped to [0, 1.5]

    Args:
        state_hist  : [n_hist, n_envs, n_links*13]
        seq_len     : [n_envs] or None
        key_links   : body indices to include (None = all)
        parent_link : root-relative origin body (None = pelvis)
        goal_tensor : [n_envs, 3]  world (x, y, z); z unused
        timer       : [n_envs]  steps remaining until goal reset
        sp_upper_bound : max speed cap
        fps         : env fps (for speed = dist*fps/timer)
        ground_height : optional [n_envs] ground offset
    Returns:
        [n_envs, body_ob_dim + 4]
    """
    # Base body observation (flattened over ob_horizon)
    ob = observe_iccgan(
        state_hist, seq_len, key_links, parent_link,
        ground_height=ground_height
    )  # [n_envs, ob_horizon, features]  OR  [n_envs, ob_dim] if seq_len handled inside

    # Root state from last history frame
    root_pos    = state_hist[-1, :, :3]   # [n_envs, 3]
    root_orient = state_hist[-1, :, 3:7]  # [n_envs, 4]  xyzw

    # Vector from root to goal in world frame
    dp = goal_tensor[:, :3] - root_pos    # [n_envs, 3]
    x  = dp[:, 0]
    y  = dp[:, 1]

    # Rotate into root-heading frame (heading = rotation around Z)
    heading_inv = -heading_zup(root_orient)  # [n_envs]
    c = torch.cos(heading_inv)
    s = torch.sin(heading_inv)
    x, y = c * x - s * y, s * x + c * y

    # Distance and required speed
    dist = (x * x + y * y).sqrt()           # [n_envs]
    sp   = dist * (fps / timer.float().clamp(min=1.0))

    # Normalise direction (avoid div-by-zero)
    too_close = dist < 1e-5
    x = torch.where(too_close, x, x / dist.clamp(min=1e-6))
    y = torch.where(too_close, y, y / dist.clamp(min=1e-6))
    sp.clamp_(max=sp_upper_bound)
    dist_obs = (dist / 3.0).clamp_(max=1.5)

    goal_ob = torch.stack([x, y, sp, dist_obs], dim=-1)  # [n_envs, 4]
    return torch.cat([ob.flatten(start_dim=1), goal_ob], dim=-1)


def observe_iccgan_target_aiming(
    state_hist:     torch.Tensor,
    seq_len:        Optional[torch.Tensor],
    key_links:      Optional[List[int]],
    parent_link:    Optional[int],
    goal_tensor:    torch.Tensor,
    timer:          torch.Tensor,
    sp_upper_bound: float,
    goal_radius:    float,
    fps:            float,
    ground_height:  Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone observation function for ICCGANHumanoidTargetAiming.
 
    Concatenates:
      • body obs      (from observe_iccgan)
      • 4-D nav obs   (from observe_iccgan_target: dir_x, dir_y, speed, dist)
      • 3-D aim obs   (aiming_dir in heading-relative frame)
 
    Total = body_dim + 4 + 3 = body_dim + 7.
 
    The aiming observation is the unit vector (x,y,z) the forearm is expected to
    point toward, expressed in the root-heading frame.  When the character is
    within goal_radius the vector is zeroed (rest pose signal).
    """
    UP_AXIS = 2
 
    target_tensor = goal_tensor[..., :3]
    aiming_tensor = goal_tensor[..., 3:]
 
    # 4-D nav observation (body obs already included inside)
    target_ob = observe_iccgan_target(
        state_hist, seq_len, key_links, parent_link,
        target_tensor, timer,
        sp_upper_bound=sp_upper_bound, fps=fps,
        ground_height=ground_height,
    )  # [n, body_dim + 4]
 
    root_pos    = state_hist[-1, :, :3]
    root_orient = state_hist[-1, :, 3:7]
 
    # heading-inverse rotation (around Z axis only)
    heading    = heading_zup(root_orient)
    up_dir     = torch.zeros_like(root_pos); up_dir[..., UP_AXIS] = 1.0
    orient_inv = axang2quat(up_dir, -heading)  # [n, 4]
 
    # direction from root to target (projected to ground plane)
    dp   = target_tensor - root_pos
    dp[..., UP_AXIS] = 0.0
    dist = torch.linalg.norm(dp, ord=2, dim=-1, keepdim=True).clamp(min=1e-6)
 
    x_dir = torch.zeros_like(dp); x_dir[..., 0] = 1.0
    target_dir = dp / dist
 
    # Quaternion: x-axis → target_dir
    q = quatdiff_normalized(x_dir, target_dir)
    # Handle 180° edge case: target exactly behind
    reverse = torch.zeros_like(q); reverse[..., UP_AXIS] = 1.0
    q = torch.where(target_dir[:, :1] < -0.99999, reverse, q)
 
    # Rotate aiming quat into heading frame and apply to x-axis
    aiming_dir = quatmultiply(q,          aiming_tensor)
    aiming_dir = quatmultiply(orient_inv, aiming_dir)
    aiming_dir = rotatepoint(aiming_dir,  x_dir)         # [n, 3]
 
    # Zero when near (rest pose)
    near = dist.squeeze(-1) < goal_radius
    aiming_dir[near, 0] = 0.0
    aiming_dir[near, 1] = 0.0
    aiming_dir[near, 2] = 0.0
 
    return torch.cat([target_ob, aiming_dir], dim=-1)   # [n, body_dim + 7]
 