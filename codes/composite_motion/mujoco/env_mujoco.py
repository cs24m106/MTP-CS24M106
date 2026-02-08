from typing import Optional, List, Union, Callable
import gymnasium as gym
import os, torch, mujoco
import numpy as np
from gymnasium import spaces


class DiscriminatorConfig(object):
    def __init__(self,
        key_links: Optional[List[str]]=None, ob_horizon: Optional[int]=None, 
        parent_link: Optional[str]=None, local_pos: Optional[bool]=None,
        replay_speed: Optional[str]=None, motion_file: Optional[str]=None,
        weight:Optional[float]=None
    ):
        self.motion_file = motion_file
        self.key_links = key_links
        self.local_pos = local_pos
        self.parent_link = parent_link
        self.replay_speed = replay_speed
        self.ob_horizon = ob_horizon
        self.weight = weight



class MujocoEnv(gym.Env):
    """Base MuJoCo environment compatible with CompositeMotion"""
    
    UP_AXIS = 2
    CHARACTER_MODEL = None
    CAMERA_POS = 0, -4.5, 2.0
    CAMERA_FOLLOWING = True
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
        n_envs: int = 1,
        fps: int = 30,
        frameskip: int = 2,
        episode_length: Optional[Union[Callable, int]] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        assert control_mode in ["position", "torque", "free"]
        self.frameskip = frameskip
        self.fps = fps
        self.step_time = 1.0 / self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length
        self.device = torch.device(f"cuda:{compute_device}" if torch.cuda.is_available() and compute_device >= 0 else "cpu")
        self.n_envs = n_envs
        self.render_mode = render_mode
        
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model
        if isinstance(self.character_model, str):
            self.character_model = [self.character_model]
        
        # Load MuJoCo model
        self._load_mujoco_model()
        # Create data
        self.data = mujoco.MjData(self.model)
        
        # Setup rendering
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Get body and joint info
        self._setup_body_joint_info()
        # Setup action normalizer
        self.setup_action_normalizer()

        import sys, traceback
        try:
            self.setup_state_spaces()
        except Exception as e:
            print(f"ERROR :: Setting up StateSpaces :: {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info() # Get exception info
            traceback.print_exception(exc_type, exc_value, exc_traceback) # Format and print the traceback


    def setup_state_spaces(self):
        """ 
        method to delay init of derived class based members to be initialized before it
        create_tensors(), observer(), reward() run on derived methods, 
        i.e. that require thier resp members to be initiallized properly
        """
        # Initialize tensors
        self.create_tensors()

        # Initialize state
        self.train()
        self.simulation_step = 0
        self.lifetime = torch.zeros(self.n_envs, dtype=torch.int64, device=self.device)
        self.done = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
        self.info = dict(lifetime=self.lifetime)
        
        # Setup spaces based on last dims
        self.act_dim = self.action_scale.size(-1)
        obs_sample = self._observe_single()
        self.ob_dim = obs_sample.shape[-1]
        rew_sample = self.reward()
        self.rew_dim = rew_sample.size(-1)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.act_dim,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.ob_dim,),
            dtype=np.float32
        )
        
        self.viewer_pause = False
        self.viewer_advance = False
        self.request_quit = False

        
    def _load_mujoco_model(self):
        """Load MuJoCo model from XML"""
        # Load the first character model
        xml_path = self.character_model[0]
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(os.path.dirname(__file__), xml_path)
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = self.step_time / self.frameskip
        
    def _setup_body_joint_info(self):
        """Setup body and joint information from MuJoCo model"""
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) 
                          for i in range(self.model.nbody)]
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) 
                           for i in range(self.model.njnt)]
        self.dof_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_DOF, i) 
                         for i in range(self.model.nv)]
        
        # Find actuated DOFs
        self.actuated_dof_ids = []
        self.actuated_joint_ids = []
        
        for i in range(self.model.nu):  # nu = number of actuators
            # Get the joint that this actuator controls
           if self.model.actuator_trnid[i, 0] >= 0:
                joint_id = self.model.actuator_trnid[i, 0]
                # Find DOF index for this joint
                dof_adr = self.model.jnt_dofadr[joint_id]
                self.actuated_dof_ids.append(dof_adr)
                self.actuated_joint_ids.append(joint_id)
        
        # Root body (pelvis) is typically body 1 (0 is world)
        self.root_body_id = 1
        
        print(f"\nBodies: {self.body_names}")
        print(f"\nJoints: {self.joint_names}")
        print(f"\nActuated DOFs: {len(self.actuated_dof_ids)}")
    
    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True

    def create_tensors(self):
        """Create tensors for state tracking"""
        n_bodies = self.model.nbody
        n_dofs = self.model.nv
        
        # Root state (position, orientation, linear vel, angular vel)
        self.root_tensor = torch.zeros((self.n_envs, 13), dtype=torch.float32, device=self.device)
        
        # Link states (all bodies)
        self.link_tensor = torch.zeros((self.n_envs, n_bodies, 13), dtype=torch.float32, device=self.device)
        
        # Joint states (position, velocity)
        self.joint_tensor = torch.zeros((self.n_envs, n_dofs, 2), dtype=torch.float32, device=self.device)
        
        # Contact forces
        self.contact_force_tensor = torch.zeros((self.n_envs, n_bodies, 3), dtype=torch.float32, device=self.device)
        
        # Action tensor
        if len(self.actuated_dof_ids) == n_dofs:
            self.action_tensor = None
        else:
            self.action_tensor = torch.zeros((self.n_envs, n_dofs), dtype=torch.float32, device=self.device)
        
        # Root link index
        self.root_links = [0]  # Simplified for single character
        
    def setup_action_normalizer(self):
        """Setup action normalization"""
        # Get joint limits from MuJoCo model
        action_lower = []
        action_upper = []
        action_scale = []
        
        for dof_id in self.actuated_dof_ids:
            jnt_id = -1
            for i in range(self.model.njnt):
                if self.model.jnt_dofadr[i] == dof_id:
                    jnt_id = i
                    break
            
            if jnt_id >= 0:
                jnt_range = self.model.jnt_range[jnt_id]
                # For position control
                if self.control_mode == "position":
                    action_lower.append(jnt_range[0])
                    action_upper.append(jnt_range[1])
                    action_scale.append(2.0)
                else:  # torque control
                    # Get actuator force limits
                    actuator_id = -1
                    for i in range(self.model.nu):
                        if self.model.actuator_trnid[i, 0] == jnt_id:
                            actuator_id = i
                            break
                    if actuator_id >= 0:
                        force_range = self.model.actuator_forcerange[actuator_id]
                        action_lower.append(force_range[0])
                        action_upper.append(force_range[1])
                        action_scale.append(1.0)
                    else:
                        action_lower.append(-100.0)
                        action_upper.append(100.0)
                        action_scale.append(1.0)
        
        action_offset = 0.5 * np.add(action_upper, action_lower)
        action_scale_arr = 0.5 * np.multiply(action_scale, np.subtract(action_upper, action_lower))
        
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale_arr, dtype=torch.float32, device=self.device)
        self.actuated_dofs = torch.tensor(self.actuated_dof_ids, dtype=torch.int64, device=self.device)
        
    def process_actions(self, actions):
        """Process actions from network"""
        a = actions * self.action_scale + self.action_offset
        if self.action_tensor is None:
            return a
        self.action_tensor[:, self.actuated_dofs] = a
        return self.action_tensor
    
    def _sync_state_from_mujoco(self, env_idx: int = 0):
        """Sync state from MuJoCo to tensors"""
        data = self.data
        
        # Root state (body 1 is typically pelvis)
        root_pos = data.xpos[self.root_body_id].copy()
        root_quat = data.xquat[self.root_body_id].copy()  # [w, x, y, z] in MuJoCo
        # Convert to [x, y, z, w] format
        root_quat_xyzw = np.array([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        root_lin_vel = data.cvel[self.root_body_id, :3].copy()
        root_ang_vel = data.cvel[self.root_body_id, 3:].copy()
        
        self.root_tensor[env_idx, :3] = torch.from_numpy(root_pos)
        self.root_tensor[env_idx, 3:7] = torch.from_numpy(root_quat_xyzw)
        self.root_tensor[env_idx, 7:10] = torch.from_numpy(root_lin_vel)
        self.root_tensor[env_idx, 10:13] = torch.from_numpy(root_ang_vel)
        
        # All link states
        for i in range(self.model.nbody):
            pos = data.xpos[i].copy()
            quat = data.xquat[i].copy()
            quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
            lin_vel = data.cvel[i, :3].copy()
            ang_vel = data.cvel[i, 3:].copy()
            
            self.link_tensor[env_idx, i, :3] = torch.from_numpy(pos)
            self.link_tensor[env_idx, i, 3:7] = torch.from_numpy(quat_xyzw)
            self.link_tensor[env_idx, i, 7:10] = torch.from_numpy(lin_vel)
            self.link_tensor[env_idx, i, 10:13] = torch.from_numpy(ang_vel)
        
        # Joint states
        for i, dof_id in enumerate(range(self.model.nv)):
            self.joint_tensor[env_idx, i, 0] = data.qpos[dof_id]
            self.joint_tensor[env_idx, i, 1] = data.qvel[dof_id]
    
    def _apply_state_to_mujoco(self, env_idx: int = 0):
        """Apply state from tensors to MuJoCo"""
        # Set root position and orientation
        root_pos = self.root_tensor[env_idx, :3].cpu().numpy()
        root_quat_xyzw = self.root_tensor[env_idx, 3:7].cpu().numpy()
        # Convert to MuJoCo format [w, x, y, z]
        root_quat = np.array([root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]])
        
        # Find free joint (root joint)
        root_jnt_id = -1
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                root_jnt_id = i
                break
        
        if root_jnt_id >= 0:
            qpos_adr = self.model.jnt_qposadr[root_jnt_id]
            self.data.qpos[qpos_adr:qpos_adr+3] = root_pos
            self.data.qpos[qpos_adr+3:qpos_adr+7] = root_quat
        
        # Set joint positions
        for i, dof_id in enumerate(range(self.model.nv)):
            if i < self.model.nv:
                self.data.qpos[dof_id] = self.joint_tensor[env_idx, i, 0].item()
                self.data.qvel[dof_id] = self.joint_tensor[env_idx, i, 1].item()
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False
        
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialize state
        env_ids = torch.arange(self.n_envs, device=self.device)
        self.reset_envs(env_ids)
        self.obs = None
    
    def reset_done(self):
        """Reset environments that are done and return observations"""
        if not self.viewer_pause:
            env_ids = torch.nonzero(self.done).view(-1)
            if len(env_ids):
                self.reset_envs(env_ids)
                if len(env_ids) == self.n_envs or self.obs is None:
                    self.obs = self.observe()
                else:
                    self.obs[env_ids] = self.observe(env_ids)
        return self.obs, self.info
        
    def reset_envs(self, env_ids):
        """Reset specific environments"""
        ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)
        
        # For single environment, just use the first entry
        if self.n_envs == 1:
            # ref_link_tensor doesn't include world body, but link_tensor does
            # Skip world body (index 0) when assigning
            n_ref_links = ref_link_tensor.shape[1]
            self.root_tensor[0] = ref_link_tensor[0, 0]  # First body is root
            self.link_tensor[0, 1:n_ref_links+1] = ref_link_tensor[0]  # Skip world body
            
            if self.action_tensor is None:
                self.joint_tensor[0] = ref_joint_tensor[0]
            else:
                self.joint_tensor[0, self.actuated_dofs] = ref_joint_tensor[0]
            
            # Apply to MuJoCo
            self._apply_state_to_mujoco(0)
            mujoco.mj_forward(self.model, self.data)
        
        self.lifetime[env_ids] = 0
    
    def init_state(self, env_ids):
        """Initialize state - to be overridden by subclasses"""
        n = len(env_ids)
        n_links = self.model.nbody
        n_dofs = self.model.nv
        
        # Default initialization
        ref_link_tensor = torch.zeros((n, n_links, 13), dtype=torch.float32, device=self.device)
        ref_joint_tensor = torch.zeros((n, n_dofs, 2), dtype=torch.float32, device=self.device)
        
        # Set default position (standing)
        ref_link_tensor[:, 0, 2] = 1.0  # z position
        ref_link_tensor[:, :, 6] = 1.0  # w quaternion
        
        return ref_link_tensor, ref_joint_tensor
    
    def do_simulation(self):
        """Step physics simulation"""
        for _ in range(self.frameskip):
            mujoco.mj_step(self.model, self.data)
        self.simulation_step += 1
    
    def step(self, actions):
        """Environment step"""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        
        if not self.viewer_pause or self.viewer_advance:
            self.apply_actions(actions)
            self.do_simulation()
            self._sync_state_from_mujoco()
            self.lifetime += 1
            if self.viewer_pause:
                self.viewer_advance = False
        
        rewards = self.reward()
        terminate = self.termination_check()
        
        if self.viewer_pause:
            overtime = None
        else:
            overtime = self.overtime_check()
        
        if torch.is_tensor(overtime):
            self.done = torch.logical_or(overtime, terminate)
        else:
            self.done = terminate
        
        self.info["terminate"] = terminate
        self.obs = self._observe()
        
        # Convert to numpy for gymnasium
        obs_np = self.obs.cpu().numpy()
        reward_np = rewards.cpu().numpy()
        done_np = self.done.cpu().numpy()
        terminated_np = terminate.cpu().numpy()
        truncated_np = overtime.cpu().numpy() if torch.is_tensor(overtime) else np.zeros_like(done_np)
        
        return obs_np, reward_np, terminated_np, truncated_np, self.info
    
    def apply_actions(self, actions):
        """Apply actions to simulation"""
        actions = self.process_actions(actions)
        
        if self.control_mode == "position":
            # Position control - set actuator targets
            for i, actuator_id in enumerate(range(min(len(self.actuated_dof_ids), self.model.nu))):
                if i < actions.shape[-1]:
                    self.data.ctrl[actuator_id] = actions[0, i].item()
        elif self.control_mode == "torque":
            # Torque control
            for i, actuator_id in enumerate(range(min(len(self.actuated_dof_ids), self.model.nu))):
                if i < actions.shape[-1]:
                    self.data.ctrl[actuator_id] = actions[0, i].item()
    
    def _observe_single(self) -> torch.Tensor:
        """Observe single environment"""
        self._sync_state_from_mujoco(0)
        return self.observe(torch.tensor([0], device=self.device))
    
    def _observe(self, env_ids=None) -> torch.Tensor:
        """Observe all environments - calls the overridden observe method"""
        return self.observe(env_ids)
    
    def observe(self, env_ids=None):
        """Get observations - to be overridden"""
        if env_ids is None:
            env_ids = torch.arange(self.n_envs, device=self.device)
        
        # Basic observation: root pos, orient, joint pos, joint vel
        n_obs = 13 + self.model.nv * 2
        obs = torch.zeros((len(env_ids), n_obs), dtype=torch.float32, device=self.device)
        
        for i, env_id in enumerate(env_ids):
            obs[i, :13] = self.root_tensor[env_id]
            obs[i, 13:13+self.model.nv] = self.joint_tensor[env_id, :, 0]
            obs[i, 13+self.model.nv:] = self.joint_tensor[env_id, :, 1]
        
        return obs
    
    def overtime_check(self):
        """Check if episode is over time"""
        if self.episode_length is not None:
            if callable(self.episode_length):
                return self.lifetime >= self.episode_length(self.simulation_step)
            return self.lifetime >= self.episode_length
        return torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
    
    def termination_check(self):
        """Check termination conditions"""
        # Check if root is too low (falling)
        root_z = self.root_tensor[:, 2]
        return root_z < 0.5  # Fallen if below 0.5m
    
    def reward(self):
        """Compute rewards - to be overridden"""
        return torch.ones((self.n_envs, 1), dtype=torch.float32, device=self.device)
    
    def render(self):
        """Render environment"""
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        elif self.render_mode == "human":
            # Use passive viewer
            pass
    
    def close(self):
        """Close environment"""
        if self.renderer is not None:
            self.renderer.close()