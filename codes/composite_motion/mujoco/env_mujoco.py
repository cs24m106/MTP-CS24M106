from typing import Optional, List, Union, Callable
import gymnasium as gym
import os, torch, mujoco
import mujoco.viewer, time
import numpy as np
from gymnasium import spaces
import threading
from concurrent.futures import ThreadPoolExecutor


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
    CAMERA_POS = 0, -4.5, 3.0
    CAMERA_FOLLOWING = True
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
        n_envs: int = 1,
        fps: int = 30,
        run_speed: int = 120, # simulation frequency (hz)
        episode_length: Optional[Union[Callable, int]] = 300,
        control_mode: str = "position",
        substeps: int = 2,
        compute_device: int = 0,
        graphics_device: Optional[int] = None,
        character_model: Optional[str] = None,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        workers: int = 4,
        **kwargs
    ):
        super().__init__()
        
        assert control_mode in ["position", "torque", "free"]
        self.run_speed = run_speed
        self.fps = fps  # update from ref motion if not passed
        self.frameskip = int(run_speed/fps) # fps * frame skip = simulation freq
        self.step_time = 1.0 / self.fps
        self.substeps = substeps
        self.control_mode = control_mode
        self.episode_length = episode_length    # best not to be set by training params (max_cycles overwrites it)
        self.device = torch.device(f"cuda:{compute_device}" if torch.cuda.is_available() and compute_device >= 0 else "cpu")
        self.n_envs = n_envs
        self.render_mode = render_mode
        self.verbose = verbose
        if verbose: print(f"[MujucoEnv] initiallized for [sim_speed:{self.run_speed}, fps:{self.fps}], [frameskip:{self.frameskip}, step_time:{self.step_time}]")
        
        self.camera_pos = self.CAMERA_POS
        self.camera_following = self.CAMERA_FOLLOWING
        
        self.character_model = self.CHARACTER_MODEL if character_model is None else character_model
        if isinstance(self.character_model, str):
            self.character_model = [self.character_model]
        
        # Load MuJoCo model
        self._load_mujoco_model()
        # Create one MjData per environment (vectorized simulation)
        self.data_list = [mujoco.MjData(self.model) for _ in range(n_envs)]
        self.data = self.data_list[0]  # alias: used by viewer/renderer (always tracks env 0)
        # thread parallelism for every env
        self.executor = ThreadPoolExecutor(max_workers=n_envs)
        if verbose: print(f"[ParallelStepper] initiallized for {n_envs} envs but sys handles only {min(n_envs, os.cpu_count() or 4)} threads max!")

        # Setup rendering
        self.renderer = None
        if self.render_mode == "rgb_array":
            self.renderer = mujoco.Renderer(self.model, width=960, height=720) # 960x720 = 4:3,  1280:720 = 16:9, aspects

        # Passive interactive viewer (non-blocking) for human mode
        self.viewer = None
        if self.render_mode == "human":
            try:
                # launch_passive returns a viewer handle; non-blocking
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception as e:
                # fall back gracefully (headless / platform issues)
                print(f"[Warning] failed to launch passive viewer: {e}")
                self.viewer = None

        
        # Get body and joint info
        self._setup_body_joint_info()
        # Setup action normalizer
        self.setup_action_normalizer()
        # Setup state spaces
        self.setup_state_spaces()


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

# ---------------------------------------------------------------------------------------------
            
    def _load_mujoco_model(self):
        """Load MuJoCo model from XML"""
        # Load the first character model
        xml_path = self.character_model[0]
        if not os.path.isabs(xml_path):
            xml_path = os.path.join(os.path.dirname(__file__), xml_path)
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 1/self.run_speed # defualt: (1/30=fps)/2=frameskip = 16.7ms → 60Hz physics
        
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
        
        print(f"\nBodies({len(self.body_names)}): {self.body_names}")
        print(f"\nJoints({len(self.joint_names)}): {self.joint_names}")
        print(f"\nActuated DOFs: {len(self.actuated_dof_ids)}")
    
    def eval(self):
        self.training = False
        
    def train(self):
        self.training = True

# ---------------------------------------------------------------------------------------------
    
    def create_tensors(self):
        """Create tensors for state tracking"""
        # EXCLUDE world body (index 0)
        n_bodies = self.model.nbody -1
        n_dofs = self.model.nv      # 34
        n_joints = self.model.nu    # 28
        
        # Root state (position, orientation, linear vel, angular vel)
        self.root_tensor = torch.zeros((self.n_envs, 13), dtype=torch.float32, device=self.device)
        
        # Link states (all bodies)
        self.link_tensor = torch.zeros((self.n_envs, n_bodies, 13), dtype=torch.float32, device=self.device)
        
        # Joint states (position, velocity)
        self.joint_tensor = torch.zeros((self.n_envs, n_joints, 2), dtype=torch.float32, device=self.device)
        
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
                    action_scale.append(1.0)         # Use scale=1.0, actions are already normalized to [-1, 1]
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

        
        if self.verbose:
            print("\n[--Init--] setup_action_normalizer():")
            print(f"  Action offset: {action_offset[:6]}")
            print(f"  Action scale:  {action_scale_arr[:6]}")
        
        self.action_offset = torch.tensor(action_offset, dtype=torch.float32, device=self.device)
        self.action_scale = torch.tensor(action_scale_arr, dtype=torch.float32, device=self.device)
        self.actuated_dofs = torch.tensor(self.actuated_dof_ids, dtype=torch.int64, device=self.device)
        
    def process_actions(self, actions):
        """Process actions from network"""
        a = actions * self.action_scale + self.action_offset  # Scale [1, 28]
        
        if self.action_tensor is None:
            return a  # Return [1, 28] directly
        
        # Since actuated_dof_ids (28) != n_dofs (34), action_tensor is created
        self.action_tensor[:, self.actuated_dofs] = a  # Put 28 values into DOF indices [6-33]
        return self.action_tensor  # Return [1, 34] with zeros at indices [0-5]
    
# ---------------------------------------------------------------------------------------------
    
    def _sync_state_from_mujoco(self, env_idx: int = None):
        """Sync state from MuJoCo to tensors.
        If env_idx is None (default), syncs ALL environments using vectorized numpy ops.
        If env_idx is given, syncs only that environment (fallback to single logic).
        """
        if env_idx is not None:
            # Fallback to original single-env logic if specific index requested
            self._sync_single_env_from_mujoco(env_idx)
            return

        # --- VECTORIZED PATH (All Environments) ---
        n_envs = self.n_envs
        n_bodies = self.model.nbody
        n_joints = self.model.nu
        
        # 1. Stack raw numpy data from all environments at once
        # This is the core parallelization step: minimizing Python overhead
        all_xpos = np.stack([data.xpos for data in self.data_list])       # [n_envs, n_bodies, 3]
        all_xquat = np.stack([data.xquat for data in self.data_list])     # [n_envs, n_bodies, 4] (wxyz)
        all_cvel = np.stack([data.cvel for data in self.data_list])       # [n_envs, n_bodies, 6] (ang, lin)
        all_qpos = np.stack([data.qpos for data in self.data_list])       # [n_envs, nq]
        all_qvel = np.stack([data.qvel for data in self.data_list])       # [n_envs, nv]

        # 2. Process Root State
        # Extract root body data (assuming self.root_body_id is consistent across envs)
        root_pos = all_xpos[:, self.root_body_id, :]                      # [n_envs, 3]
        root_quat_wxyz = all_xquat[:, self.root_body_id, :]               # [n_envs, 4]
        root_cvel = all_cvel[:, self.root_body_id, :]                     # [n_envs, 6]

        # Transform Quaternion: MuJoCo [w,x,y,z] -> Target [x,y,z,w]
        root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]
        
        # Transform Velocity: MuJoCo [ang(0:3), lin(3:6)] -> Target [lin, ang]
        root_lin_vel = root_cvel[:, 3:6]
        root_ang_vel = root_cvel[:, 0:3]

        # Assign to root_tensor [n_envs, 13]
        # Layout: pos(3), quat(4), lin_vel(3), ang_vel(3)
        self.root_tensor[:, :3]    = torch.from_numpy(root_pos).float().to(self.device)
        self.root_tensor[:, 3:7]   = torch.from_numpy(root_quat_xyzw).float().to(self.device)
        self.root_tensor[:, 7:10]  = torch.from_numpy(root_lin_vel).float().to(self.device)
        self.root_tensor[:, 10:13] = torch.from_numpy(root_ang_vel).float().to(self.device)

        # 3. Process Link States (Skip world body 0)
        # Source logic: range(self.model.nbody - 1), body_id = i + 1
        # Vectorized: Slice [:, 1:, :]
        link_pos = all_xpos[:, 1:, :]                                     # [n_envs, n_bodies-1, 3]
        link_quat_wxyz = all_xquat[:, 1:, :]                              # [n_envs, n_bodies-1, 4]
        link_cvel = all_cvel[:, 1:, :]                                    # [n_envs, n_bodies-1, 6]

        # Transform Quaternion
        link_quat_xyzw = link_quat_wxyz[..., [1, 2, 3, 0]]
        
        # Transform Velocity
        link_lin_vel = link_cvel[..., 3:6]
        link_ang_vel = link_cvel[..., 0:3]

        # Assign to link_tensor [n_envs, n_bodies-1, 13]
        self.link_tensor[:, :, :3]    = torch.from_numpy(link_pos).float().to(self.device)
        self.link_tensor[:, :, 3:7]   = torch.from_numpy(link_quat_xyzw).float().to(self.device)
        self.link_tensor[:, :, 7:10]  = torch.from_numpy(link_lin_vel).float().to(self.device)
        self.link_tensor[:, :, 10:13] = torch.from_numpy(link_ang_vel).float().to(self.device)

        # 4. Process Joint States
        # Source logic: qpos[7:7+n_joints], qvel[6:6+n_joints]
        joint_pos = all_qpos[:, 7:7+n_joints]                             # [n_envs, n_joints]
        joint_vel = all_qvel[:, 6:6+n_joints]                             # [n_envs, n_joints]

        # Assign to joint_tensor [n_envs, n_joints, 2]
        self.joint_tensor[:, :, 0] = torch.from_numpy(joint_pos).float().to(self.device)
        self.joint_tensor[:, :, 1] = torch.from_numpy(joint_vel).float().to(self.device)

        # 5. Process Contact Forces (Iterative)
        # Logic maintained: ncon varies per env, mj_contactForce requires specific MjData
        self.contact_force_tensor[:] = 0.0
        
        # We still need to loop for contacts as ncon is dynamic per environment
        for e_idx in range(n_envs):
            data = self.data_list[e_idx]
            for c in range(data.ncon):
                contact = data.contact[c]
                geom1_body = self.model.geom_bodyid[contact.geom1]
                geom2_body = self.model.geom_bodyid[contact.geom2]
                
                # Get contact force in world frame
                force = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, c, force)
                force_vec = torch.from_numpy(force[:3]).float().to(self.device)
                
                # Add to both bodies involved (skipping world body 0)
                # Logic maintained: 0 < body_id < self.model.nbody
                for body_id in [geom1_body, geom2_body]:
                    if 0 < body_id < n_bodies:
                        self.contact_force_tensor[e_idx, body_id - 1] += force_vec

        # 6. Verbose Logging (Maintain logic for env_idx 0)
        if self.verbose and self.simulation_step % 100 == 0:
            # In vectorized mode, we debug-check all env 
            root_h = self.root_tensor[:, 2].detach().cpu().numpy()
            print(f"\n[Step {self.simulation_step}] Sync env-all (vectorized): root_h_vec={root_h}")

    def _sync_single_env_from_mujoco(self, env_idx: int):
        """Sync a single environment's state from its MjData to tensors"""
        data = self.data_list[env_idx]

        # --- Root state from xpos/xquat (more reliable than qpos after mj_step) ---
        root_pos       = data.xpos[self.root_body_id].copy()
        root_quat_wxyz = data.xquat[self.root_body_id].copy()   # MuJoCo: [w, x, y, z]
        root_quat_xyzw = np.array([root_quat_wxyz[1], root_quat_wxyz[2],
                                    root_quat_wxyz[3], root_quat_wxyz[0]])  # our: [x,y,z,w]
        
        # MuJoCo's cvel (body spatial velocity) layout is: [[← angular first →] [← linear second →]]
        # cvel layout: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
        root_ang_vel = data.cvel[self.root_body_id, :3].copy()
        root_lin_vel = data.cvel[self.root_body_id, 3:].copy()

        self.root_tensor[env_idx, :3]    = torch.from_numpy(root_pos)
        self.root_tensor[env_idx, 3:7]   = torch.from_numpy(root_quat_xyzw)
        self.root_tensor[env_idx, 7:10]  = torch.from_numpy(root_lin_vel)
        self.root_tensor[env_idx, 10:13] = torch.from_numpy(root_ang_vel)

        # --- All link states (skip world body index 0) ---
        for i in range(self.model.nbody - 1):
            body_id  = i + 1    # ← was `i`, missing offset for world body
            pos = data.xpos[body_id].copy()
            quat = data.xquat[body_id].copy()
            quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
            ang_vel = data.cvel[body_id, :3].copy()
            lin_vel = data.cvel[body_id, 3:].copy()

            self.link_tensor[env_idx, i, :3]    = torch.from_numpy(pos)
            self.link_tensor[env_idx, i, 3:7]   = torch.from_numpy(quat_xyzw)
            self.link_tensor[env_idx, i, 7:10]  = torch.from_numpy(lin_vel)
            self.link_tensor[env_idx, i, 10:13] = torch.from_numpy(ang_vel)

        # --- Actuated joint positions & velocities from qpos[7:] / qvel[6:] ---
        n_joints  = self.model.nu   # 28
        joint_pos = data.qpos[7:7+n_joints].copy()
        joint_vel = data.qvel[6:6+n_joints].copy()
        self.joint_tensor[env_idx, :n_joints, 0] = torch.from_numpy(joint_pos)
        self.joint_tensor[env_idx, :n_joints, 1] = torch.from_numpy(joint_vel)

        # Reset contact forces
        self.contact_force_tensor[env_idx] = 0.0

        # Accumulate contact forces from active contacts
        for c in range(data.ncon):
            contact = data.contact[c]
            geom1_body = self.model.geom_bodyid[contact.geom1]
            geom2_body = self.model.geom_bodyid[contact.geom2]
            
            # Get contact force in world frame
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, c, force)
            force_vec = torch.from_numpy(force[:3]).float().to(self.device)
            
            # Add to both bodies involved (skipping world body 0)
            for body_id in [geom1_body, geom2_body]:
                if 0 < body_id < self.model.nbody:
                    self.contact_force_tensor[env_idx, body_id - 1] += force_vec

        if self.verbose and self.simulation_step % 100 == 0 and env_idx == 0:
            print(f"\n[Step {self.simulation_step}] Sync env-{env_idx}: root_h={root_pos[2]:.3f}")
    
# ---------------------------------------------------------------------------------------------
    
    def _apply_state_to_mujoco(self, env_idx: int = 0):
        """Apply state from tensors to MuJoCo.
        
        qpos layout (35 total): [x, y, z, qw, qx, qy, qz,  joint_0 .. joint_27]
                                  ←── root free joint ──→   ←── 28 actuated ──→
        qvel layout (34 total): [vx, vy, vz, wx, wy, wz,  jvel_0 .. jvel_27]
                                  ←── root (6 DOFs) ────→   ←── 28 actuated ──→
        NOTE: qpos has 7 root entries but qvel only has 6 (no quaternion derivative).
        """
        data = self.data_list[env_idx]   # ← use per-env data

        # --- Root position & orientation ---
        root_pos = self.root_tensor[env_idx, :3].cpu().numpy()
        root_quat_xyzw = self.root_tensor[env_idx, 3:7].cpu().numpy()
        # Convert from our [x, y, z, w] storage to MuJoCo's [w, x, y, z]
        root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0],
                                    root_quat_xyzw[1], root_quat_xyzw[2]])

        # Find the free joint (root) qpos address — cached after first call
        if not hasattr(self, '_root_jnt_qpos_adr'):
            self._root_jnt_qpos_adr = 0
            for i in range(self.model.njnt):
                if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    self._root_jnt_qpos_adr = self.model.jnt_qposadr[i]
                    break
        qpos_adr = self._root_jnt_qpos_adr   # typically 0
        qvel_adr = qpos_adr                   # same start; qvel root has 6 entries, qpos has 7

        # Write root state: qpos[0:7], qvel[0:6]
        data.qpos[qpos_adr  :qpos_adr+3] = root_pos
        data.qpos[qpos_adr+3:qpos_adr+7] = root_quat_wxyz
        data.qvel[qvel_adr  :qvel_adr+3] = self.root_tensor[env_idx, 7:10].cpu().numpy()   # lin vel
        data.qvel[qvel_adr+3:qvel_adr+6] = self.root_tensor[env_idx, 10:13].cpu().numpy()  # ang vel        
        # ↑ NOTE: qvel root slice ends at +6 (not +7) — free joint contributes 6 DOFs to qvel, 7 to qpos.

        # --- Actuated joint positions & velocities ---
        # joint_tensor has shape (n_envs, n_joints=28, 2) → local indices 0..27
        # These map directly to qpos[7:35] and qvel[6:34] (right after the free-joint root entries)
        n_joints  = self.model.nu   # 28
        data.qpos[qpos_adr+7 : qpos_adr+7+n_joints] = self.joint_tensor[env_idx, :n_joints, 0].cpu().numpy()
        data.qvel[qvel_adr+6 : qvel_adr+6+n_joints] = self.joint_tensor[env_idx, :n_joints, 1].cpu().numpy()

        # ── Ground-clearance clamp (FIX: QACC NaN at initialization) ────────────
        # Reference-motion poses (e.g., mid-kick frames) can place feet below the
        # ground plane. When the physics engine resolves this penetration on the
        # first step it fires explosive constraint forces → QACC NaN on the root DOF.
        # Fix: after writing all qpos, find the lowest geom contact point and lift
        # the root body upward by the penetration depth before mj_forward is called.
        # This costs one call to mj_kinematics (fast, no dynamics), and is only
        # done during reset (caller pattern: _apply_state_to_mujoco → mj_forward).
        # Run kinematics only (no dynamics) to get current geom positions
        mujoco.mj_kinematics(self.model, data)   # fills data.geom_xpos
        floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        min_z = float('inf')
        for g in range(self.model.ngeom):
            if g == floor_geom_id:
                continue
            body_id = self.model.geom_bodyid[g]
            if body_id == 0:     # world body
                continue
            # Approximate lowest point: geom center z - half-size for sphere/box/capsule
            gz = data.geom_xpos[g, 2]
            gtype = self.model.geom_type[g]
            gsize = self.model.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                gz -= gsize[0]
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                gz -= gsize[2]
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                gz -= gsize[0]   # radius (half-length is gsize[1])
            min_z = min(min_z, gz)
        
        penetration = max(0.0, -min_z + 0.001)   # 1 mm safety margin
        if penetration > 0.0:
            data.qpos[qpos_adr + 2] += penetration  # lift root z
        # ── end ground-clearance clamp ────────────────────────────────────────

        if self.verbose and self.simulation_step % 100 == 0 and env_idx == 0:
            print(f"\n[Step {self.simulation_step}] _apply_state_to_mujoco(env={env_idx}): qpos_adr={qpos_adr}, penetration_lift={penetration:.4f}")
    
# ---------------------------------------------------------------------------------------------
    
    def reset(self, seed=None, options=None):
        """Reset all environments"""
        super().reset(seed=seed)

        self.lifetime.zero_()
        self.done.fill_(True)
        self.info = dict(lifetime=self.lifetime)
        self.request_quit = False

        # Reset all per-env MuJoCo data instances
        for data in self.data_list:
            mujoco.mj_resetData(self.model, data)

        # Initialize state from reference motion
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
        """Reset specific environments Parallelly"""
        ref_link_tensor, ref_joint_tensor = self.init_state(env_ids)
        # For each env in env_ids, apply the ref state.
        # ref_link_tensor : (n, n_bodies, 13)  — link poses/vels from reference motion
        # ref_joint_tensor: (n, n_joints, 2)   — joint pos/vel, n_joints == model.nu == 28
        n = len(env_ids)
        for k, env_id in enumerate(env_ids):
            self.root_tensor[env_id] = ref_link_tensor[k, 0]   # pelvis (body 0 in link_tensor = body 1 in MuJoCo)
            self.link_tensor[env_id] = ref_link_tensor[k]       # world body already excluded from link_tensor

            # joint_tensor has shape (n_envs, n_joints=28, 2).
            # ref_joint_tensor[k] is (n_joints, 2) — direct assignment, no DOF-address remapping needed.
            # (self.actuated_dofs holds raw DOF addresses [6..33] — valid only for indexing qpos/qvel,
            #  NOT for indexing joint_tensor which uses compact 0-based joint indices.)
            n_ref_joints = ref_joint_tensor.shape[1]  # usually 28; clamp if motion file differs
            n_store = min(n_ref_joints, self.joint_tensor.shape[1])
            self.joint_tensor[env_id, :n_store] = ref_joint_tensor[k, :n_store]

            # Write tensors back into MuJoCo data and run a kinematics update
            eid = env_id.item() if hasattr(env_id, 'item') else env_id
            self._apply_state_to_mujoco(eid)
            #mujoco.mj_forward(self.model, self.data_list[eid]) # sequential
        
        def forward_single(eid): # thread fn
            mujoco.mj_forward(self.model, self.data_list[eid])        
        futures = [self.executor.submit(forward_single, eid) for eid in env_ids]
        for f in futures:
            f.result()

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
    
# ---------------------------------------------------------------------------------------------

    def do_simulation(self):
        """Step physics simulation for all environments Parallely"""
        def step_single(eid): # thread fn
            data = self.data_list[eid]
            for _ in range(self.frameskip):
                mujoco.mj_step(self.model, data)
        
        futures = [self.executor.submit(step_single, eid) for eid in range(self.n_envs)]
        for f in futures:
            f.result()  # propagate exceptions
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
        """Apply actions to all simulation environments"""
        actions = self.process_actions(actions)
        
        if self.control_mode == "position" or self.control_mode == "torque":
            for env_idx in range(self.n_envs):
                data = self.data_list[env_idx]
                if self.action_tensor is None:
                    # actions shape: [n_envs, n_actuators]
                    for i in range(min(self.model.nu, actions.shape[-1])):
                        data.ctrl[i] = actions[env_idx, i].item()
                else:
                    # actions shape: [n_envs, n_dofs] — extract at actuated DOF indices
                    for i in range(self.model.nu):
                        dof_id = self.actuated_dof_ids[i]
                        data.ctrl[i] = actions[env_idx, dof_id].item()
        
        # Debug (after setting ctrl values) @env-idx-0
        if self.verbose and self.simulation_step % 100 == 0:
            print(f"\n[Step {self.simulation_step}] CTRL check:")
            print(f"  Ctrl values (first 10): {self.data.ctrl[:10]}")
            print(f"  Joint pos (first 10): {self.data.qpos[6:6+10]}")
            print(f"  Root height: {self.data.xpos[self.root_body_id][2]:.4f}")
    
# ---------------------------------------------------------------------------------------------

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
        """Compute rewards - no task reward for base class"""
        return torch.ones((self.n_envs, 0), dtype=torch.float32, device=self.device)
    
# ---------------------------------------------------------------------------------------------
    
    def render(self):
        """
        Render environment
        mode: 
            rgb_array --> off-screen rendering and returns numpy img
            human --> interactive window / GUI that must be sync every simulation step

        """
        # Offscreen renderer -> return numpy frame
        if self.render_mode == "rgb_array":
            if self.renderer is None: # lazy create if not present
                self.renderer = mujoco.Renderer(self.model, width=960, height=720) # 960x720 = 4:3,  1280:720 = 16:9, aspects

            # ADDED: Set camera position for third-person view to follow character
            if self.camera_following:
                root_pos = self.data.xpos[self.root_body_id]
                
                # Update camera to look at character
                # Find the tracking camera
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
                if cam_id >= 0:
                    # Use the named camera
                    self.renderer.update_scene(self.data, camera="track")
                else:
                    # Fallback: update_scene expects mjData (and optional camera arg)
                    self.renderer.update_scene(self.data)
                
            # returns an (H,W,3) numpy array
            return self.renderer.render()
        
        # Human viewer -> show live window (non-blocking) and sync frame
        elif self.render_mode == "human":
            # passive viewer mode: sync guarantees the viewer shows the latest step,
            # but if someone calls render() explicitly we also sync.
            if self.viewer is None:
                try: # lazy-launch the passive viewer if it doesn't exist
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    # NEW: Set initial camera on first launch
                    if self.viewer is not None:
                        self.viewer.cam.lookat[:] = [0, 0, 1.0]  # Look at character
                        self.viewer.cam.distance = 4.0  # Distance from lookat point
                        self.viewer.cam.azimuth = 90  # Viewing angle
                        self.viewer.cam.elevation = -15  # Slightly from above
                except Exception as e:
                    print(f"[Warning] failed to launch passive viewer in render(): {e}")
                    return None
            try: # sync the viewer so it displays the latest data from the just-run step()
                # NEW: Update camera to follow character
                if self.viewer is not None and self.camera_following:
                    root_pos = self.data.xpos[self.root_body_id]
                    self.viewer.cam.lookat[:] = root_pos
                # Optional: if you want a stricter critical section while viewer reads data,
                # you can wrap modifications in viewer.lock() in your step() (not done here).
                self.viewer.sync()
                # throttle so rendering roughly matches env fps
                # step_time should be a simulation dt (sec) present in your env
                # (simple sleep; acceptable for testing/visualization)
                time.sleep(self.step_time)
            except Exception as e:
                print(f"[Warning] viewer.sync() in render() failed: {e}")
        
        return None

    def __del__(self):
        self.executor.shutdown(wait=False)
        #self.close()

    def close(self):
        """Close environment"""
        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        if hasattr(self, "viewer") and self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass