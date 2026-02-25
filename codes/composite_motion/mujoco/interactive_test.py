"""
Interactive Policy Switcher for Testing
========================================
Implements discriminator-based policy switching as described in the ICCGAN paper
(Section 5.3, Eq. 6). Run during test/evaluation, not training.

Usage:
    python interactive_test.py --policies walk jaunty_walk --env joyful_walk
    
Controls:
    1-9 : switch to policy by number
    r   : reset character to random pose
    q   : quit
    
Requires trained policy checkpoints in ./checkpoints/{name}/
"""

import torch
import numpy as np
import mujoco
import mujoco.viewer
import time
import threading
import sys
import os
from pathlib import Path
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Policy Switcher (discriminator-based, as per ICCGAN paper Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

class PolicySwitcher:
    """
    Manages multiple trained policies and switches between them using
    the discriminator score threshold check from ICCGAN paper Eq. 6:
    
        switch feasible iff: (1/N) * sum_i clip(D_i^target(o_{t-4:t}), -1, 1) >= tau
    
    tau is typically 0.0 to 0.2 depending on required success rate.
    """
    
    def __init__(self, tau: float = 0.0):
        self.tau = tau
        self.policies = {}       # name -> ACModel
        self.discriminators = {} # name -> Discriminator
        self.policy_names = []
        self.current_policy = None
        self.current_name = None
        self.pending_switch = None  # name of target policy to try
        self.switch_lock = threading.Lock()
    
    def load_policy(self, name: str, checkpoint_path: str, device='cpu'):
        """Load a trained policy+discriminator checkpoint."""
        from models import ACModel, Discriminator
        
        if not os.path.exists(checkpoint_path):
            print(f"[PolicySwitcher] WARNING: checkpoint not found: {checkpoint_path}")
            return False
        
        state = torch.load(checkpoint_path, map_location=device)
        
        # Load model (handles shape mismatches gracefully)
        model = state.get('model') or state.get('policy')
        disc = state.get('discriminator') or state.get('disc')
        
        if model is None:
            print(f"[PolicySwitcher] WARNING: no 'model' key in {checkpoint_path}")
            return False
        
        self.policies[name] = model.to(device).eval()
        if disc is not None:
            self.discriminators[name] = disc.to(device).eval()
        
        self.policy_names.append(name)
        if self.current_policy is None:
            self.current_policy = self.policies[name]
            self.current_name = name
        
        print(f"[PolicySwitcher] Loaded '{name}' from {checkpoint_path}")
        return True
    
    def check_switch_feasibility(self, obs: torch.Tensor, seq_end_frame: torch.Tensor,
                                  target_name: str) -> float:
        """
        Compute discriminator score for target policy on current observation.
        Returns score in [-1, 1]. Switch is feasible if score >= self.tau.
        """
        if target_name not in self.discriminators:
            return 1.0  # No discriminator loaded → allow switch freely
        
        disc = self.discriminators[target_name]
        with torch.no_grad():
            # obs shape: [1, n_frames, obs_dim]
            scores = disc(obs, seq_end_frame)  # [1, n_disc]
            scores = torch.clamp(scores, -1, 1)
            avg_score = scores.mean().item()
        return avg_score
    
    def request_switch(self, target_name: str):
        """Request a policy switch (called from keyboard handler)."""
        with self.switch_lock:
            if target_name in self.policies:
                self.pending_switch = target_name
                print(f"[PolicySwitcher] Requested switch to '{target_name}' (score check pending)")
            else:
                print(f"[PolicySwitcher] Unknown policy: '{target_name}'")
    
    def try_execute_switch(self, obs: torch.Tensor, seq_end_frame: torch.Tensor) -> bool:
        """
        Called every frame. Executes pending switch if discriminator says it's feasible.
        Returns True if a switch occurred.
        """
        with self.switch_lock:
            if self.pending_switch is None:
                return False
            target = self.pending_switch
        
        score = self.check_switch_feasibility(obs, seq_end_frame, target)
        
        if score >= self.tau:
            with self.switch_lock:
                self.current_policy = self.policies[target]
                self.current_name = target
                self.pending_switch = None
            print(f"[PolicySwitcher] Switched to '{target}' (score={score:.3f} >= tau={self.tau})")
            return True
        else:
            # Not feasible yet - keep trying next frame
            # print(f"[PolicySwitcher] Switch to '{target}' pending (score={score:.3f} < tau={self.tau})")
            return False
    
    def act(self, obs: torch.Tensor, seq_end_frame: torch.Tensor) -> torch.Tensor:
        """Get action from current policy."""
        with torch.no_grad():
            if self.current_policy is None:
                raise RuntimeError("No policy loaded")
            return self.current_policy.act(obs, seq_end_frame, stochastic=False)


# ─────────────────────────────────────────────────────────────────────────────
# Fall Detection + Simple Recovery
# ─────────────────────────────────────────────────────────────────────────────

class FallDetector:
    """
    Detects when character has fallen and triggers get-up recovery.
    
    For now implements a simple heuristic recovery (not trained get-up policy).
    Once a get-up policy is trained, swap in PolicySwitcher.request_switch('getup').
    """
    
    def __init__(self, hip_height_threshold: float = 0.6, 
                 recovery_frames: int = 60):
        """
        hip_height_threshold: if hip (root) height < this, character has fallen
        recovery_frames: frames to hold recovery pose before returning to main policy
        """
        self.hip_height_threshold = hip_height_threshold
        self.recovery_frames = recovery_frames
        self._recovery_countdown = 0
        self.is_recovering = False
    
    def check(self, root_pos: np.ndarray) -> bool:
        """Returns True if character has fallen."""
        hip_height = root_pos[2]  # z coordinate
        return hip_height < self.hip_height_threshold
    
    def step(self, root_pos: np.ndarray, data: mujoco.MjData) -> bool:
        """
        Call every frame. If fallen and no get-up policy available,
        applies a gentle stand-up torque heuristic.
        Returns True if in recovery mode.
        """
        if self.check(root_pos):
            if not self.is_recovering:
                print(f"[FallDetector] Character fallen (height={root_pos[2]:.2f}), starting recovery")
                self.is_recovering = True
                self._recovery_countdown = self.recovery_frames
        
        if self.is_recovering:
            self._apply_standup_torques(data)
            self._recovery_countdown -= 1
            if self._recovery_countdown <= 0:
                self.is_recovering = False
                print("[FallDetector] Recovery complete")
            return True
        return False
    
    def _apply_standup_torques(self, data: mujoco.MjData):
        """
        Simple heuristic: zero all joint targets (neutral pose).
        A proper get-up policy would replace this.
        """
        data.ctrl[:] = 0.0  # return to neutral


# ─────────────────────────────────────────────────────────────────────────────
# Main Interactive Test Loop
# ─────────────────────────────────────────────────────────────────────────────

class InteractiveTestRunner:
    """
    Runs a single MuJoCo environment with interactive policy switching.
    Designed to be simple - single env, visualized, keyboard control.
    """
    
    def __init__(self, model_path: str, switcher: PolicySwitcher,
                 obs_horizon: int = 4, device: str = 'cpu'):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.switcher = switcher
        self.obs_horizon = obs_horizon
        self.device = device
        
        # Observation history buffer  [n_frames, obs_dim] - filled lazily
        self.state_hist = None
        self.lifetime = 0
        
        # Fall recovery
        self.fall_detector = FallDetector()
        
        # Control flags
        self.running = True
        self.paused = False
    
    def reset(self):
        """Reset to a neutral standing pose."""
        mujoco.mj_resetData(self.model, self.data)
        # Set initial height so character starts standing
        self.data.qpos[2] = 0.9  # rough hip height for standing
        mujoco.mj_forward(self.model, self.data)
        self.lifetime = 0
        if self.state_hist is not None:
            self.state_hist.zero_()
        print("[Runner] Reset to standing pose")
    
    def get_observation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build observation tensor from state history.
        Returns (obs [1, n_frames, state_dim], seq_end_frame [1])
        """
        # Extract current state from MuJoCo data
        # This needs to match your env_iccgan.py observation format exactly
        # The below is a placeholder - replace with your actual observe() logic
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        state = np.concatenate([qpos[3:], qvel])  # exclude global xy pos
        
        obs_dim = len(state)
        
        if self.state_hist is None:
            self.state_hist = torch.zeros(self.obs_horizon, 1, obs_dim)
        
        # Roll history and insert current frame
        self.state_hist = torch.roll(self.state_hist, -1, dims=0)
        self.state_hist[-1, 0] = torch.from_numpy(state).float()
        
        seq_len = min(self.lifetime + 1, self.obs_horizon)
        seq_end_frame = torch.tensor([seq_len - 1])
        
        # obs: [1, n_frames, obs_dim] 
        obs = self.state_hist[:, 0:1, :].permute(1, 0, 2)  # [1, n_frames, obs_dim]
        return obs, seq_end_frame
    
    def step(self):
        """Run one control step (30 Hz equivalent)."""
        if self.paused:
            return
        
        obs, seq_end_frame = self.get_observation()
        
        # Try executing any pending policy switch
        self.switcher.try_execute_switch(obs, seq_end_frame)
        
        # Check for fall - if fallen and have get-up policy, trigger it
        root_pos = self.data.qpos[:3]
        if self.fall_detector.check(root_pos):
            if 'getup' in self.switcher.policies:
                self.switcher.request_switch('getup')
            else:
                self.fall_detector.step(root_pos, self.data)
        
        # Get action from current policy
        try:
            action = self.switcher.act(obs, seq_end_frame)
            if isinstance(action, tuple):
                action = action[0]  # (action, value, logprob) → action
            action_np = action.cpu().numpy().flatten()
        except Exception as e:
            print(f"[Runner] Policy inference error: {e}")
            action_np = np.zeros(self.model.nu)
        
        # Apply action via PD controller (simplified - use your actual PD impl)
        self.data.ctrl[:len(action_np)] = action_np
        
        # Step physics (substeps to match training frequency)
        substeps = self.model.opt.timestep  # use model's timestep
        for _ in range(20):  # 20 substeps at 600Hz for 30Hz control
            mujoco.mj_step(self.model, self.data)
        
        self.lifetime += 1
    
    def run_with_viewer(self):
        """Launch MuJoCo viewer with keyboard policy switching."""
        print(f"\n{'='*60}")
        print("INTERACTIVE POLICY SWITCHER")
        print(f"{'='*60}")
        for i, name in enumerate(self.switcher.policy_names):
            print(f"  {i+1}: Switch to '{name}'")
        print("  r: Reset character")
        print("  Space: Pause/Resume")
        print("  q: Quit")
        print(f"  tau (switch threshold): {self.switcher.tau}")
        print(f"{'='*60}\n")
        
        self.reset()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Keyboard handler runs in separate thread to not block rendering
            key_thread = threading.Thread(target=self._keyboard_handler, daemon=True)
            key_thread.start()
            
            while viewer.is_running() and self.running:
                step_start = time.perf_counter()
                
                self.step()
                
                viewer.sync()
                
                # Maintain ~30Hz
                elapsed = time.perf_counter() - step_start
                time.sleep(max(0, 1/30 - elapsed))
    
    def _keyboard_handler(self):
        """Minimal keyboard input (stdin-based, works cross-platform)."""
        import sys
        print("[Runner] Keyboard handler active (type commands + Enter)")
        while self.running:
            try:
                cmd = input().strip().lower()
                if cmd == 'q':
                    self.running = False
                elif cmd == 'r':
                    self.reset()
                elif cmd == ' ' or cmd == 'p':
                    self.paused = not self.paused
                    print(f"[Runner] {'Paused' if self.paused else 'Resumed'}")
                elif cmd.isdigit():
                    idx = int(cmd) - 1
                    if 0 <= idx < len(self.switcher.policy_names):
                        self.switcher.request_switch(self.switcher.policy_names[idx])
                    else:
                        print(f"[Runner] No policy at index {cmd}")
                elif cmd in self.switcher.policy_names:
                    self.switcher.request_switch(cmd)
            except EOFError:
                break


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive policy switching test")
    parser.add_argument('--model', default='assets/humanoid.xml', help='MuJoCo model XML path')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='Checkpoint paths: name:path name:path ...')
    parser.add_argument('--tau', type=float, default=0.0,
                        help='Switch feasibility threshold (paper uses 0.0 to 0.2)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    
    switcher = PolicySwitcher(tau=args.tau)
    
    for spec in args.checkpoints:
        if ':' in spec:
            name, path = spec.split(':', 1)
        else:
            name = Path(spec).stem
            path = spec
        switcher.load_policy(name, path, device=args.device)
    
    if not switcher.policy_names:
        print("ERROR: No policies loaded. Check checkpoint paths.")
        sys.exit(1)
    
    runner = InteractiveTestRunner(
        model_path=args.model,
        switcher=switcher,
        device=args.device
    )
    runner.run_with_viewer()


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────────────────────────
# NOTES ON GET-UP POLICY FROM PYBULLET DATA
# ─────────────────────────────────────────────────────────────────────────────
"""
Can we use the PyBullet get-up motion data directly?
=====================================================

SHORT ANSWER: No, not directly - but it's convertible with work.

The PyBullet humanoid3d_getup_*.txt files use a JSON-like format with:
  - 34 DOF humanoid (8 spherical + 4 revolute joints)
  - Joint rotations stored as quaternions
  - Different joint ordering than MuJoCo's humanoid.xml

To convert:
  1. Parse the .txt files (they're JSON arrays of frames)
  2. Map joint names from PyBullet's humanoid to MuJoCo's humanoid.xml
     - Both have similar topology (based on CMU mocap skeleton)
     - Need to verify joint axes and reference poses match
  3. Convert quaternion format if needed (PyBullet uses xyzw, MuJoCo wxyz)
  4. Adjust scaling if body proportions differ

However, TRAINING a get-up policy from this converted data is still required.
The motion data gives reference frames; the policy still needs RL training to
achieve it physically in your MuJoCo environment.

PRACTICAL RECOMMENDATION FOR LAPTOP TRAINING:
=============================================
Rather than training a separate get-up policy (which takes 12+ hours on V100,
much more on a laptop), use a simpler fallback:

Option A: Rule-based get-up heuristic (FallDetector above)
  - Detect fall by hip height < threshold
  - Zero all joint targets → character slumps to minimum energy
  - After N frames, reset episode
  → No training required, works now

Option B: Train get-up alongside main policy (low priority, use smaller model)
  - Use the ICCGAN approach but with joyful_walk checkpoint as warm-start
  - Train for fewer epochs (500-1000) since get-up is simpler constraint
  → Requires ~2-4 hours on a decent laptop (with parallelization fixes)

Option C: Use fall-in-place as terminal condition, skip recovery
  - Simplest for now: when character falls, just reset to reference pose
  - No get-up needed during training, episode ends
  → Current behavior (just make it cleaner)
"""
