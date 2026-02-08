"""
Motion Viewer Utility for CompositeMotion
Displays reference motions before training
"""

import os
import sys
import numpy as np
import torch
import mujoco
import imageio
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ref_motion import ReferenceMotion


class MotionViewer:
    """Viewer for reference motion data"""
    
    def __init__(self, motion_file: str, character_model: str = "assets/humanoid.xml"):
        """
        Args:
            motion_file: Path to motion file (json, yaml, or joblib)
            character_model: Path to character XML model
        """
        self.motion_file = motion_file
        self.character_model = character_model
        
        # Load reference motion
        self.ref_motion = ReferenceMotion(
            motion_file=motion_file,
            character_model=[character_model],
            device="cpu"
        )
        
        # Load MuJoCo model for visualization
        self.model = mujoco.MjModel.from_xml_path(character_model)
        self.data = mujoco.MjData(self.model)
        
        # Setup renderer
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        
        # Body names for highlighting
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) 
                          for i in range(self.model.nbody)]
        
    def visualize_motion(self, motion_idx: int = 0, save_path: Optional[str] = None, 
                         fps: int = 30, highlight_links: Optional[List[str]] = None):
        """
        Visualize a specific motion
        
        Args:
            motion_idx: Index of motion to visualize
            save_path: Path to save video (if None, just display)
            fps: Frames per second
            highlight_links: List of link names to highlight
        """
        # Get motion info
        motion_info = self.ref_motion.motions[motion_idx]
        motion_length = motion_info['length']
        motion_fps = motion_info['fps']
        
        print(f"Motion {motion_idx}: {motion_length:.2f}s at {motion_fps}fps")
        
        # Generate frames
        frames = []
        n_frames = int(motion_length * fps)
        
        for i in range(n_frames):
            t = i / fps
            motion_ids = np.array([motion_idx])
            motion_times = np.array([t])
            
            # Get state
            link_tensor, joint_tensor = self.ref_motion.state(motion_ids, motion_times)
            
            # Apply to MuJoCo
            self._apply_state(link_tensor[0])
            mujoco.mj_forward(self.model, self.data)
            
            # Render
            self.renderer.update_scene(self.data)
            frame = self.renderer.render()
            
            # Add overlay with motion info
            frame = self._add_overlay(frame, motion_idx, t, highlight_links)
            
            frames.append(frame)
        
        # Save or display
        if save_path:
            imageio.mimsave(save_path, frames, fps=fps)
            print(f"Saved video to {save_path}")
        
        return frames
    
    def _apply_state(self, link_tensor):
        """Apply link state to MuJoCo"""
        # Root body (pelvis)
        root_pos = link_tensor[0, :3].cpu().numpy()
        root_quat_xyzw = link_tensor[0, 3:7].cpu().numpy()
        root_quat = np.array([root_quat_xyzw[3], root_quat_xyzw[0], root_quat_xyzw[1], root_quat_xyzw[2]])
        
        # Find free joint
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                qpos_adr = self.model.jnt_qposadr[i]
                self.data.qpos[qpos_adr:qpos_adr+3] = root_pos
                self.data.qpos[qpos_adr+3:qpos_adr+7] = root_quat
                break
    
    def _add_overlay(self, frame, motion_idx: int, time: float, 
                     highlight_links: Optional[List[str]] = None):
        """Add text overlay to frame"""
        import cv2
        
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Add background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text
        cv2.putText(frame, f"Motion: {os.path.basename(self.motion_file)}", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time:.2f}s / {self.ref_motion.motions[motion_idx]['length']:.2f}s", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Highlight links if specified
        if highlight_links:
            y_offset = 95
            cv2.putText(frame, "Highlighted Links:", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            for i, link in enumerate(highlight_links[:5]):  # Show max 5
                y_offset += 20
                cv2.putText(frame, f"  - {link}", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def display_motion_info(self):
        """Display information about all motions in the file"""
        print(f"\n{'='*60}")
        print(f"Motion File: {self.motion_file}")
        print(f"{'='*60}")
        print(f"Total Motions: {len(self.ref_motion.motions)}")
        print(f"{'-'*60}")
        
        for i, motion in enumerate(self.ref_motion.motions):
            print(f"Motion {i}:")
            print(f"  Length: {motion['length']:.3f}s")
            print(f"  FPS: {motion['fps']}")
            print(f"  Frames: {len(motion['data'])}")
        
        print(f"{'='*60}\n")
    
    def preview_all_motions(self, output_dir: str = "motion_previews"):
        """Generate preview videos for all motions"""
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(len(self.ref_motion.motions)):
            save_path = os.path.join(output_dir, f"motion_{i:03d}.mp4")
            self.visualize_motion(motion_idx=i, save_path=save_path)


def view_motions_before_training(config_file: str, output_dir: str = "motion_previews"):
    """
    Preview motions specified in a config file before training
    
    Args:
        config_file: Path to training config file
        output_dir: Directory to save preview videos
    """
    import importlib.util
    
    # Load config
    spec = importlib.util.spec_from_file_location("config", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    # Get motion file
    motion_file = config.env_params.get("motion_file")
    character_model = config.env_params.get("character_model", "assets/humanoid.xml")
    
    if not motion_file:
        print("No motion file specified in config")
        return
    
    # Create viewer
    viewer = MotionViewer(motion_file, character_model)
    
    # Display info
    viewer.display_motion_info()
    
    # Generate previews
    print(f"\nGenerating preview videos in {output_dir}...")
    viewer.preview_all_motions(output_dir)
    
    # Show discriminator info
    if hasattr(config, "discriminators"):
        print(f"\n{'='*60}")
        print("Discriminator Configuration:")
        print(f"{'='*60}")
        for name, disc in config.discriminators.items():
            print(f"\n{name}:")
            if "key_links" in disc:
                print(f"  Key Links: {disc['key_links']}")
            if "parent_link" in disc:
                print(f"  Parent Link: {disc['parent_link']}")
            if "weight" in disc:
                print(f"  Weight: {disc['weight']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View reference motions")
    parser.add_argument("config", type=str, help="Config file or motion file")
    parser.add_argument("--output", type=str, default="motion_previews", 
                       help="Output directory for previews")
    parser.add_argument("--motion-only", action="store_true",
                       help="Directly view a motion file instead of config")
    
    args = parser.parse_args()
    
    if args.motion_only:
        # Direct motion file viewing
        viewer = MotionViewer(args.config)
        viewer.display_motion_info()
        viewer.preview_all_motions(args.output)
    else:
        # Config-based viewing
        view_motions_before_training(args.config, args.output)
