"""
Motion Viewer Utility for CompositeMotion
Displays reference motions before training
"""

import os, sys, time
import numpy as np
import torch
import mujoco
import imageio
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from include.ref_motion import ReferenceMotion


class MotionViewer:
    """Viewer for reference motion data"""
    
    def __init__(self, motion_file: str, character_model: str):
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
        avg_fps = sum(self.ref_motion.fps) / len(self.ref_motion.fps)
        self.fps = int(np.ceil(avg_fps))
        
        # Load MuJoCo model for visualization
        self.model = mujoco.MjModel.from_xml_path(character_model)
        self.data = mujoco.MjData(self.model)
        
        # Setup renderer
        self.renderer = mujoco.Renderer(self.model, width=960, height=720) # 960x720 = 4:3,  1280:720 = 16:9, aspects
        
        # Body names for highlighting
        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) 
                          for i in range(self.model.nbody)]
        
    def generate_preview(self, motion_idx: int = 0, output_path: str = "preview.mp4", fps: int = None, camera: str = "smpl_view"):
        """Generates an MP4 video of a specific reference motion.
        
        Args:
            camera: Camera name from XML. 'smpl_view' matches the SMPL
                    matplotlib visualisation angle; 'track' is the legacy
                    side-behind view.
        """
        print(f"Generating {output_path} ...", end="", flush=True)
        if fps is None:
            fps = self.fps
        
        # Get motion properties from tensors
        motion_len = self.ref_motion.motion_length[motion_idx].item()
        num_steps = int(motion_len * fps)
        
        frames = []
        
        for i in range(num_steps):
            t = i / fps
            
            # Query the ReferenceMotion state
            link_tensor, joint_tensor = self.ref_motion.state(
                torch.tensor([motion_idx]), torch.tensor([t])
            )
            
            # Extract root state (batch 0, link 0)
            root_state = link_tensor[0, 0]
            
            # 1. Apply Root Position
            self.data.qpos[0:3] = root_state[0:3].cpu().numpy()
            
            # 2. FIX QUATERNION: IsaacGym [x,y,z,w] -> MuJoCo [w,x,y,z]
            q_ig = root_state[3:7].cpu().numpy()
            self.data.qpos[3:7] = [q_ig[3], q_ig[0], q_ig[1], q_ig[2]]
            
            # 3. Apply Joint Positions
            if joint_tensor is not None:
                # Extract only the Position (index 0) from [Position, Velocity]
                joints = joint_tensor[0, :, 0].cpu().numpy()
                num_joints = min(len(joints), len(self.data.qpos) - 7)
                self.data.qpos[7:7+num_joints] = joints[:num_joints]
            
            # 4. Forward Kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # 5. Render frame
            cam_id = camera if self.model.ncam > 0 else -1
            self.renderer.update_scene(self.data, camera=cam_id)
            frame = self.renderer.render()
            frames.append(frame)
            
        # Save to video
        try:
            imageio.mimsave(output_path, frames, fps=fps, macro_block_size=1)
            print(" Done!")
        except Exception as e:
            print(f" Failed! Error: {e}")
    
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
        num_motions = len(self.ref_motion.motion_length)
        print(f"Total Motions: {num_motions}")
        print(f"{'-'*60}")
        
        for i in range(num_motions):
            # Read directly from the generated tensors in ref_motion.py
            length = self.ref_motion.motion_length[i].item()
            n_frames = self.ref_motion.motion_n_frames_tensor[i].item()
            print(f"  Motion {i}: Length = {length:.3f}s, Frames = {n_frames}")
        
        print(f"{'='*60}\n")
    
    def preview_all_motions(self, output_dir: str = "motion_previews"):
        """Generates preview videos for all loaded motions."""
        os.makedirs(output_dir, exist_ok=True)
        
        # FIX: Use motion_length tensor instead of .motions list
        num_motions = len(self.ref_motion.motion_length)
        print(f"Generating previews for {num_motions} motions...")
        
        for i in range(num_motions):
            base = os.path.splitext(os.path.basename(self.motion_file))[0]
            fname = f"{base}_{i+1}.mp4" if num_motions > 1 else f"{base}.mp4"
            output_path = os.path.join(output_dir, fname)
            self.generate_preview(i, output_path)


def render_motions(file: str, type: str, output_dir: str = "motion_previews", char_model: str = None):
    """
    Preview motions specified in a config file or directly from motion data file
    Args:
        file: Path to config file or motion data file
        type: Type of file - 'config' or 'data'
        output_dir: Directory to save preview videos
        char_model: Optional override for character model path
    """
    import importlib.util
    
    # Determine motion_file and character_model based on type
    if type == "config":
        # Load config
        spec = importlib.util.spec_from_file_location("config", file)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        # Get motion file
        motion_file = config.env_params.get("motion_file")
        # Show discriminator info (config type only)
        show_discriminators = hasattr(config, "discriminators")
        discriminators_config = config.discriminators if show_discriminators else None
        character_model = config.env_params.get("character_model", "assets/humanoid.xml")

    elif type == "data":
        # Direct motion file path provided
        motion_file = file
        show_discriminators = False
        discriminators_config = None
        character_model = "assets/humanoid.xml"
        
    if char_model:
        character_model = char_model
    
    print(f"Character Model: {character_model}")

    if not motion_file:
        print("No motion file specified!")
        return
    
    # Create viewer
    viewer = MotionViewer(motion_file, character_model)
    
    # Display info
    viewer.display_motion_info()
    
    # Generate previews
    print(f"\nGenerating preview videos in {output_dir}...")
    viewer.preview_all_motions(output_dir)
    
    # Show discriminator info
    if show_discriminators:
        print(f"\n{'='*60}")
        print("Discriminator Configuration:")
        print(f"{'='*60}")
        for name, disc in discriminators_config.items():
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
    parser.add_argument("file", type=str, help="Path to config file or motion data file")
    parser.add_argument("-t", "--type", type=str, choices=["config", "data"], default="config",
                       help="Type of file: 'config' (training config) or 'data' (motion file directly)")
    parser.add_argument("--output", type=str, default="assets/motion_previews/composite", 
                       help="Output directory for previews")
    parser.add_argument("--model", type=str, default=None,
                       help="Optional path to character XML model")

    args = parser.parse_args()

    render_motions(args.file, args.type, args.output, args.model)
