"""
SMPL to CompositeMotion JSON Converter
Converts SMPL-format motion data to CompositeMotion's JSON format
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import pickle


class SMPLConverter:
    """Converter from SMPL format to CompositeMotion format"""
    
    # SMPL joint names (24 joints)
    SMPL_JOINT_NAMES = [
        "pelvis",          # 0
        "left_hip",        # 1
        "right_hip",       # 2
        "spine1",          # 3
        "left_knee",       # 4
        "right_knee",      # 5
        "spine2",          # 6
        "left_ankle",      # 7
        "right_ankle",     # 8
        "spine3",          # 9
        "left_foot",       # 10
        "right_foot",      # 11
        "neck",            # 12
        "left_collar",     # 13
        "right_collar",    # 14
        "head",            # 15
        "left_shoulder",   # 16
        "right_shoulder",  # 17
        "left_elbow",      # 18
        "right_elbow",     # 19
        "left_wrist",      # 20
        "right_wrist",     # 21
        "left_hand",       # 22
        "right_hand",      # 23
    ]
    
    # CompositeMotion body names (must match humanoid.xml)
    CM_BODY_NAMES = [
        "world",
        "pelvis",
        "torso",
        "head",
        "right_upper_arm",
        "right_lower_arm",
        "right_hand",
        "left_upper_arm",
        "left_lower_arm",
        "left_hand",
        "right_thigh",
        "right_shin",
        "right_foot",
        "left_thigh",
        "left_shin",
        "left_foot",
    ]
    
    # Mapping from SMPL joints to CompositeMotion bodies
    SMPL_TO_CM_MAPPING = {
        "pelvis": "pelvis",
        "spine3": "torso",
        "head": "head",
        "right_shoulder": "right_upper_arm",
        "right_elbow": "right_lower_arm",
        "right_wrist": "right_hand",
        "left_shoulder": "left_upper_arm",
        "left_elbow": "left_lower_arm",
        "left_wrist": "left_hand",
        "right_hip": "right_thigh",
        "right_knee": "right_shin",
        "right_ankle": "right_foot",
        "left_hip": "left_thigh",
        "left_knee": "left_shin",
        "left_ankle": "left_foot",
    }
    
    def __init__(self, smpl_model_path: Optional[str] = None):
        """
        Args:
            smpl_model_path: Path to SMPL model file (optional)
        """
        self.smpl_model_path = smpl_model_path
        
    def load_smpl_pkl(self, pkl_path: str) -> Dict:
        """Load SMPL data from pickle file"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    
    def load_smpl_npz(self, npz_path: str) -> Dict:
        """Load SMPL data from npz file"""
        data = np.load(npz_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    
    def convert_poses(self, smpl_poses: np.ndarray) -> List[Dict]:
        """
        Convert SMPL poses to CompositeMotion format
        
        Args:
            smpl_poses: Array of shape (N, 72) or (N, 24, 3) - rotation vectors
            
        Returns:
            List of frame dictionaries in CompositeMotion format
        """
        if smpl_poses.ndim == 2 and smpl_poses.shape[1] == 72:
            # Reshape from (N, 72) to (N, 24, 3)
            smpl_poses = smpl_poses.reshape(-1, 24, 3)
        
        n_frames = len(smpl_poses)
        frames = []
        
        for i in range(n_frames):
            frame = self._convert_single_frame(smpl_poses[i])
            frames.append(frame)
        
        return frames
    
    def _convert_single_frame(self, smpl_pose: np.ndarray) -> Dict:
        """Convert a single SMPL pose frame"""
        # Convert rotation vectors to quaternions
        from utils import axang2quat
        
        frame = {"data": []}
        
        # World body (identity)
        frame["data"].append({
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        })
        
        # Convert each SMPL joint
        for smpl_joint, cm_body in self.SMPL_TO_CM_MAPPING.items():
            joint_idx = self.SMPL_JOINT_NAMES.index(smpl_joint)
            rot_vec = smpl_pose[joint_idx]
            
            # Convert rotation vector to quaternion
            angle = np.linalg.norm(rot_vec)
            if angle < 1e-6:
                quat = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                axis = rot_vec / angle
                quat = axang2quat(
                    torch.tensor(axis, dtype=torch.float32),
                    torch.tensor(angle, dtype=torch.float32)
                ).numpy()
            
            # Position (simplified - would need proper forward kinematics)
            position = self._estimate_joint_position(cm_body)
            
            frame["data"].append({
                "position": position.tolist(),
                "orientation": quat.tolist(),
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
            })
        
        return frame
    
    def _estimate_joint_position(self, body_name: str) -> np.ndarray:
        """Estimate joint position based on body name (simplified)"""
        # Default standing pose positions (approximate)
        positions = {
            "pelvis": [0.0, 0.0, 1.0],
            "torso": [0.0, 0.0, 1.3],
            "head": [0.0, 0.0, 1.7],
            "right_upper_arm": [0.2, 0.0, 1.5],
            "right_lower_arm": [0.3, 0.0, 1.2],
            "right_hand": [0.35, 0.0, 0.9],
            "left_upper_arm": [-0.2, 0.0, 1.5],
            "left_lower_arm": [-0.3, 0.0, 1.2],
            "left_hand": [-0.35, 0.0, 0.9],
            "right_thigh": [0.1, 0.0, 0.8],
            "right_shin": [0.1, 0.0, 0.4],
            "right_foot": [0.1, 0.0, 0.0],
            "left_thigh": [-0.1, 0.0, 0.8],
            "left_shin": [-0.1, 0.0, 0.4],
            "left_foot": [-0.1, 0.0, 0.0],
        }
        return np.array(positions.get(body_name, [0.0, 0.0, 0.0]))
    
    def convert_and_save(self, input_path: str, output_path: str, fps: float = 30.0):
        """
        Convert SMPL file to CompositeMotion JSON
        
        Args:
            input_path: Path to SMPL file (.pkl or .npz)
            output_path: Path to output JSON file
            fps: Frames per second
        """
        # Load SMPL data
        if input_path.endswith('.pkl'):
            data = self.load_smpl_pkl(input_path)
        elif input_path.endswith('.npz'):
            data = self.load_smpl_npz(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        # Extract poses
        if 'poses' in data:
            poses = data['poses']
        elif 'pose' in data:
            poses = data['pose']
        else:
            raise ValueError("No pose data found in file")
        
        # Convert
        frames = self.convert_poses(poses)
        
        # Create output
        output = {
            "fps": fps,
            "frames": frames,
        }
        
        # Save
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Converted {len(frames)} frames to {output_path}")
        return output


def batch_convert_smpl(input_dir: str, output_dir: str, fps: float = 30.0):
    """
    Batch convert all SMPL files in a directory
    
    Args:
        input_dir: Directory containing SMPL files
        output_dir: Directory to save JSON files
        fps: Frames per second
    """
    os.makedirs(output_dir, exist_ok=True)
    
    converter = SMPLConverter()
    
    for filename in os.listdir(input_dir):
        if filename.endswith(('.pkl', '.npz')):
            input_path = os.path.join(input_dir, filename)
            output_name = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(output_dir, output_name)
            
            try:
                converter.convert_and_save(input_path, output_path, fps)
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


if __name__ == "__main__":
    import argparse
    import torch
    
    parser = argparse.ArgumentParser(description="Convert SMPL to CompositeMotion format")
    parser.add_argument("input", type=str, help="Input SMPL file or directory")
    parser.add_argument("--output", type=str, help="Output JSON file or directory")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second")
    parser.add_argument("--batch", action="store_true", help="Batch convert directory")
    
    args = parser.parse_args()
    
    if args.batch:
        batch_convert_smpl(args.input, args.output or "converted_motions", args.fps)
    else:
        converter = SMPLConverter()
        output_path = args.output or args.input.replace('.pkl', '.json').replace('.npz', '.json')
        converter.convert_and_save(args.input, output_path, args.fps)
