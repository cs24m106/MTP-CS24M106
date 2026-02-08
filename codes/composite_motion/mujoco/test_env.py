"""
Test script to verify MuJoCo environment setup
"""

import os
import sys
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def test_basic_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import mujoco
        print("  ✓ mujoco imported")
    except ImportError as e:
        print(f"  ✗ mujoco import failed: {e}")
        return False
    
    try:
        import gymnasium
        print("  ✓ gymnasium imported")
    except ImportError as e:
        print(f"  ✗ gymnasium import failed: {e}")
        return False
    
    try:
        from env_iccgan import MujocoEnv, ICCGANHumanoidMujoco, ICCGANHumanoidTargetMujoco, DiscriminatorConfig
        print("  ✓ env_iccgan imported")
    except ImportError as e:
        print(f"  ✗ env_iccgan import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from models import ACModel, Discriminator
        print("  ✓ models imported")
    except ImportError as e:
        print(f"  ✗ models import failed: {e}")
        return False
    
    try:
        from ref_motion import ReferenceMotion
        print("  ✓ ref_motion imported")
    except ImportError as e:
        print(f"  ✗ ref_motion import failed: {e}")
        return False
    
    try:
        from utils import heading_zup, rotatepoint, quatmultiply
        print("  ✓ utils imported")
    except ImportError as e:
        print(f"  ✗ utils import failed: {e}")
        return False
    
    return True


def test_mujoco_model_loading():
    """Test loading the humanoid XML model"""
    print("\nTesting MuJoCo model loading...")
    
    import mujoco
    
    xml_path = "assets/humanoid.xml"
    if not os.path.exists(xml_path):
        print(f"  ✗ XML file not found: {xml_path}")
        print("    Please ensure the assets folder is in the correct location")
        return False
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"  ✓ Model loaded successfully")
        print(f"    - Bodies: {model.nbody}")
        print(f"    - Joints: {model.njnt}")
        print(f"    - DOFs: {model.nv}")
        print(f"    - Actuators: {model.nu}")
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False


def test_reference_motion_loading():
    """Test loading reference motion data"""
    print("\nTesting reference motion loading...")
    
    from ref_motion import ReferenceMotion
    
    motion_file = "assets/motions/iccgan/jaunty_walk.json"
    if not os.path.exists(motion_file):
        print(f"  ⚠ Motion file not found: {motion_file}")
        print("    Skipping reference motion test")
        return True  # Not a critical failure
    
    try:
        motion_lib = ReferenceMotion(
            motion_file=motion_file,
            character_model="assets/humanoid.xml",
            device="cpu"
        )
        print(f"  ✓ Reference motion loaded successfully")
        
        # Test sampling
        motion_ids, motion_times = motion_lib.sample(10)
        print(f"    - Sampled {len(motion_ids)} motions")
        
        # Test state retrieval
        link_tensor, joint_tensor = motion_lib.state(motion_ids, motion_times)
        print(f"    - Link tensor shape: {link_tensor.shape}")
        print(f"    - Joint tensor shape: {joint_tensor.shape}")
        
        # Check for NaN
        if torch.isnan(link_tensor).any():
            print(f"  ⚠ Warning: NaN detected in link tensor")
        else:
            print(f"  ✓ No NaN in link tensor")
        
        return True
    except Exception as e:
        print(f"  ✗ Reference motion loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test creating the environment"""
    print("\nTesting environment creation...")
    
    from env_iccgan import ICCGANHumanoidMujoco, DiscriminatorConfig
    
    motion_file = "assets/motions/iccgan/jaunty_walk.json"
    if not os.path.exists(motion_file):
        print(f"  ⚠ Motion file not found: {motion_file}")
        print("    Skipping environment creation test")
        return True
    
    try:
        discriminators = {
            "walk": DiscriminatorConfig(
                parent_link=None,
                weight=1.0
            )
        }
        
        env = ICCGANHumanoidMujoco(
            n_envs=1,
            fps=30,
            frameskip=2,
            episode_length=300,
            control_mode="position",
            motion_file=motion_file,
            discriminators=discriminators,
            character_model="assets/humanoid.xml",
            compute_device=-1,  # CPU
            render_mode=None
        )
        
        print(f"  ✓ Environment created successfully")
        print(f"    - Observation dim: {env.ob_dim}")
        print(f"    - Action dim: {env.act_dim}")
        print(f"    - Reward dim: {env.rew_dim}")
        print(f"    - State dim: {env.state_dim}")
        
        # Test reset
        env.reset()
        obs, info = env.reset_done()
        print(f"  ✓ Environment reset successful")
        print(f"    - Observation shape: {obs.shape}")
        
        # Check for NaN in initial observation
        if torch.isnan(obs).any():
            nan_count = torch.isnan(obs).sum().item()
            print(f"  ✗ ERROR: NaN detected in initial observation! Count: {nan_count}")
            print(f"    Sample obs: {obs[0, :20]}")
            return False
        else:
            print(f"  ✓ No NaN in initial observation")
        
        # Test step
        action = torch.zeros((1, env.act_dim), device=env.device)
        obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        print(f"  ✓ Environment step successful")
        print(f"    - Reward: {reward}")
        print(f"    - Terminated: {terminated}")
        
        # Check for NaN after step
        obs_tensor = torch.from_numpy(obs).to(env.device)
        if torch.isnan(obs_tensor).any():
            nan_count = torch.isnan(obs_tensor).sum().item()
            print(f"  ✗ ERROR: NaN detected after step! Count: {nan_count}")
            return False
        else:
            print(f"  ✓ No NaN after step")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_target_environment():
    """Test target-based environment"""
    print("\nTesting target environment...")
    
    from env_iccgan import ICCGANHumanoidTargetMujoco, DiscriminatorConfig
    
    motion_file = "assets/motions/clips_walk.yaml"
    if not os.path.exists(motion_file):
        print(f"  ⚠ Motion file not found: {motion_file}")
        print("    Skipping target environment test")
        return True
    
    try:
        discriminators = {
            "walk/full": DiscriminatorConfig(
                parent_link=None,
                weight=1.0
            )
        }
        
        env = ICCGANHumanoidTargetMujoco(
            n_envs=1,
            fps=30,
            frameskip=2,
            episode_length=500,
            control_mode="position",
            motion_file=motion_file,
            discriminators=discriminators,
            character_model="assets/humanoid.xml",
            compute_device=-1,
            render_mode=None,
            goal_radius=0.5,
            sp_lower_bound=1.2,
            sp_upper_bound=1.5,
        )
        
        print(f"  ✓ Target environment created successfully")
        print(f"    - Goal dim: {env.goal_dim}")
        
        env.reset()
        obs, info = env.reset_done()
        
        # Check for NaN
        if torch.isnan(obs).any():
            nan_count = torch.isnan(obs).sum().item()
            print(f"  ✗ ERROR: NaN in target env observation! Count: {nan_count}")
            return False
        else:
            print(f"  ✓ No NaN in target environment")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"  ✗ Target environment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test creating the AC model"""
    print("\nTesting model creation...")
    
    from models import ACModel, Discriminator
    
    try:
        state_dim = 256
        act_dim = 28
        goal_dim = 0
        value_dim = 1
        
        model = ACModel(state_dim, act_dim, goal_dim, value_dim)
        print(f"  ✓ ACModel created successfully")
        
        disc = Discriminator(128)
        print(f"  ✓ Discriminator created successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configs():
    """Test that all config files can be loaded"""
    print("\nTesting config files...")
    
    import importlib.util
    
    config_dir = "config"
    if not os.path.exists(config_dir):
        print(f"  ⚠ Config directory not found: {config_dir}")
        return True
    
    configs = []
    for root, dirs, files in os.walk(config_dir):
        for f in files:
            if f.endswith('.py'):
                configs.append(os.path.join(root, f))
    
    print(f"  Found {len(configs)} config files")
    
    passed = 0
    for config_path in configs:
        try:
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            # Check required attributes
            assert hasattr(config, 'env_cls'), f"{config_path}: missing env_cls"
            assert hasattr(config, 'env_params'), f"{config_path}: missing env_params"
            assert hasattr(config, 'training_params'), f"{config_path}: missing training_params"
            assert hasattr(config, 'discriminators'), f"{config_path}: missing discriminators"
            
            passed += 1
        except Exception as e:
            print(f"  ✗ {config_path}: {e}")
    
    print(f"  ✓ {passed}/{len(configs)} configs loaded successfully")
    return passed == len(configs)


def main():
    """Run all tests"""
    print("="*70)
    print("CompositeMotion MuJoCo - Environment Test Suite")
    print("="*70)
    
    results = []
    
    results.append(("Imports", test_basic_imports()))
    results.append(("MuJoCo Model Loading", test_mujoco_model_loading()))
    results.append(("Reference Motion Loading", test_reference_motion_loading()))
    results.append(("Environment Creation", test_environment_creation()))
    results.append(("Target Environment", test_target_environment()))
    results.append(("Model Creation", test_model_creation()))
    results.append(("Config Files", test_configs()))
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests passed!")
        print("="*70)
        print("\nYou can now run training with:")
        print("  python main.py config/iccgan/jaunty_walk.py --ckpt ./checkpoints")
        print("\nOr preview motions with:")
        print("  python motion_viewer.py config/iccgan/jaunty_walk.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
