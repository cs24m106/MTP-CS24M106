"""
Composite Test Script - MuJoCo Environment & Motion Loading Compatibility
Combines test_env.py and test_motion_loading.py functionality
"""
import sys
import os
import torch
import numpy as np
import mujoco
import importlib.util

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================================
# SECTION 1: Basic Imports Testing
# ============================================================================

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
        from .include.env_iccgan import MujocoEnv, ICCGANHumanoid, ICCGANHumanoidTarget, DiscriminatorConfig
        print("  ✓ env_iccgan imported")
    except ImportError as e:
        print(f"  ✗ env_iccgan import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from .include.models import ACModel, Discriminator
        print("  ✓ models imported")
    except ImportError as e:
        print(f"  ✗ models import failed: {e}")
        return False
    
    try:
        from .include.ref_motion import ReferenceMotion
        print("  ✓ ref_motion imported")
    except ImportError as e:
        print(f"  ✗ ref_motion import failed: {e}")
        return False
    
    try:
        from .include.utils import heading_zup, rotatepoint, quatmultiply
        print("  ✓ utils imported")
    except ImportError as e:
        print(f"  ✗ utils import failed: {e}")
        return False
    
    # Also test env_mujoco if available
    try:
        from .include.env_mujoco import MujocoEnv, DiscriminatorConfig
        print("  ✓ env_mujoco imported")
    except ImportError as e:
        print(f"  ⚠ env_mujoco not available: {e}")
    
    return True


# ============================================================================
# SECTION 2: MuJoCo Model & XML Loading Testing
# ============================================================================

def test_xml_loading():
    """Test that the XML loads correctly"""
    print("=" * 60)
    print("Testing XML Loading")
    print("=" * 60)
    xml_path = "assets/humanoid_posctrl.xml"
    if not os.path.exists(xml_path):
        print(f"ERROR: XML file not found: {xml_path}")
        return False
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"✓ XML loaded successfully")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  DOFs: {model.nv}")
        print(f"  Actuators: {model.nu}")
        
        # Get body names
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) 
                      for i in range(model.nbody)]
        print(f"\n  Body names: {body_names}")
        
        # Expected bodies from reference motion
        expected_bodies = ['world', 'pelvis', 'torso', 'head', 'right_upper_arm', 
                          'right_lower_arm', 'right_hand', 'left_upper_arm', 
                          'left_lower_arm', 'left_hand', 'right_thigh', 'right_shin', 
                          'right_foot', 'left_thigh', 'left_shin', 'left_foot']
        
        if body_names == expected_bodies:
            print("✓ Body names match reference motion format")
        else:
            print("⚠ Body names differ from reference motion:")
            print(f"  Expected: {expected_bodies}")
            print(f"  Got: {body_names}")
        
        return True
        
    except Exception as e:
        print(f"✗ XML loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


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


# ============================================================================
# SECTION 3: Reference Motion Loading Testing
# ============================================================================

def test_motion_loading():
    """Test that reference motion loads correctly"""
    print("\n" + "=" * 60)
    print("Testing Motion Loading")
    print("=" * 60)
    from .include.ref_motion import ReferenceMotion
    
    motion_file = "assets/motions/iccgan/jaunty_walk.json"
    if not os.path.exists(motion_file):
        print(f"⚠ Motion file not found: {motion_file}")
        print("  Skipping motion loading test")
        return True
    
    try:
        ref_motion = ReferenceMotion(
            motion_file=motion_file,
            character_model="assets/humanoid_posctrl.xml",
            device="cpu"
        )
        
        print(f"✓ Motion loaded successfully")
        print(f"  Total motions: {len(ref_motion.motion_length)}")
        print(f"  Total length: {sum(ref_motion.motion_length):.3f}s")
        
        # Test sampling
        motion_ids, motion_times = ref_motion.sample(5)
        print(f"\n  Sampled {len(motion_ids)} motions")
        
        # Test state retrieval
        link_tensor, joint_tensor = ref_motion.state(motion_ids, motion_times)
        print(f"  Link tensor shape: {link_tensor.shape}")
        print(f"  Joint tensor shape: {joint_tensor.shape}")
        
        # Check for NaN
        if torch.isnan(link_tensor).any():
            print("✗ NaN detected in link tensor!")
            return False
        if torch.isnan(joint_tensor).any():
            print("✗ NaN detected in joint tensor!")
            return False
        
        print("✓ No NaN in motion data")
        
        # Check value ranges
        print(f"\n  Link position range: [{link_tensor[..., :3].min():.3f}, {link_tensor[..., :3].max():.3f}]")
        print(f"  Joint position range: [{joint_tensor[..., 0].min():.3f}, {joint_tensor[..., 0].max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Motion loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reference_motion_loading():
    """Test loading reference motion data"""
    print("\nTesting reference motion loading...")
    from .include.ref_motion import ReferenceMotion
    
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


# ============================================================================
# SECTION 4: State Initialization Testing
# ============================================================================

def test_state_initialization():
    """Test state initialization from reference motion"""
    print("\n" + "=" * 60)
    print("Testing State Initialization")
    print("=" * 60)
    try:
        from .include.env_mujoco import MujocoEnv, DiscriminatorConfig
        
        motion_file = "assets/motions/iccgan/jaunty_walk.json"
        if not os.path.exists(motion_file):
            print(f"⚠ Motion file not found, skipping")
            return True
        
        env = MujocoEnv(
            n_envs=1,
            fps=30,
            run_speed=120,
            episode_length=300,
            control_mode="position",
            character_model="assets/humanoid.xml",
            render_mode=None,
            verbose=True
        )
        
        # Test init_state
        env_ids = torch.tensor([0])
        ref_link_tensor, ref_joint_tensor = env.init_state(env_ids)
         
        print(f"✓ init_state works")
        print(f"  Link tensor shape: {ref_link_tensor.shape}")
        print(f"  Joint tensor shape: {ref_joint_tensor.shape}")
        
        # Check for NaN
        if torch.isnan(ref_link_tensor).any():
            print("✗ NaN in link tensor!")
            return False
        if torch.isnan(ref_joint_tensor).any():
            print("✗ NaN in joint tensor!")
            return False
        
        print("✓ No NaN in initial state")
        
        # Check root height
        root_z = ref_link_tensor[0, 0, 2].item()
        print(f"  Initial root height: {root_z:.3f}")
        
        if root_z < 0.5:
            print("⚠ Root height seems low, may cause ground penetration")
        
        return True
        
    except Exception as e:
        print(f"✗ State initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# SECTION 5: Environment Creation & Step Testing
# ============================================================================

def test_environment_step():
    """Test environment step"""
    print("\n" + "=" * 60)
    print("Testing Environment Step")
    print("=" * 60)
    try:
        from .include.env_mujoco import MujocoEnv
        
        motion_file = "assets/motions/iccgan/jaunty_walk.json"
        if not os.path.exists(motion_file):
            print(f"⚠ Motion file not found, skipping")
            return True
        
        env = MujocoEnv(
            n_envs=2,
            fps=30,
            run_speed=120,
            episode_length=300,
            control_mode="position",
            character_model="assets/humanoid.xml",
            render_mode=None,
            verbose=True
        )
        
        # Reset
        env.reset()
        obs, info = env.reset_done()
        
        print(f"✓ Environment reset")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        
        if torch.isnan(obs).any():
            print("✗ NaN in observation!")
            return False
        
        print("✓ No NaN in observation")
        
        # Take a few steps
        for step in range(5):
            actions = torch.zeros((2, env.act_dim), device=env.device)
            obs, reward, terminated, truncated, info = env.step(actions.cpu().numpy())
            
            if torch.isnan(torch.from_numpy(obs)).any():
                print(f"✗ NaN in observation at step {step}!")
                return False
            
            print(f"  Step {step}: reward={reward.mean():.4f}, terminated={terminated.any()}")
        
        print("✓ Environment step works")
        return True
        
    except Exception as e:
        print(f"✗ Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_creation():
    """Test creating the environment"""
    print("\nTesting environment creation...")
    from .include.env_iccgan import ICCGANHumanoid, DiscriminatorConfig
    
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
        
        env = ICCGANHumanoid(
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
    from .include.env_iccgan import ICCGANHumanoidTarget, DiscriminatorConfig
    
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
        
        env = ICCGANHumanoidTarget(
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


# ============================================================================
# SECTION 6: Model Creation Testing
# ============================================================================

def test_model_creation():
    """Test creating the AC model"""
    print("\nTesting model creation...")
    from .include.models import ACModel, Discriminator
    
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


# ============================================================================
# SECTION 7: Configuration Files Testing
# ============================================================================

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


# ============================================================================
# MAIN - Run All Tests
# ============================================================================

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("CompositeMotion - Complete Test Suite (Motion + Environment)")
    print("=" * 70)
    results = []
    
    # Section 1: Imports
    results.append(("Basic Imports", test_basic_imports()))
    
    # Section 2: XML/Model Loading
    results.append(("XML Loading", test_xml_loading()))
    results.append(("MuJoCo Model Loading", test_mujoco_model_loading()))
    
    # Section 3: Motion Loading
    results.append(("Motion Loading", test_motion_loading()))
    results.append(("Reference Motion Loading", test_reference_motion_loading()))
    
    # Section 4: State Initialization
    results.append(("State Initialization", test_state_initialization()))
    
    # Section 5: Environment Testing
    results.append(("Environment Step", test_environment_step()))
    results.append(("Environment Creation", test_environment_creation()))
    results.append(("Target Environment", test_target_environment()))
    
    # Section 6: Model Creation
    results.append(("Model Creation", test_model_creation()))
    
    # Section 7: Configuration Files
    results.append(("Config Files", test_configs()))
    
    # Print Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests passed!")
        print("=" * 70)
        print("\nYou can now train with:")
        print("  python main.py config/iccgan/jaunty_walk.py --ckpt checkpoints/jaunty_walk")
        print("\nOr preview motions with:")
        print("  python motion_viewer.py config/iccgan/jaunty_walk.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())