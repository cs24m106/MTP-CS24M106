from pytorch3d import transforms # contains useful 3D transform utils
from pathlib import Path
import os, re, glob, smplx, torch, traceback
import torch.nn as nn
import torch.nn.functional as F
from .ddpm_utils import get_snr_weight

#DATA_DIR = Path.cwd() / "datasets"
DATA_DIR = Path("D:/STUDIES/MTech/#MTP/codes/vimo") / "datasets"
SMPL_DIR = DATA_DIR  / "SMPL"
SMPL_MODEL = None
SMPL_NORM_MAXABS = 4.0 # default 3, but very few clips have 3.5 ranges

# Define SMPL skeleton connections (24 joints)
smpl_skeleton = [
	(0, 1), (0, 2), (0, 3),  	# Pelvis(Root) to Hips, Spine
	(1, 4), (2, 5),          	# Hips to Knees
	(4, 7), (5, 8),          	# Knees to Ankles
	(7, 10), (8, 11),        	# Ankles to Foots
	(3, 6), (6, 9),          	# Spine to head
	(9, 12), (9, 13), (9, 14),  # Spine to Neck, Collars
	(12, 15),  					# Neck to Head
	(13, 16), (14, 17),			# Collars to Shoulders
	(16, 18), (17, 19),      	# Shoulders to Elbows
	(18, 20), (19, 21),      	# Elbows to Wrists
	(20, 22), (21, 23)       	# Wrists to Hands
]

# Foot joint ids for SMPL (refer bg-studies docs)
ANKLE_IDX = (7, 8)   # L_Ankle, R_Ankle
FOOT_IDX  = (10,11)  # L_Foot,  R_Foot
CONTACT_IDX = ANKLE_IDX + FOOT_IDX  # (L_Ankle, R_Ankle, L_Foot, R_Foot) use same order used in preprocessing

# ------------------ Load SMPL model ------------------
def try_load_smpl_model(verbose=False):
    global SMPL_MODEL
    smpl_files = glob.glob(f"{SMPL_DIR}/v*/smpl/*.pkl")

    def version_tuple_from_path(f, iter=2):
        # infer version from parent folder name like "v1.0.0" or "v1.1.0"
        p = Path(f)
        for i in range(iter):
            p = p.parent
        parent = p.name
        nums = re.findall(r"\d+", parent)
        if not nums:
            return (0,)
        return tuple(int(x) for x in nums)

    # Prefer a neutral model if present (case-insensitive); otherwise fall back to first candidate
    neutral_types = [file for file in smpl_files if "neutral" in os.path.basename(file).lower()]
    if len(neutral_types) > 0:
        chosen = max(neutral_types, key=version_tuple_from_path)
    else:
        # fallback: pick highest-version candidate regardless of gender
        chosen = max(smpl_files, key=version_tuple_from_path)
    chosen_pkl = Path(chosen)
    model_folder = chosen_pkl.parent.parent
    if verbose: print(f"Chosen SMPL model.pkl: {chosen_pkl.name} | folder: {model_folder.name}")

    # Create SMPL model using the neutral gender option
    try:
        SMPL_MODEL = smplx.create(model_path=model_folder, model_type='smpl', gender='neutral', use_pca=False)
        if verbose: print(f"SMPL-model: {SMPL_MODEL}, created successfully")
    except Exception as e:
        # If smplx.create fails, re-raise with helpful debug info
        raise RuntimeError(f"smplx.create failed for model_folder={model_folder} with error: {e}")

# ------------------ FK using SMPL ------------------
def fk_for_worldpos(rotmats, root_pos, verbose=False):
    """
    Forward Kinematics (FK): Convert joint rotation matrices and root positions to world joint positions.
    Parameters:
        - rotmats: (B,S,24,3,3) rotation matrices for each joint
        - root_pos: (B,S,3) root joint positions
    returns: j3d_world: (B.S,24,3) world joint positions
    """
    global SMPL_MODEL
    if SMPL_MODEL is None: try_load_smpl_model()
    if root_pos.dim() > 2:
        B, S, _ = root_pos.shape
    else:
        S, _ = root_pos.shape; B = 1
    root3d = root_pos.reshape(B*S, 3)          # (B*S,3)
    axis3d = transforms.matrix_to_axis_angle(rotmats.reshape(B*S, 24, 3, 3))  # (B*S,24,3)
    
    max_abs = root3d.abs().max().item()
    is_normalized = max_abs < SMPL_NORM_MAXABS # Heuristic: scale-normalized if max_abs < 3
    if not is_normalized: # max try to avoid by increasing norm range within acceptable margins
        print(f"\033[1;33mWARNING: Given input is not scale normalized (max-abs of root-pos:{max_abs} must be less than {SMPL_NORM_MAXABS}) --> apply norm\033[0m")
        # Center around mean and scale to reasonable range
        if verbose: 
            traceback.print_stack(file=sys.stdout)
            mean_root = root3d.mean(dim=0, keepdim=True)
            root_centered = root3d - mean_root
            # Scale to standard range (~2m max extent)
            scale = root_centered.abs().max() / 2.0
            if scale > 0:
                root3d = root_centered / scale

    # compute joint world positions with SMPL FK
    j3d_world = None
    SMPL_MODEL.to(root3d.device)
    try:
        # smplx expects global_orient (B,3), body_pose (B,69), transl (B,3)
        # call in batches (smpl can process batch) To feed these into the SMPL model, we have to split the data into:
        # global_orient — rotation of the root joint only (pelvis:joint-id=0)
        # body_pose — rotations of all other 23 joints
        # transl — translation vector for the whole body
        with torch.no_grad():
            go = axis3d[:, 0, :]                        # (_,1,3)=>(_,3)
            body = axis3d[:, 1:, :].reshape(-1, 23*3)   # (_,23,3)=>(_,69)
            tr = root3d                                 # (_,3)
            final_output = []
            for i in range(S): # perform batch-wise to avoid memory bottle neck
                st = i*B; ed = st+B
                output = SMPL_MODEL(global_orient=go[st:ed], body_pose=body[st:ed], transl=tr[st:ed])
                # output.joints shape (S, J, 3) where J >= 24. We'll take the first 24 SMPL joints.
                final_output.append(output.joints[:, :24, :]) # (_,24,3)
            j3d_world = torch.cat(final_output)     
            
    except Exception as e:
        raise RuntimeError(f"SMPL FK failed for input (axis3d:{axis3d.shape}, root3d:{root3d.shape}) -> Error: {e}")

    return j3d_world.reshape(B, S, 24, 3)  # (B,S,24,3)

# --------------------- Helper Utility ---------------------
def get_joint_positions(m3d):
    """
    Parameters:
        m3d: (B,S,D) where D = 151 {J3D=24*6 + Foot=4 + Root=3}
    Returns:
        Batchwise reduction using 'mean' as standard, i.e. scalar value
    """
    B, S, _ = m3d.shape
    root = m3d[:,:, -3:]     # (B,S,3)
    #contacts_gt = m3d[:,:, -7:-3]     # (B,S,4)
    j6d = m3d[:, :, :-7].reshape(B, S, 24, -1)     # (B,S,24,6)
    rmats = transforms.rotation_6d_to_matrix(j6d)     # (B,S,24,3,3)
    pos3d = fk_for_worldpos(rmats, root)       # (B,S,24,3)
    return pos3d

# --------------------- Metric Utility ---------------------
def comp_statistics_across_frames(rotmats):
    """
    Input: rotmats (B,S,24,3,3) or (S,24,3,3) rotation matrices for each joint
    Returns average (motion variance and standard deviation in axis-angle rep) per batch sample (B,)
    """
    # Convert to axis-angle
    if rotmats.dim() == 4: rotmats.unsqueeze(0)
    assert rotmats.dim() == 5, "Rotation matrix dim must be (B,S,24,3,3) or (S,24,3,3)!"
    axisang = transforms.matrix_to_axis_angle(rotmats)  # (B, S, J, 3)
    # Compute var & std over time axis (S) for each joint and axis
    var_per_joint = axisang.var(dim=1)
    std_per_joint = axisang.std(dim=1)  # (B, J, 3)
    # Average over batch and joints and axes
    avg_var = var_per_joint.mean(dim=tuple(range(1, var_per_joint.dim())))
    avg_std = std_per_joint.mean(dim=tuple(range(1, std_per_joint.dim())))
    return avg_var, avg_std

# --------------------- Optional Utility ---------------------
def comp_var_diff_loss(m_pred, m_gt): # --> increases std but rotations are rigid
    L_diff_per_frame = mse(m_pred, m_gt).mean(dim=-1)  # (B,S,D) => (B,S)
    # Compute temporal variance of predictions
    var_pred = m_pred.var(dim=1, keepdim=True).mean(dim=-1)
    var_gt = m_gt.var(dim=1, keepdim=True).mean(dim=-1)  # var on S => (B, 1, D), mean on D => (B, 1)
    # Penality based on variance: Low = (static, incr w); High = (dynamic, decr w)
    var_penalty = torch.abs(torch.log(var_pred) - torch.log(var_gt))  # logarithmic penality L1 norm (B, 1)
    scaled_penalty = 1.0 + 0.1 * torch.clamp(var_penalty, max=1.0) # step bump incr at a time
    # Apply variance penalty broadcasts to (B, S), mean on S => (B,)
    L_simple_adjusted_per_sample = (L_diff_per_frame * scaled_penalty).mean(dim=1)
    return L_simple_adjusted_per_sample

# ======================== Main Loss Fn ========================
def compute_losses(m_pred, m_gt, t, lambda_joints=1.0, lambda_vel=1.0, lambda_foot=1.0, lambda_temporal=0, lambda_diversity=0, 
                   snr_type="cp", snr_cap=1.0, snr_norm=False, snr_ratio=0.05, var_ratio=0.1):
    """
    Parameters:
        - m_pred, m_gt: (B,S,D) where D = 151 {J3D=24*6 + Foot=4 + Root=3}
        - lambda values for L_joints, L_vel, L_foot resp
        - t: timestep or diffusion-step to calc snr weights
    Returns: 
        - Total_Loss(L) = L_simple + λ1*L_joints + λ2*L_vel + λ3*L_foot
          Batchwise reduction using 'mean' as standard, i.e. scalar value
        - Dict containing individual loss (total, simple, joints, vel, foot) per sample, i.e. (B,)
    Features:
        - std across frames for monitoring training, must not reach near 0
        - SNR (Signal-to-Noise Ratio) Timestep-dependent weighting
    """
    assert m_pred.shape == m_gt.shape, "Predicted and Ground Truth motion shapes must match!"
    B, S, _ = m_gt.shape
    R = t.dim() # reduce dim till, in order to match weight dims
    mse = nn.MSELoss(reduction='none')  # ← Changed to 'none' as we manually add SNR batchwise
    weights = get_snr_weight(t, type=snr_type, w_cap=snr_cap, batch_norm=snr_norm, ratio=snr_ratio)  # (B,)||(B,S)
    s_minus_1_weights = weights[:,1:,...] if R>1 else weights
    s_minus_2_weights = weights[:,2:,...] if R>1 else weights

    root_pred = m_pred[:,:, -3:]
    root_gt = m_gt[:,:, -3:]     # (B,S,3)
    contacts_pred = m_pred[:,:, -7:-3]
    #contacts_gt = m_gt[:,:, -7:-3]     # (B,S,4)
    j6d_pred = m_pred[:, :, :-7].reshape(B, S, 24, -1)
    j6d_gt = m_gt[:, :, :-7].reshape(B, S, 24, -1)     # (B,S,24,6)

    # Convert 6D rotation representation to 3x3 rotation matrices using Gram-Schmidt algorithm
    rmats_pred = transforms.rotation_6d_to_matrix(j6d_pred)  
    rmats_gt = transforms.rotation_6d_to_matrix(j6d_gt)     # (B,S,24,3,3)
    pos3d_pred = fk_for_worldpos(rmats_pred, root_pred)
    pos3d_gt = fk_for_worldpos(rmats_gt, root_gt)       # (B,S,24,3)

    # Diffusion Loss: MSE on motions instead of epsilon/noise
    L_simple_per_sample = mse(m_pred, m_gt).mean(dim=tuple(range(R, m_pred.dim())))  # (B,S,D) => (B,)||(B,S)
    #L_simple_per_sample = comp_var_diff_loss(m_pred, m_gt)  # (B,S,D) variance weighted => (B,)
    L_simple = (L_simple_per_sample * weights).mean()  # weighted batch mean

    # Joint Loss: MSE on joint world positions
    L_joints_per_sample = mse(pos3d_pred, pos3d_gt).mean(dim=tuple(range(R, pos3d_pred.dim())))  # (B,S,24,3) => (B,)||(B,S)
    L_joints = (L_joints_per_sample * weights).mean() # weighted batch mean
    
    # Velocity (Kindoff Angular) Loss: difference between consecutive frames of rotation matrices
    Avel_pred = rmats_pred[:,1:,...] - rmats_pred[:,:-1,...]
    Avel_gt   = rmats_gt[:,1:,...] - rmats_gt[:,:-1,...]
    L_vel_per_sample = mse(Avel_pred, Avel_gt).mean(dim=tuple(range(R, Avel_pred.dim())))  # (B,S-1,24,3,3) => (B,)||(B,S) loss for resp batch
    L_vel = (L_vel_per_sample * s_minus_1_weights).mean() # weighted batch mean for S-1 frames

    # Foot Contact Loss: BCE on foot contact predictions
    vel_pred = pos3d_pred[:,1:,CONTACT_IDX,:] - pos3d_pred[:,:-1,CONTACT_IDX,:]
    # f_bar: (B, S, 4) -> (B, S-1, 4) -> (B, S-1, 4, 1) to multiply with (B, S-1, 4, 3)
    foot_pred = vel_pred*contacts_pred[:, :-1].unsqueeze(-1)
    # L2 norm over all last dims other than (0:B,1:S) => shape: (B, S-1)
    foot_norm = torch.linalg.norm(foot_pred, ord=2, dim=tuple(range(2, foot_pred.dim())))
    foot_norm_per_sample = foot_norm**2
    if foot_norm_per_sample.dim() > R: # apply mean square over S frames => (B,) if t.shape = (B,)
        foot_norm_per_sample = torch.mean(foot_norm_per_sample, dim=tuple(range(R, foot_norm_per_sample.dim())))
    L_foot = (foot_norm_per_sample * s_minus_1_weights).mean() # reduction batch via weighted mean

    # Temporal Consistency Loss - Compute temporal derivatives (frame-to-frame changes)
    Racc_pred = Avel_pred[:, 1:] - Avel_pred[:, :-1] # Predicted motion acceleration (2nd derivative)
    Racc_gt = Avel_gt[:, 1:] - Avel_gt[:, :-1] # (B, S-2, 24, 3, 3)
    Racc_mse = mse(Racc_pred, Racc_gt).mean(dim=tuple(range(R, Racc_pred.dim())))  # to penalize jerky motion (B, S-2, 24, 3, 3) => (B,)
    tmp_smooth = (Racc_pred ** 2).mean(dim=tuple(range(R, Racc_pred.dim())))  # Temporal smoothness - penalize high-frequency noise (B,)
    L_temporal = ((Racc_mse + 0.01 * tmp_smooth) * s_minus_2_weights).mean()  # weighted batch mean for S-2 frames for S-1 frames

    # Diversity loss based on std on rmats in axis angle
    var_pred, std_pred = comp_statistics_across_frames(rmats_pred)
    var_gt, std_gt = comp_statistics_across_frames(rmats_gt) # (B, S, 24, 3) => (B,1)
    # RelU -> hinge loss why? penalize only LOW var. if abs or **2 used => Penalty when: score != target (two-sided)
    div_weights = weights.mean(dim=tuple(range(var_pred.dim(), weights.dim()))) if R > var_pred.dim() else weights # reduce weights as L_div works differently
    L_diversity = (F.relu(torch.log(var_gt) - torch.log(var_pred)) * div_weights).mean() # convert to log scale as value range is very low

    total = L_simple + lambda_joints*L_joints + lambda_vel*L_vel + lambda_foot*L_foot + lambda_temporal*L_temporal + lambda_diversity*L_diversity
    return total, {'pred_std': std_pred.mean().item(), 'gt_std': std_gt.mean().item(),
                   'simple': L_simple.item(), 'joints': L_joints.item(), 'vel': L_vel.item(), 'foot': L_foot.item(), 
                   'temporal': L_temporal.item(), 'diversity': L_diversity.item()}


# --------------------- Optimizer Utils ---------------------
# To Generate a concise string representation of the optimizer with all parameters.
def compact_optim_str(optimizer, skip_non_numeric=True):
    if optimizer is None:
        raise RuntimeError("No optimizer passed!")

    def formatted(value):
        if isinstance(value, float):
            if value == 0:
                fmt_val = '0'
            elif value < 0.001:
                fmt_val = f'{value:.0e}'
            else:
                fmt_val = f'{value:.4g}'
        elif isinstance(value, tuple):
            fmt_val = f'({",".join(formatted(v) for v in value)})'
        elif isinstance(value, bool):
            fmt_val = str(value)[0]  # T or F
        else:
            fmt_val = str(value).lower()
            splits = fmt_val.split('_')
            if len(splits)>1:
                fmt_val = ''.join([s[0] for s in splits])            
        return fmt_val
        
    # Get optimizer name
    opt_name = optimizer.__class__.__name__
    
    # Get all parameter groups (usually just one)
    param_group = optimizer.param_groups[0]
    
    # Define short names for common parameters
    short_names = {
        'lr': 'lr',
        'learning_rate': 'lr',  # skip caz we modify to converge faster
        'weight_decay': 'wd',
        'momentum': 'mom',
        'betas': 'b',
        'eps': 'eps',           # skip (epsilon) preventing division by zero param
        'alpha': 'alpha',
        'rho': 'rho',
        'dampening': 'damp',
        'nesterov': 'nest',
        'amsgrad': 'ams',
        'maximize': 'max',
        'foreach': 'fe',
        'capturable': 'cap',
        'differentiable': 'diff',
        'fused': 'fused',
    }
    # skip certain params that you want it be modifyable
    skip_names = ['lr', 'eps']
    
    # Build parameter string
    param_parts = []
    for key, value in param_group.items():
        if key == 'params':  # Skip the params list
            continue
        if value == None:
            continue
        if skip_non_numeric and (not isinstance(value, (int, float, complex)) or isinstance(value, bool)):
            continue
        
        # Use short name if available
        short_key = short_names.get(key, formatted(key))
        if short_key in skip_names:
            continue
        # Format the value appropriately        
        param_parts.append(f'{short_key}={formatted(value)}')
    
    return f"{opt_name}({','.join(param_parts)})"