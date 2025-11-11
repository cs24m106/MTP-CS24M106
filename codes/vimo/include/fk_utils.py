from pytorch3d import transforms # contains useful 3D transform utils
from pathlib import Path
import os, re, glob, smplx, torch

DATA_DIR = Path.cwd() / "datasets"
SMPL_DIR = DATA_DIR  / "SMPL"
SMPL_MODEL = None

# Foot joint ids for SMPL (refer bg-studies docs)
ANKLE_IDX = (7, 8)   # L_Ankle, R_Ankle
FOOT_IDX  = (10,11)  # L_Foot,  R_Foot
CONTACT_IDX = ANKLE_IDX + FOOT_IDX  # (L_Ankle, R_Ankle, L_Foot, R_Foot) use same order used in preprocessing

# ------------------ Load SMPL model ------------------
def try_load_smpl_model():
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
    print(f"Chosen SMPL model.pkl: {chosen_pkl.name} | folder: {model_folder.name}")

    # Create SMPL model using the neutral gender option
    try:
        SMPL_MODEL = smplx.create(model_path=model_folder, model_type='smpl', gender='neutral', use_pca=False)
        print(f"SMPL-model: {SMPL_MODEL}, created successfully")
    except Exception as e:
        # If smplx.create fails, re-raise with helpful debug info
        raise RuntimeError(f"smplx.create failed for model_folder={model_folder} with error: {e}")

# ------------------ FK using SMPL ------------------
def fk_for_worldpos(rotmats, root_pos):
    """
    Forward Kinematics (FK): Convert joint rotation matrices and root positions to world joint positions.
    Parameters:
        - rotmats: (B,S,24,3,3) rotation matrices for each joint
        - root_pos: (B,S,3) root joint positions
    returns: j3d_world: (B.S,24,3) world joint positions
    """
    global SMPL_MODEL
    if SMPL_MODEL is None: try_load_smpl_model()
    B, S, _  = root_pos.shape
    root3d = root_pos.reshape(B*S, 3)          # (B*S,3)
    axis3d = transforms.matrix_to_axis_angle(rotmats.reshape(B*S, 24, 3, 3))  # (B*S,24,3)
    
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
        raise RuntimeError(f"\nSMPL FK failed for input (axis3d:{axis3d.shape}, root3d:{root3d.shape}) -> Error: {e}")

    return j3d_world.reshape(B, S, 24, 3)  # (B,S,24,3)

# ------------------ Helper Utility ------------------
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