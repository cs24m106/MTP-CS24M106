import os
import json
import torch
import smplx
import numpy as np
from pytorch3d import transforms
from scipy.spatial.transform import Rotation as R

# SMPL Kinematics & Rotation Utilities
class SMPLHandler:
	'''Helper class to handle 6D Rotation conversions (used in ViMo) and loads the SMPL body model using smplx.'''
	
	def __init__(self, model_path, gender='neutral'):
		"""Initializes the SMPL layer for Forward Kinematics."""
		self.device = torch.device("cpu")
		
		# Load standard SMPL model
		try:
			self.smpl_layer = smplx.create(
				model_path=str(model_path), 
				model_type='smpl',
				gender=gender, 
				ext='pkl'
			).to(self.device)
			print("✅ SMPL Model Loaded.")
		except Exception as e:
			print(f"⚠️ Could not load SMPL model: {e}")
			self.smpl_layer = None

	def run_fk(self, pose_aa, trans, beta=None):
		"""
		Runs Forward Kinematics to get 3D Joint positions.
		Input: 
			pose_aa: (N, 72) Axis-Angle
			trans: (N, 3) Translation
		Returns:
			joints_3d: (N, 24, 3)
		"""
		if self.smpl_layer is None: return None
		
		N = pose_aa.shape[0]
		# Convert to torch
		body_pose = torch.tensor(pose_aa[:, 3:], dtype=torch.float32) # Exclude root (first 3)
		global_orient = torch.tensor(pose_aa[:, :3], dtype=torch.float32)
		transl = torch.tensor(trans, dtype=torch.float32)
		
		output = self.smpl_layer(
			body_pose=body_pose,
			global_orient=global_orient,
			transl=transl,
			return_verts=False
		)
		return output.joints.detach().numpy() # (N, 45, 3) - SMPL returns 45 joints usually


class SMPL2GymJson:
	"""
	Converts SMPL motion (AIST++ PKL / ViMo NPZ) → CompositeMotion DeepMimic JSON.

	Coordinate systems
	──────────────────
	SMPL  : +X = left,    +Y = up,   +Z = forward   (right-handed, Y-up)
	MuJoCo: +X = forward, +Y = left, +Z = up         (right-handed, Z-up)

	The correct change-of-basis G is a 120° rotation around [1,1,1]:
		G = [[0,0,1],[1,0,0],[0,1,0]]
		smpl (x,y,z) → mujoco (z,x,y)

	NOTE: The previous R_x(90°) was WRONG — it only mapped Y→Z correctly
	but swapped the horizontal axes (left↔forward), causing joints near the
	ground to be pulled toward the wrong axis.

	Pipeline: similarity transform + per-joint bind rotations + height scaling.
	"""
	FLOOR_LEFT_JOINTS = [7, 10] 	# L_Ankle, L_Foot 
	FLOOR_RIGHT_JOINTS = [8, 11]	# R_Ankle, R_Foot
	FLOOR_JOINT_WEIGHTS = [1./4, 3./4]	# weighted avg, priortizing foot contact point more

	SMPL_PARENTS = [
		-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21
	]

	def __init__(self, smpl_handler=None):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.smpl_handler = smpl_handler

		# ── Correct G: SMPL (X=left,Y=up,Z=fwd) → MuJoCo (X=fwd,Y=left,Z=up)
		# Cyclic permutation: (x,y,z) → (z,x,y)
		self.global_rot = torch.tensor(
			[[0., 0., 1.],
			[1., 0., 0.],
			[0., 1., 0.]], device=self.device, dtype=torch.float32
		)

		# SMPL joints: 0 Pelvis  1 L_Hip  2 R_Hip  3 Spine1  4 L_Knee  5 R_Knee
		#  6 Spine2  7 L_Ankle  8 R_Ankle  9 Spine3  10 L_Foot  11 R_Foot
		# 12 Neck  13 L_Collar  14 R_Collar  15 Head  16 L_Shoulder  17 R_Shoulder
		# 18 L_Elbow  19 R_Elbow  20 L_Wrist  21 R_Wrist

		self.body_chain = {
			"torso":           [3, 6, 9], # spine1, spine2, spine3
			"head":            [12, 15],
			"right_upper_arm": [14, 17],
			"left_upper_arm":  [13, 16],
			"right_lower_arm": [19],
			"left_lower_arm":  [18],
			"right_hand":      [21],
			"left_hand":       [20],
			"right_thigh":     [2],
			"left_thigh":      [1],
			"right_shin":      [5],
			"left_shin":       [4],
			"right_foot":      [8], # mujuco.r_foot -> smpl.r_ankle
			"left_foot":       [7], # mujuco.r_foot -> smpl.r_ankle
		}

		self._mujoco_parent = {
			"torso": "pelvis", "head": "torso",
			"right_upper_arm": "torso", "right_lower_arm": "right_upper_arm",
			"right_hand": "right_lower_arm",
			"left_upper_arm": "torso", "left_lower_arm": "left_upper_arm",
			"left_hand": "left_lower_arm",
			"right_thigh": "pelvis", "right_shin": "right_thigh",
			"right_foot": "right_shin",
			"left_thigh": "pelvis", "left_shin": "left_thigh",
			"left_foot": "left_shin",
		}

		# (smpl_tip_joint, smpl_child_joint, mujoco_bone_dir_from_xml)
		# For leaf bodies (feet/hands) we use the geom direction as reference
		self._bone_info = {
			"torso":           (9,  12, [0, 0, 0.223894]),
			"head":            (15, None, None),
			"right_upper_arm": (17, 19, [0, 0, -0.274788]),
			"right_lower_arm": (19, 21, [0, 0, -0.258947]),
			"right_hand":      (21, None, None),
			"left_upper_arm":  (16, 18, [0, 0, -0.274788]),
			"left_lower_arm":  (18, 20, [0, 0, -0.258947]),
			"left_hand":       (20, None, None),
			"right_thigh":     (2,   5, [0, 0, -0.421546]),
			"right_shin":      (5,   8, [0, 0, -0.409870]),
			"right_foot":      (8,  11, [0.045, 0, -0.0225]),  # foot geom dir
			"left_thigh":      (1,   4, [0, 0, -0.421546]),
			"left_shin":       (4,   7, [0, 0, -0.409870]),
			"left_foot":       (7,  10, [0.045, 0, -0.0225]),  # foot geom dir
		}

		self.bind_rots = self._compute_bind_rotations()
		self.height_scale = self._compute_height_scale()

	def get_avg_floor_height(self, j3d):
		"""
		SMPL Y-up: use ankle + foot joint Y, weighted toward foot (contact).
		offset: +ve for strict, -ve for loose
		"""
		# Per-frame weighted mean over 2 joints (axis 1); then min over time.
		avg_left = np.average(j3d[:, self.FLOOR_LEFT_JOINTS, 1], axis=1, weights=self.FLOOR_JOINT_WEIGHTS).min()
		avg_right = np.average(j3d[:, self.FLOOR_RIGHT_JOINTS, 1], axis=1, weights=self.FLOOR_JOINT_WEIGHTS).min()
		floor_y = float(min(avg_left, avg_right))
		#floor_y = np.ceil(floor_y * 100) / 100 # strict offset
		#floor_y = np.floor(floor_y * 100) / 100 # loose offset
		return floor_y


	# ═══════════════════════════════════════════════════════════════════════
	# Bind rotation computation
	# ═══════════════════════════════════════════════════════════════════════

	def _compute_bind_rotations(self):
		"""Per-body R_bind: maps MuJoCo rest bone dir → SMPL rest bone dir (Z-up)."""
		G = self.global_rot.cpu().numpy()
		smpl_rest = self._get_smpl_rest_joints()

		bind = {}
		for body, info in self._bone_info.items():
			tip_j, child_j, mj_trans = info
			if child_j is None or mj_trans is None:
				bind[body] = np.eye(3)
				continue
			smpl_bone = smpl_rest[child_j] - smpl_rest[tip_j]
			norm = np.linalg.norm(smpl_bone)
			if norm < 1e-6:
				bind[body] = np.eye(3)
				continue
			d_smpl_zup = G @ (smpl_bone / norm)
			d_mujoco = np.array(mj_trans, dtype=np.float64)
			d_mujoco = d_mujoco / np.linalg.norm(d_mujoco)
			bind[body] = self._rotation_between(d_mujoco, d_smpl_zup)

		bind["pelvis"] = np.eye(3)
		return {k: torch.tensor(v, dtype=torch.float32, device=self.device)
				for k, v in bind.items()}

	def _get_smpl_rest_joints(self):
		"""SMPL zero-pose joint positions (24, 3) in Y-up."""
		if self.smpl_handler is not None and hasattr(self.smpl_handler, 'run_fk'):
			try:
				rest = self.smpl_handler.run_fk(
					np.zeros((1, 72), dtype=np.float32),
					np.zeros((1, 3), dtype=np.float32))
				if rest is not None:
					#print("  Bind rotations: using SMPL FK rest-pose joints")
					return rest[0, :24, :]
			except Exception:
				pass
		print("  Bind rotations: using approximate SMPL neutral joints")
		return np.array([
			[ 0.0000,  0.0000,  0.0000],  #  0 Pelvis
			[ 0.0621, -0.0886, -0.0170],  #  1 L_Hip
			[-0.0621, -0.0886, -0.0170],  #  2 R_Hip
			[ 0.0000,  0.0672,  0.0259],  #  3 Spine1
			[ 0.0886, -0.4943,  0.0048],  #  4 L_Knee
			[-0.0886, -0.4943,  0.0048],  #  5 R_Knee
			[ 0.0000,  0.2124, -0.0143],  #  6 Spine2
			[ 0.0700, -0.8763, -0.0105],  #  7 L_Ankle
			[-0.0700, -0.8763, -0.0105],  #  8 R_Ankle
			[ 0.0000,  0.3490,  0.0017],  #  9 Spine3
			[ 0.0926, -0.9912,  0.1189],  # 10 L_Foot
			[-0.0926, -0.9912,  0.1189],  # 11 R_Foot
			[ 0.0000,  0.5951, -0.0318],  # 12 Neck
			[ 0.0529,  0.5631, -0.0043],  # 13 L_Collar
			[-0.0529,  0.5631, -0.0043],  # 14 R_Collar
			[ 0.0000,  0.6633, -0.0127],  # 15 Head
			[ 0.1722,  0.5340, -0.0371],  # 16 L_Shoulder
			[-0.1722,  0.5340, -0.0371],  # 17 R_Shoulder
			[ 0.4418,  0.5376, -0.0315],  # 18 L_Elbow
			[-0.4418,  0.5376, -0.0315],  # 19 R_Elbow
			[ 0.6720,  0.5403, -0.0131],  # 20 L_Wrist
			[-0.6720,  0.5403, -0.0131],  # 21 R_Wrist
			[ 0.7300,  0.5400,  0.0000],  # 22 L_Hand
			[-0.7300,  0.5400,  0.0000],  # 23 R_Hand
		], dtype=np.float64)

	@staticmethod
	def _rotation_between(a, b):
		"""Rotation matrix R such that R @ a = b (unit vectors)."""
		a = a / np.linalg.norm(a)
		b = b / np.linalg.norm(b)
		v = np.cross(a, b)
		c = float(np.dot(a, b))
		if c > 1.0 - 1e-8:
			return np.eye(3)
		if c < -1.0 + 1e-8:
			perp = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
			perp = perp - np.dot(perp, a) * a
			perp = perp / np.linalg.norm(perp)
			return 2 * np.outer(perp, perp) - np.eye(3)
		vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
		return np.eye(3) + vx + vx @ vx / (1 + c)

	# ═══════════════════════════════════════════════════════════════════════
	# Height scaling — compensate for skeleton proportion differences
	# ═══════════════════════════════════════════════════════════════════════

	def _compute_height_scale(self):
		"""Ratio of MuJoCo leg length to SMPL leg length (pelvis→foot sole)."""
		smpl_rest = self._get_smpl_rest_joints()
		smpl_foot_y = min(smpl_rest[10, 1], smpl_rest[11, 1])
		smpl_pelvis_to_floor = smpl_rest[0, 1] - smpl_foot_y

		mj_leg = 0.421546 + 0.409870  # thigh + shin from XML
		mj_foot_sole = 0.0225 + 0.0275  # foot geom center z + half-height
		mj_pelvis_to_floor = mj_leg + mj_foot_sole

		scale = mj_pelvis_to_floor / max(smpl_pelvis_to_floor, 0.01)
		return float(scale)

	def print_bone_comparison(self):
		"""Print side-by-side bone lengths for SMPL vs MuJoCo."""
		smpl_rest = self._get_smpl_rest_joints()
		G = self.global_rot.cpu().numpy()
		pairs = [
			("Spine (pelvis→torso)",  [0,3,6,9],   0.236151),
			("Neck+Head (torso→head)", [9,12,15],   0.223894),
			("R Upper Arm",           [17,19],      0.274788),
			("R Lower Arm",           [19,21],      0.258947),
			("R Thigh",               [2,5],        0.421546),
			("R Shin",                [5,8],        0.409870),
			("R Foot (ankle→toe)",    [8,11],       0.050),
			("L Upper Arm",           [16,18],      0.274788),
			("L Thigh",               [1,4],        0.421546),
		]
		print(f"\n{'Segment':30s}  {'SMPL':>8s}  {'MuJoCo':>8s}  {'Ratio':>7s}")
		print("-"*60)
		for name, jlist, mj_len in pairs:
			smpl_len = 0
			for i in range(len(jlist)-1):
				bone = smpl_rest[jlist[i+1]] - smpl_rest[jlist[i]]
				smpl_len += np.linalg.norm(bone)
			ratio = mj_len / smpl_len if smpl_len > 0.001 else 0
			print(f"  {name:28s}  {smpl_len:8.4f}  {mj_len:8.4f}  {ratio:7.3f}")
		print(f"\n  Height scale factor: {self.height_scale:.4f}")

	# ═══════════════════════════════════════════════════════════════════════
	# Floor correction via SMPL FK
	# ═══════════════════════════════════════════════════════════════════════

	def compute_floor_offset(self, smpl_poses, smpl_trans):
		"""
		Compute the SMPL floor level from foot/ankle joint positions.

		Uses SMPL FK to find the lowest point any foot/ankle joint reaches
		across ALL frames.  This is the floor level in the capture volume.

		For motions with jumps, the lowest point is when feet touch the ground.
		During airborne frames, feet will naturally be above this level.

		Returns: floor_y (float) — height to subtract from smpl_trans[:, 1]
		"""
		if self.smpl_handler is None or not hasattr(self.smpl_handler, 'run_fk'):
			print("  ⚠ No SMPL handler — falling back to pelvis-based floor estimate")
			return float(smpl_trans[:, 1].min()) - 0.88

		poses_flat = np.array(smpl_poses, dtype=np.float32).reshape(-1, 72)
		trans_np = np.array(smpl_trans, dtype=np.float32)
		joints_3d = self.smpl_handler.run_fk(poses_flat, trans_np)
		if joints_3d is None:
			return float(smpl_trans[:, 1].min()) - 0.88

		return self.get_avg_floor_height(joints_3d)

	# ═══════════════════════════════════════════════════════════════════════
	# Quaternion / composition helpers
	# ═══════════════════════════════════════════════════════════════════════

	@staticmethod
	def _to_xyzw(q_wxyz_list):
		"""Reorder PyTorch3D [w,x,y,z] → CompositeMotion JSON [x,y,z,w]."""
		w, x, y, z = q_wxyz_list
		return [x, y, z, w]

	@staticmethod
	def _qmul(q0, q1):
		"""Hamilton product for batched PyTorch3D [w,x,y,z] quaternions (N, 4)."""
		w0, x0, y0, z0 = q0.unbind(-1)
		w1, x1, y1, z1 = q1.unbind(-1)
		return torch.stack([
			w0*w1 - x0*x1 - y0*y1 - z0*z1,
			w0*x1 + x0*w1 + y0*z1 - z0*y1,
			w0*y1 - x0*z1 + y0*w1 + z0*x1,
			w0*z1 + x0*y1 - y0*x1 + z0*w1,
		], dim=-1)

	@staticmethod
	def _qconj(q):
		"""Quaternion conjugate for [w,x,y,z] — negates the vector part."""
		return q * torch.tensor([1, -1, -1, -1], dtype=q.dtype, device=q.device)

	def _compose(self, rot_quats, indices):
		"""Compose LOCAL [w,x,y,z] quaternions along a SMPL joint chain."""
		q = rot_quats[:, indices[0]]
		for idx in indices[1:]:
			q = self._qmul(q, rot_quats[:, idx])
		return q

	# ═══════════════════════════════════════════════════════════════════════
	# Approach 2: FK-based retargeting (position-guided)
	# ═══════════════════════════════════════════════════════════════════════

	def process_motion_fk(self, smpl_poses, smpl_trans, output_path, fps=30):
		"""
		Alternative FK-based conversion: uses SMPL joint POSITIONS to compute
		MuJoCo bone directions, combined with SMPL orientations for twist.

		This handles bone length differences naturally because bone directions
		come from actual (scaled) world positions rather than rotations.
		"""
		N = smpl_poses.shape[0]
		G = self.global_rot.cpu().numpy()
		trans_np = np.array(smpl_trans, dtype=np.float32)
		poses_np = np.array(smpl_poses, dtype=np.float32).reshape(-1, 72)
		poses_t = torch.tensor(smpl_poses, dtype=torch.float32
							).reshape(N, 24, 3).to(self.device)

		# FK for world positions
		joints_3d = self.smpl_handler.run_fk(poses_np, trans_np)  # (N, 45, 3)
		if joints_3d is None:
			print("  ⚠ FK failed, falling back to rotation-based method")
			return self.process_motion(smpl_poses, smpl_trans, output_path, fps)
		joints = joints_3d[:, :24, :]  # (N, 24, 3) in SMPL Y-up

		# Floor correction from positions
		floor_y = self.get_avg_floor_height(joints)
		joints[:, :, 1] -= floor_y

		# Scale + convert to MuJoCo Z-up
		joints_mj = np.einsum('ij,ntj->nti', G, joints) * self.height_scale

		# Root position (from FK pelvis — already accounts for body offset)
		root_pos_np = joints_mj[:, 0, :]  # pelvis position
		print(f"  [FK] Floor: {floor_y:.3f}m | Scale: {self.height_scale:.4f} | "
			f"Pelvis Z: [{root_pos_np[:,2].min():.3f}, {root_pos_np[:,2].max():.3f}]")

		# Similarity-transformed SMPL world orientations (for twist extraction)
		rot_mats = transforms.axis_angle_to_matrix(poses_t)  # (N, 24, 3, 3)
		G_t = self.global_rot
		G_b = G_t.unsqueeze(0).unsqueeze(0)
		rot_mats_mj = torch.matmul(torch.matmul(G_b, rot_mats),
									G_b.transpose(-1, -2))

		# Accumulate world orientations via SMPL kinematic tree
		world_orient = torch.zeros(N, 24, 3, 3, device=self.device)
		for j in range(24):
			p = self.SMPL_PARENTS[j]
			if p < 0:
				world_orient[:, j] = rot_mats_mj[:, j]
			else:
				world_orient[:, j] = torch.bmm(world_orient[:, p], rot_mats_mj[:, j])

		# For each MuJoCo body: compute local rotation from positions + twist
		body_quats_np = {}
		smpl_tip_map = {b: info[0] for b, info in self._bone_info.items()}
		smpl_child_map = {b: info[1] for b, info in self._bone_info.items()}
		mj_bone_map = {b: info[2] for b, info in self._bone_info.items()}

		pelvis_orient = world_orient[:, 0]  # (N, 3, 3)
		mj_world_orients = {"pelvis": pelvis_orient}

		for body_name, chain in self.body_chain.items():
			parent_name = self._mujoco_parent[body_name]
			tip_j = smpl_tip_map[body_name]
			child_j = smpl_child_map[body_name]
			mj_bone = mj_bone_map[body_name]

			if child_j is None or mj_bone is None:
				# Leaf: use similarity-transformed SMPL orientation directly
				smpl_world = world_orient[:, tip_j]
				mj_world_orients[body_name] = smpl_world
			else:
				d_rest = np.array(mj_bone, dtype=np.float64)
				d_rest = d_rest / np.linalg.norm(d_rest)

				parent_orient = mj_world_orients[parent_name]
				local_quats = []
				for i in range(N):
					p_body = joints_mj[i, tip_j]
					p_child = joints_mj[i, child_j]
					d_target = p_child - p_body
					d_len = np.linalg.norm(d_target)
					if d_len < 1e-6:
						local_quats.append(torch.tensor([1,0,0,0], dtype=torch.float32))
						continue
					d_target = d_target / d_len

					# Align rest bone dir to target bone dir
					R_align = self._rotation_between(d_rest, d_target)
					R_align_t = torch.tensor(R_align, dtype=torch.float32, device=self.device)

					# Body world orient = parent_world @ local_q
					# We want: orient @ d_rest = d_target
					# orient = parent_world @ local_q  →  local_q = parent^{-1} @ orient
					R_parent = parent_orient[i]  # (3, 3)
					R_body_world = R_align_t
					R_local = R_parent.T @ R_body_world
					q_local = transforms.matrix_to_quaternion(R_local.unsqueeze(0))[0]
					local_quats.append(q_local)

				body_q = torch.stack(local_quats)  # (N, 4)
				body_quats_np[body_name] = body_q

				# Update world orient for children
				parent_orient_np = mj_world_orients[parent_name]
				new_orient = torch.bmm(parent_orient_np, 
					transforms.quaternion_to_matrix(body_q))
				mj_world_orients[body_name] = new_orient

		# Pelvis quaternion
		q_pelvis = transforms.matrix_to_quaternion(pelvis_orient)  # (N, 4) wxyz

		# Assemble frames
		frames = []
		for i in range(N):
			fd = {}
			fd["pelvis"] = [
				root_pos_np[i].tolist(),
				self._to_xyzw(q_pelvis[i].tolist())
			]
			for body_key, bq in body_quats_np.items():
				fd[body_key] = self._to_xyzw(bq[i].tolist())
			frames.append(fd)

		stem = (output_path.stem if hasattr(output_path, 'stem')
				else os.path.splitext(os.path.basename(str(output_path)))[0])
		output_data = {"_": stem, "fps": fps, "loop": "none", "frames": frames}
		try:
			self.motion_dump(output_data, str(output_path))
			print(f"  ✅ [FK] Saved: {output_path}  ({N} frames @ {fps} Hz = {N/fps:.1f} s)")
		except Exception as e:
			print(f"  ❌ [FK] Failed '{output_path}': {e}")

	# ── JSON writer ──────────────────────────────────────────────────────────

	def motion_dump(self, save_data, save_path, indent=2, compress_arrays=True):
		"""Write JSON with readable dict structure and compact numeric arrays."""
		def _is_num(obj):
			return isinstance(obj, (int, float)) or (
				isinstance(obj, list) and bool(obj) and all(_is_num(i) for i in obj)
			)
		def _enc(obj, lv=0):
			pad  = ' ' * (lv * indent)
			npad = ' ' * ((lv + 1) * indent)
			if isinstance(obj, dict):
				if not obj: return '{}'
				rows = []
				for i, (k, v) in enumerate(obj.items()):
					comma = '' if i == len(obj) - 1 else ','
					rows.append(f'{npad}"{k}": {_enc(v, lv+1)}{comma}')
				return '{\n' + '\n'.join(rows) + f'\n{pad}}}'
			if isinstance(obj, list):
				if compress_arrays and _is_num(obj):
					return json.dumps(obj, separators=(',', ':'))
				if not obj: return '[]'
				rows = []
				for i, v in enumerate(obj):
					comma = '' if i == len(obj) - 1 else ','
					rows.append(f'{npad}{_enc(v, lv+1)}{comma}')
				return '[\n' + '\n'.join(rows) + f'\n{pad}]'
			return json.dumps(obj)
		abs_path = os.path.abspath(str(save_path))
		os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
		with open(abs_path, 'w') as f:
			f.write(_enc(save_data) + '\n')

	# ── Main conversion ──────────────────────────────────────────────────────

	def process_motion(self, smpl_poses, smpl_trans, output_path, fps=30):
		"""
		Convert SMPL motion data → CompositeMotion DeepMimic JSON.

		Args
		────
		smpl_poses  : (N, 72) or (N, 24, 3)  SMPL axis-angle joint rotations
		smpl_trans  : (N, 3)  root translation in AIST++ world (SMPL Y-up)
		output_path : str or pathlib.Path
		fps         : int  (default 30)
		"""
		N = smpl_poses.shape[0]
		trans = np.array(smpl_trans, dtype=np.float32)
		poses = torch.tensor(smpl_poses, dtype=torch.float32
							).reshape(N, 24, 3).to(self.device)

		# ── Step 1a: Get SMPL FK pelvis world positions ───────────────────
		# IMPORTANT: SMPL `transl` is NOT the pelvis position. The actual
		# pelvis world pos = FK(pose)_pelvis + transl.  In the SMPL neutral
		# model the rest-pose pelvis is ~0.22 m below the origin.  Using raw
		# `trans` as root position causes a systematic upward offset.
		poses_flat = np.array(smpl_poses, dtype=np.float32).reshape(-1, 72)
		trans_orig = np.array(smpl_trans, dtype=np.float32)

		j3d = None
		if self.smpl_handler is not None and hasattr(self.smpl_handler, 'run_fk'):
			j3d = self.smpl_handler.run_fk(poses_flat, trans_orig)

		if j3d is not None:
			pelvis_pos = j3d[:, 0, :].copy()  # (N, 3) actual pelvis world pos
			floor_y = self.get_avg_floor_height(j3d)
		else:
			pelvis_pos = trans.copy()
			floor_y = float(trans[:, 1].min()) - 0.88  # --> avg height offset as default

		# ── Step 1b: Height scaling + Floor correction ────────────────────
		pelvis_pos *= self.height_scale
		pelvis_pos[:, 1] -= floor_y
		print(f"  Floor: {floor_y:.3f}m | Scale: {self.height_scale:.4f} | "
			f"Pelvis Z range: [{pelvis_pos[:,1].min():.3f}, {pelvis_pos[:,1].max():.3f}]")

		# ── Step 2: Root position — SMPL → MuJoCo via cyclic permutation ─
		pelvis_t = torch.tensor(pelvis_pos, device=self.device, dtype=torch.float32)
		root_pos = torch.matmul(pelvis_t, self.global_rot.T)    # (N, 3)

		# ── Step 3: Axis-angle → rotation matrices ────────────────────────
		rot_mats = transforms.axis_angle_to_matrix(poses)      # (N, 24, 3, 3)

		# ── Step 4: Similarity transform on ALL 24 joints ─────────────────
		G   = self.global_rot                                   # (3, 3)
		G_b = G.unsqueeze(0).unsqueeze(0)                       # (1, 1, 3, 3)
		rot_mats = torch.matmul(
			torch.matmul(G_b, rot_mats),
			G_b.transpose(-1, -2)
		)                                                       # (N, 24, 3, 3)

		# ── Step 5: Rotation matrices → quaternions [w, x, y, z] ─────────
		rot_quats = transforms.matrix_to_quaternion(rot_mats)   # (N, 24, 4)

		# ── Step 6: Compose chains + apply bind rotations ─────────────────
		# R_adjusted[b] = conj(R_bind[parent]) ⊗ R_sim[b] ⊗ R_bind[b]
		body_quats = {}
		for key, indices in self.body_chain.items():
			r_sim = self._compose(rot_quats, indices)

			parent_name = self._mujoco_parent[key]
			q_bind_b = transforms.matrix_to_quaternion(
				self.bind_rots[key].unsqueeze(0))[0]
			q_bind_p = transforms.matrix_to_quaternion(
				self.bind_rots[parent_name].unsqueeze(0))[0]

			q_bp_inv = self._qconj(q_bind_p).unsqueeze(0).expand(N, -1)
			q_bb_exp = q_bind_b.unsqueeze(0).expand(N, -1)

			body_quats[key] = self._qmul(self._qmul(q_bp_inv, r_sim), q_bb_exp)

		# ── Step 6b: Pelvis orientation ───────────────────────────────────
		q_pelvis_sim = rot_quats[:, 0]
		q_bind_pel = transforms.matrix_to_quaternion(
			self.bind_rots["pelvis"].unsqueeze(0))[0]
		q_pelvis = self._qmul(q_pelvis_sim,
							q_bind_pel.unsqueeze(0).expand(N, -1))

		# ── Step 7: Assemble per-frame dicts ──────────────────────────────
		frames = []
		for i in range(N):
			fd = {}
			fd["pelvis"] = [
				root_pos[i].tolist(),
				self._to_xyzw(q_pelvis[i].tolist())
			]
			for body_key, bq in body_quats.items():
				fd[body_key] = self._to_xyzw(bq[i].tolist())
			frames.append(fd)

		# ── Step 8: Write JSON ─────────────────────────────────────────────
		stem = (output_path.stem if hasattr(output_path, 'stem')
				else os.path.splitext(os.path.basename(str(output_path)))[0])
		output_data = {
			"_":      stem,
			"fps":    fps,
			"loop":   "none",
			"frames": frames,
		}
		try:
			self.motion_dump(output_data, str(output_path))
			print(f"  ✅ Saved: {output_path}  ({N} frames @ {fps} Hz = {N/fps:.1f} s)")
		except Exception as e:
			print(f"  ❌ Failed '{output_path}': {e}")


