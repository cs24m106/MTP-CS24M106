import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
# Simple 3D skeleton visualization with matplotlib
from mpl_toolkits.mplot3d import Axes3D
from .loss_utils import smpl_skeleton

import matplotlib as mpl
from imageio_ffmpeg import get_ffmpeg_exe
# Set ffmpeg path for matplotlib BEFORE importing animation modules
mpl.rcParams['animation.ffmpeg_path'] = get_ffmpeg_exe()
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def animate_3d_skeleton(joint_positions, out_path='motion.gif', 
					fps=30, dpi=100, elev=15, azim=45,
					create_subplot=False, subplot_interval=15):
	"""
	Create animated GIF of 3D skeleton motion with XZ plane as ground (Y-axis up)
	
	Args:
		joint_positions: (T, J, 3) - T frames, J joints, 3D coordinates (X, Y, Z)
							Assumes Y-axis is vertical (up)
		out_path: Path to save GIF
		fps: Frames per second
		dpi: Image resolution
		elev: Elevation angle for 3D view
		azim: Azimuth angle for 3D view
		create_subplot: If True, also create subplot visualization
		subplot_interval: Interval between frames in subplot (default: 15)
	"""

	if isinstance(out_path, str):
		out_path = Path(out_path)
	
	title = out_path.stem
	T, J, _ = joint_positions.shape
	
	# Set up figure for animation
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')
	
	# Calculate bounds for consistent axis limits
	all_points = joint_positions.reshape(-1, 3)
	x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
	y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
	z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
	
	# Make axes equal
	max_range = max(x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0])
	mid_x = np.mean(x_range)
	mid_y = np.mean(y_range)
	mid_z = np.mean(z_range)
	
	def check_subplot():
		# Create subplot visualization if requested
		if create_subplot:
			subplot_path = out_path.with_name(out_path.stem + '_frames.png')
			create_frame_subplot(joint_positions, subplot_path, subplot_interval, 
							elev, azim, mid_x, mid_y, mid_z, max_range, 
							x_range, y_range, z_range)
	if out_path.exists():
		#print(f"Animation save already exists! (skipping)")
		check_subplot()
		return out_path
	
	def init():
		ax.clear()
		ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
		ax.set_ylim(mid_z - max_range/2, mid_z + max_range/2)
		ax.set_zlim(mid_y - max_range/2, mid_y + max_range/2)
		ax.set_xlabel('X (horizontal)', fontsize=10, fontweight='bold')
		ax.set_ylabel('Z (depth)', fontsize=10, fontweight='bold')
		ax.set_zlabel('Y (vertical)', fontsize=10, fontweight='bold')
		ax.view_init(elev=elev, azim=azim)
		return []
	
	def update(frame):
		ax.clear()
		
		joints = joint_positions[frame]  # (J, 3) - [X, Y, Z]
		
		# Reorder axes: matplotlib's Z-axis → our Y-axis (vertical)
		# So we plot: (X, Z, Y) → (x_plot, y_plot, z_plot)
		x_plot = joints[:, 0]  # X stays X
		y_plot = joints[:, 2]  # Z becomes Y in plot (horizontal depth)
		z_plot = joints[:, 1]  # Y becomes Z in plot (vertical)
		
		# Plot joints
		ax.scatter(x_plot, y_plot, z_plot, 
					c='red', s=50, alpha=0.8, marker='o')
		
		# Plot skeleton connections
		for connection in smpl_skeleton:
			if connection[0] < J and connection[1] < J:
				start = joints[connection[0]]
				end = joints[connection[1]]
				
				# Reorder axes for line plotting
				ax.plot([start[0], end[0]],  # X
						[start[2], end[2]],   # Z → Y in plot
						[start[1], end[1]],   # Y → Z in plot
						'b-', linewidth=2, alpha=0.7)
		
		# Draw ground plane (XZ plane at Y=0 or minimum Y)
		ground_y = y_range[0]  # Minimum Y value (ground level)
		
		# Create ground grid
		xx, zz = np.meshgrid(
			np.linspace(mid_x - max_range/2, mid_x + max_range/2, 10),
			np.linspace(mid_z - max_range/2, mid_z + max_range/2, 10)
		)
		yy = np.ones_like(xx) * ground_y
		
		# Plot ground plane (reordered for visualization)
		ax.plot_surface(xx, zz, yy, alpha=0.2, color='whitesmoke', 
						edgecolor='lightgray', linewidth=0.5)
		
		# Set consistent limits (reordered)
		ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
		ax.set_ylim(mid_z - max_range/2, mid_z + max_range/2)  # Z range
		ax.set_zlim(mid_y - max_range/2, mid_y + max_range/2)  # Y range (vertical)
		
		ax.set_xlabel('X (horizontal)', fontsize=10, fontweight='bold')
		ax.set_ylabel('Z (depth)', fontsize=10, fontweight='bold')
		ax.set_zlabel('Y (vertical)', fontsize=10, fontweight='bold')
		ax.set_title(f'{title} Frame {frame}/{T}', fontsize=14, fontweight='bold')
		ax.view_init(elev=elev, azim=azim)
		
		return []
	
	# Save animation as GIF or MP4 based on file extension
	out_path.parents[0].mkdir(parents=True, exist_ok=True)

	# Create animation
	anim = FuncAnimation(fig, update, frames=T, init_func=init, 
					interval=1000/fps, blit=False, repeat=True)

	# Determine format and writer
	ext = out_path.suffix.lower()

	if ext in ['.mp4']:
		writer = FFMpegWriter(fps=fps)
		fmt = 'MP4'
	else:
		# Default to GIF if unknown extension
		writer = PillowWriter(fps=fps)
		fmt = 'GIF'

	anim.save(out_path, writer=writer, dpi=dpi)
	plt.close()
	print(f"Animation saved as {fmt} ({T} frames at {fps} fps) @{str(out_path.relative_to(Path.cwd()))}")
	check_subplot()
	return out_path


def create_frame_subplot(joint_positions, save_path, interval=15, 
						elev=15, azim=45, mid_x=0, mid_y=0, mid_z=0, 
						max_range=1, x_range=None, y_range=None, z_range=None):
	"""
	Create a single-row subplot showing frames at regular intervals
	
	Args:
		joint_positions: (T, J, 3) - T frames, J joints, 3D coordinates
		save_path: Path to save the subplot image
		interval: Frame interval for subplots
		elev, azim: 3D view angles
		mid_x, mid_y, mid_z: Axis midpoints
		max_range: Maximum range for consistent scaling
		x_range, y_range, z_range: Original axis ranges
	"""
	T, J, _ = joint_positions.shape
	
	# Recalculate ranges if not provided
	if x_range is None or y_range is None or z_range is None:
		all_points = joint_positions.reshape(-1, 3)
		x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
		y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
		z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
		
		max_range = max(x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0])
		mid_x = np.mean(x_range)
		mid_y = np.mean(y_range)
		mid_z = np.mean(z_range)
	
	# Select frames at intervals
	frame_indices = np.arange(0, T, interval)
	if frame_indices[-1] != T - 1:
		frame_indices = np.append(frame_indices, T - 1)  # Always include last frame
	
	n_subplots = len(frame_indices)
	
	# Create figure with subplots in a single row
	fig = plt.figure(figsize=(4 * n_subplots, 5))
	
	for idx, frame in enumerate(frame_indices):
		ax = fig.add_subplot(1, n_subplots, idx + 1, projection='3d')
		
		joints = joint_positions[frame]  # (J, 3) - [X, Y, Z]
		
		# Reorder axes for plotting
		x_plot = joints[:, 0]  # X stays X
		y_plot = joints[:, 2]  # Z becomes Y in plot (horizontal depth)
		z_plot = joints[:, 1]  # Y becomes Z in plot (vertical)
		
		# Plot joints
		ax.scatter(x_plot, y_plot, z_plot, 
					c='red', s=30, alpha=0.8, marker='o')
		
		# Plot skeleton connections
		for connection in smpl_skeleton:
			if connection[0] < J and connection[1] < J:
				start = joints[connection[0]]
				end = joints[connection[1]]
				
				ax.plot([start[0], end[0]],
						[start[2], end[2]],
						[start[1], end[1]],
						'b-', linewidth=1.5, alpha=0.7)
		
		# Draw ground plane
		ground_y = y_range[0]
		xx, zz = np.meshgrid(
			np.linspace(mid_x - max_range/2, mid_x + max_range/2, 10),
			np.linspace(mid_z - max_range/2, mid_z + max_range/2, 10)
		)
		yy = np.ones_like(xx) * ground_y
		
		ax.plot_surface(xx, zz, yy, alpha=0.15, color='whitesmoke', 
						edgecolor='lightgray', linewidth=0.5)
		
		# Set consistent limits
		ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
		ax.set_ylim(mid_z - max_range/2, mid_z + max_range/2)
		ax.set_zlim(mid_y - max_range/2, mid_y + max_range/2)
		
		# Smaller labels for subplots
		ax.set_xlabel('X', fontsize=8)
		ax.set_ylabel('Z', fontsize=8)
		ax.set_zlabel('Y', fontsize=8)
		ax.set_title(f'Frame {frame}', fontsize=10, fontweight='bold')
		ax.view_init(elev=elev, azim=azim)
		
		# Smaller tick labels
		ax.tick_params(labelsize=6)
	
	plt.tight_layout()
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	plt.close()
	
	print(f"Frame subplot saved ({n_subplots} frames) @{str(save_path.relative_to(Path.cwd()))}")
	return save_path
