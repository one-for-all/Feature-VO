"""Plot given the trajectory data
"""
import numpy as np


def plot_positions(ax, poses, color='green'):
    """Plot the 3D trajectory's positions
    """
    traj = []
    for pose in poses:
        traj.append(pose.t)
    traj = np.asarray(traj)
    assert(traj.shape == (len(poses), 3))
    ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color)


def plot_pose(ax, pose, length=1, style='-'):
    colors = ['r', 'g', 'b']
    lines = sum([ax.plot([], [], [], c=c) for c in colors], [])
    pos = pose.t
    start_points = np.array([pos, pos, pos])
    # end_points = np.array([[length, 0, 0], [0, 0, -length], [0, length, 0]]).T
    end_points = np.array([[length, 0, 0], [0, length, 0], [0, 0, length]]).T
    end_points = np.matmul(pose.R, end_points).T + start_points
    
    for line, start, end in zip(lines, start_points, end_points):
        line.set_data((start[0], end[0]), (start[1], end[1]))
        line.set_3d_properties((start[2], end[2]))
        line.set_linestyle(style)
        ax.draw_artist(line)


def plot_poses(ax, poses, length=1, style='-'):
    for pose in poses:
        plot_pose(ax, pose, length=length, style=style)