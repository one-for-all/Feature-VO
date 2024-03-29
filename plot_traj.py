"""Plot the estimated and groundtruth trajectory
"""
import argparse
from dataset_util.reader import TrajectoryData
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from plot_util.plot import plot_positions, plot_poses
from pose_estimation.estimation import compute_traj_error


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot the estimated and ground truth trajectories")
    parser.add_argument("estimate_fpath", type=str, nargs='?',
                         default="estimate.txt", help="estimate output file path")
    parser.add_argument("gt_fpath", type=str, nargs='?',
                         default="gt.txt", help="groudtruth output file path")
    args = parser.parse_args()

    # Read trajectories
    estimate_traj = TrajectoryData(args.estimate_fpath)
    gt_traj = TrajectoryData(args.gt_fpath)

    # Plot trajectories
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.canvas.draw()
    print("Trajectory error per frame pair: {}".format(
        compute_traj_error(gt_traj, estimate_traj)))
    plot_poses(ax, gt_traj.poses, length=0.1, style="--")
    plot_poses(ax, estimate_traj.poses, length=0.1, style="-")
    plt.show()
