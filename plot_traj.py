"""Plot the estimated and groundtruth trajectory
"""
import argparse
from dataset_util.reader import TrajectoryData
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from plot_util.plot import plot_positions, plot_poses


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot the estimated and ground truth trajectories")
    parser.add_argument("estimate_fpath", type=str, nargs='?', 
                         default="estimate.txt", help="estimate output file path")
    parser.add_argument("gt_fpath", type=str, nargs='?', 
                         default="gt.txt", help="groudtruth output file path")
    args = parser.parse_args()

    estimate_traj = TrajectoryData(args.estimate_fpath)
    gt_traj = TrajectoryData(args.gt_fpath)

    # print(gt_traj.poses[-1])

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.canvas.draw()
    # plot_positions(ax, gt_traj.poses, color="green")
    # plot_positions(ax, estimate_traj.poses, color="blue")
    plot_poses(ax, gt_traj.poses, length=0.1, style="--")
    plot_poses(ax, estimate_traj.poses, length=0.1, style="-")
    plt.show()
