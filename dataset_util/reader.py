"""Utility for reading the dataset
"""
from .pose import Pose
import numpy as np
import cv2


def read_rgb(path):
    """Read image in RGB
    """
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


class Dataset:
    """Representation of a dataset
    """
    def __init__(self, folder_path):
        """Read the dataset from folder
        """
        self.poses = {} # idx to pose

        # Read groundtruth poses
        gt_path = folder_path + "/groundtruth.txt"
        with open(gt_path, 'r') as f:
            line = f.readline()
            while line:
                values = line.split()
                assert(len(values) == 8)
                idx = int(values[0])
                t = [float(x) for x in values[1:4]]
                q = [float(x) for x in values[4:]]
                self.poses[idx] = Pose.from_t_q(t, q)
                line = f.readline()

        self.rgb_folder = folder_path + "/rgb"
        self.min_idx = 1
        self.max_idx = len(self.poses)

    def relative_pose(self, i, j):
        """Get the relative pose from i to j
        """
        pose_i = self.poses[i]
        pose_j = self.poses[j]
        return pose_i.inverse() * pose_j

    def image_at(self, idx):
        """Get RGB image at idx
        """
        path = self.rgb_folder + "/{}.png".format(idx)
        return read_rgb(path)


class TrajectoryData:
    """Representation of the trajectory data
    """
    def __init__(self, path):
        """Read relative poses data and reconstruct the poses
        """
        initial_pose = Pose()
        # Rotate such that camera z-axis facing front
        initial_pose.R = np.array([[1, 0, 0],
                                   [0, 0, 1],
                                   [0, -1, 0]])
        self.poses = [initial_pose]
        with open(path, "r") as f:
            line = f.readline()
            while line and line.strip():
                values = line.split()
                assert(len(values) == 9)
                # Read relative pose
                t = [float(x) for x in values[2:5]]
                q = [float(x) for x in values[5:]]
                relative_pose = Pose.from_t_q(t, q)

                # Construct next pose
                current_pose = self.poses[-1]
                self.poses.append(current_pose * relative_pose)

                line = f.readline()
