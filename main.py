"""Estimate the trajectory given a sequence of images
"""

import argparse
from dataset_util.reader import Dataset
from dataset_util.writer import RelativePoseWriter
from image_processing.util import read_rgb
from pose_estimation.estimation import PoseEstimator, fix_scale
import numpy as np
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Estimate the trajectory given a sequence of images")
    parser.add_argument("dataset_folder", type=str, help="path to dataset",
                        nargs='?', default="./datasets/traj1_frei_png")
    parser.add_argument("estimate_fpath", type=str, nargs='?', 
                         default="estimate.txt", help="estimate output file path")
    parser.add_argument("gt_fpath", type=str, nargs='?', 
                         default="gt.txt", help="groudtruth output file path")
    args = parser.parse_args()

    # Read the dataset
    dataset = Dataset(args.dataset_folder)

    # TODO: read camera K
    K = np.array([[481.20, 0, 319.50],
                  [0, -480.00, 239.50],
                  [0, 0, 1]])

    # Iterate through images and estimate relative poses
    estimate_writer = RelativePoseWriter(args.estimate_fpath)
    gt_writer = RelativePoseWriter(args.gt_fpath)

    window_a = "image 1"
    window_b = "image 2"

    pose_estimator = PoseEstimator(K=K, method="SIFT")

    start = 1 #dataset.min_idx
    step = 10
    end = dataset.max_idx
    for i in tqdm(range(start, end-step+1, step)):
        a_idx = i
        b_idx = i+step
        im_a_path = dataset.rgb_folder + "/{}.png".format(a_idx)
        im_b_path = dataset.rgb_folder + "/{}.png".format(b_idx)
        im_a = read_rgb(im_a_path)
        im_b = read_rgb(im_b_path)
        
        # Display
        cv2.imshow(window_a, im_a)
        cv2.imshow(window_b, im_b)

        # Compute relative pose
        estimated_pose = pose_estimator.estimate(im_a, im_b)
        true_pose = dataset.relative_pose(a_idx, b_idx)
        estimated_pose = fix_scale(true_pose, estimated_pose)

        # Write the results
        estimate_writer.write_pose(a_idx, b_idx, estimated_pose)
        gt_writer.write_pose(a_idx, b_idx, true_pose)

        # Time for displaying video
        cv2.waitKey(25)

    cv2.destroyAllWindows()