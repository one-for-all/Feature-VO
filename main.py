"""Estimate the trajectory given a sequence of images
"""

import argparse
from dataset_util.reader import Dataset, TrajectoryData
from dataset_util.writer import RelativePoseWriter
from pose_estimation.estimation import (PoseEstimator, fix_scale,
    compute_t_error, compute_traj_error)
import numpy as np
import cv2
from tqdm import tqdm

import random
random.seed(3)

cv2.setRNGSeed(3)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Estimate the trajectory given a sequence of images")
    parser.add_argument("dataset_folder", type=str, help="path to dataset",
                        nargs='?', default="./datasets/living_room_traj2_frei_png")
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

    window_a = "image 1"
    window_b = "image 2"

    errors = []

    for step in range(30, 0, -2):

        # Iterate through images and estimate relative poses
        estimate_writer = RelativePoseWriter(args.estimate_fpath)
        gt_writer = RelativePoseWriter(args.gt_fpath)

        pose_estimator = PoseEstimator(K=K, method="SIFT")

        start = 1 #dataset.min_idx
        end = dataset.max_idx

        estimate_scale = False

        pos_errors = []
        for i in tqdm(range(start, end-step+1, step)):
            idx = i
            im = dataset.image_at(idx)

            # Display
            last_img = im
            if pose_estimator.last_img is not None:
                last_img = pose_estimator.last_img
            cv2.imshow(window_a, last_img)
            cv2.imshow(window_b, im)

            # Process image
            pose_estimator.process(im, idx, estimate_scale=estimate_scale)

            if len(pose_estimator.frame_pairs) == 0:
                continue

            # Get relative pose
            frame_pair = pose_estimator.frame_pairs[-1]
            estimated_pose = frame_pair.pose
            a_idx = frame_pair.idx1
            b_idx = frame_pair.idx2
            true_pose = dataset.relative_pose(a_idx, b_idx)


            # Correct the scale for first pair
            if len(pose_estimator.frame_pairs) == 1 or not estimate_scale:
                frame_pair.pose = fix_scale(true_pose, estimated_pose)

            # Compute translation error
            pos_errors.append( compute_t_error(true_pose, estimated_pose) )
            # print("Average t error per frame pair: {}".format(np.average(pos_errors)))

            # norm1 = np.linalg.norm(true_pose.t)
            # norm2 = np.linalg.norm(estimated_pose.t)
            # print("scale ratio: {}".format(norm1 / norm2))

            # Write the results
            estimate_writer.write_pose(a_idx, b_idx, estimated_pose)
            gt_writer.write_pose(a_idx, b_idx, true_pose)

            # Time for displaying video
            cv2.waitKey(25)

        estimate_writer.close()
        gt_writer.close()

        estimate_traj = TrajectoryData("estimate.txt")
        gt_traj = TrajectoryData("gt.txt")
        print("step size: {}".format(step))
        error = compute_traj_error(gt_traj, estimate_traj)
        errors.append(error)
        print("Trajectory error per frame pair: {}".format(error))
        print("=================================")

    print(errors)

    # for i in tqdm(range(start, end-step+1, step)):
    #     a_idx = i
    #     b_idx = i+step
    #     im_a = dataset.image_at(a_idx)
    #     im_b = dataset.image_at(b_idx)

    #     # Display
    #     cv2.imshow(window_a, im_a)
    #     cv2.imshow(window_b, im_b)

    #     # Compute relative pose
    #     estimated_pose = pose_estimator.estimate(im_a, im_b)
    #     true_pose = dataset.relative_pose(a_idx, b_idx)
    #     estimated_pose = fix_scale(true_pose, estimated_pose)

    #     # Write the results
    #     estimate_writer.write_pose(a_idx, b_idx, estimated_pose)
    #     gt_writer.write_pose(a_idx, b_idx, true_pose)

    #     # Time for displaying video
    #     cv2.waitKey(25)

    # cv2.destroyAllWindows()