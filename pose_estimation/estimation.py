"""Estimate relative pose between two images
"""
import cv2
import numpy as np
import sys
import copy
sys.path.append("../")
from dataset_util.pose import Pose

def fix_scale(pose1, pose2):
    """Fix the scale of pose 2 translation such that it has same norm as pose 1
    """
    norm1 = np.linalg.norm(pose1.t)
    norm2 = np.linalg.norm(pose2.t)
    fixed_pose = copy.copy(pose2)
    if not np.isclose(norm2, 0):
        fixed_pose.t *= norm1/norm2
    return fixed_pose

class PoseEstimator:
    """Estimator for computing relative pose from 2 images
    """
    def __init__(self, K, method="SIFT"):
        """
        Args:
            K: camera matrix, 3x3
        """
        if method == "SIFT": 
            self.detector = cv2.xfeatures2d.SIFT_create()
        elif method == "ORB":
            self.detector = cv2.ORB_create()
        self.method = method
        self.K = K

    def estimate(self, im1, im2):
        """Estimate the relative pose from 2 images
        """
        detector = self.detector
        
        def find_kp(im):
            """Detect keypoints
            """
            kp = detector.detect(im, None)
            return kp
        
        kp1 = find_kp(im1)
        kp2 = find_kp(im2)
        kp1, des1 = detector.compute(im1, kp1)
        kp2, des2 = detector.compute(im2, kp2)

        def match_kp(des1, des2):
            """Match keypoints
            Return:
                list of good matches
            """
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)

            if self.method == "SIFT":
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                matches = matcher.knnMatch(des1, des2, k=2)
                matches_mask = [[0,0] for i in range(len(matches))]

                good_matches = []
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        matches_mask[i]=[1,0]
                        good_matches.append(m)
            elif self.method == "ORB":
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                good_matches = matcher.match(des1, des2)

            return good_matches
        
        good_matches = match_kp(des1, des2)

        def get_matched_pts(kp1, kp2, good_matches):
            """Get points that are good matches
            Returns:
                numpy Nx2 array of points in image 1
                numpy Nx2 array of points in image 2
            """
            matched_kp1 = []
            matched_kp2 = []
            for match in good_matches:
                matched_kp1.append(kp1[match.queryIdx].pt)
                matched_kp2.append(kp2[match.trainIdx].pt)
            return np.asarray(matched_kp1), np.asarray(matched_kp2)
        
        pts1, pts2 = get_matched_pts(kp1, kp2, good_matches)
        
        # Compute relative pose
        pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), 
                        cameraMatrix=self.K, distCoeffs=None)
        pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), 
                        cameraMatrix=self.K, distCoeffs=None)
        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0, 
                    pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.001)
        
        # TODO: Currently when not enough points, assume no movement
        # Somehow when given only 5 points, the obtained E becomes not 3x3
        if E is None or E.shape != (3, 3):
            print("Warn: encounter invalid Essential matrix")
            return Pose()
        points, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)

        pose = Pose.from_t_R(t.flatten(), R)
        pose = pose.inverse()
        return pose