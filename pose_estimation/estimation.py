"""Estimate relative pose between two images
"""
import cv2
import numpy as np
import sys
import copy
sys.path.append("../")
from dataset_util.pose import Pose
import itertools


def fix_scale(pose1, pose2):
    """Fix the scale of pose 2 translation such that it has same norm as pose 1
    """
    norm1 = np.linalg.norm(pose1.t)
    norm2 = np.linalg.norm(pose2.t)
    fixed_pose = copy.copy(pose2)
    if not np.isclose(norm2, 0):
        fixed_pose.t *= norm1/norm2
    return fixed_pose


def compute_t_error(pose1, pose2):
    """
    Compute the error in translation
    """
    dt = pose1.t - pose2.t
    return np.linalg.norm(dt)


def compute_traj_error(traj1, traj2):
    errors = [compute_t_error(pose1, pose2) for pose1, pose2 in
                zip(traj1.poses, traj2.poses)]
    return np.average(errors)


class FramePair:
    """Result of processing two frames
    """
    def __init__(self, pts1, pts2, pose):
        """
        Args:
            pts1: Nx2 numpy array
            pts2: Nx2 numpy array
            pose: Pose object, relative pose between two frames
        """
        self.idx1 = None
        self.idx2 = None
        self.pts1 = pts1
        self.pts2 = pts2
        self.pose = pose


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
        self.frame_pairs = []
        self.last_img = None
        self.last_idx = None

    def process(self, im, idx, estimate_scale=True):
        if self.last_img is None:
            self.last_img = im
            self.last_idx = idx
            return

        # Compute frame
        im1 = self.last_img
        last_idx = self.last_idx
        im2 = im
        self.last_img = im
        self.last_idx = idx
        frame_pair = self.estimate(im1, im2)
        frame_pair.idx1 = last_idx
        frame_pair.idx2 = idx

        # Do not estimate relative scale for first pair
        if len(self.frame_pairs) == 0:
            self.frame_pairs.append(frame_pair)
            return

        if not estimate_scale:
            self.frame_pairs.append(frame_pair)
            return

        # Estimate relative scale
        # Get point matches across three frames
        last_pair = self.frame_pairs[-1]
        pts_matches = [[] for _ in range(3)]
        for pt1, pt2 in zip(last_pair.pts1, last_pair.pts2):
            for pt3, pt4 in zip(frame_pair.pts1, frame_pair.pts2):
                if np.allclose(pt2, pt3):
                    pts_matches[0].append(pt1)
                    pts_matches[1].append(pt2)
                    pts_matches[2].append(pt4)
                    continue
        print("idx: {}".format(idx))
        assert(len(pts_matches[0]) >= 2)
        pts_matches = [np.asarray(pts_array) for pts_array in pts_matches]

        # Construct projection matrices
        PA = np.zeros(shape=(3, 4))
        PA[:3, :3] = np.identity(3)

        pose_b = last_pair.pose.inverse()
        R_b, t_b = pose_b.R, pose_b.t.reshape(-1, 1)
        PB = np.concatenate((R_b, t_b), axis=1)

        pose_c = (last_pair.pose * frame_pair.pose).inverse()
        R_c, t_c = pose_c.R, pose_c.t.reshape(-1, 1)
        PC = np.concatenate((R_c, t_c), axis=1)

        # Triangulate 3D point positions
        pts_3d = []
        pts_3d.append(cv2.triangulatePoints(PA,PB, pts_matches[0].T, pts_matches[1].T).T)
        pts_3d.append(cv2.triangulatePoints(PB,PC, pts_matches[1].T, pts_matches[2].T).T)

        # Homogenenous to inhomogenous coordinates
        pts_3d[0] = (pts_3d[0]/pts_3d[0][:, 3:])[:, :3]
        pts_3d[1] = (pts_3d[1]/pts_3d[1][:, 3:])[:, :3]

        scales = []
        for i, j in itertools.combinations(range(len(pts_3d[0])), 2):
            # Compute relative scale
            pts_a1, pts_a2 = pts_3d[0][i], pts_3d[0][j]
            pts_b1, pts_b2 = pts_3d[1][i], pts_3d[1][j]

            dist1 = np.linalg.norm(pts_a1 - pts_a2)
            dist2 = np.linalg.norm(pts_b1 - pts_b2)

            if np.isclose(dist1, 0) or np.isclose(dist2, 0):
                continue
            r = dist1 / dist2
            if not np.isnan(r):
                scales.append(r)

        scale = np.median(scales)
        print("scale: {}".format(scale))
        frame_pair.pose.t *= scale
        self.frame_pairs.append(frame_pair)

    def estimate(self, im1, im2):
        """Estimate the relative pose from 2 images
        """
        E, pts1_norm, pts2_norm = self.compute_essential(im1, im2)

        # TODO: Currently when not enough points, assume no movement
        # Somehow when given only 5 points, the obtained E becomes not 3x3
        if E is None or E.shape != (3, 3):
            print("Warn: encounter invalid Essential matrix")
            return Pose()

        R, t = self.recover_pose(E, pts1_norm, pts2_norm)
        # points, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm)

        pose = Pose.from_t_R(t.flatten(), R)
        pose = pose.inverse()
        return FramePair(pts1_norm, pts2_norm, pose)

    def compute_essential(self, im1, im2):
        """Compute the essential matrix, and corresponding points
        Return:
            Essential matrix,
        """
        detector = self.detector

        kp1 = self.find_kp(im1)
        kp2 = self.find_kp(im2)
        kp1, des1 = detector.compute(im1, kp1)
        kp2, des2 = detector.compute(im2, kp2)

        good_matches = self.match_kp(des1, des2, self.method)
        pts1, pts2 = self.get_matched_pts(kp1, kp2, good_matches)

        # Compute relative pose
        pts1_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1),
                        cameraMatrix=self.K, distCoeffs=None)
        pts2_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1),
                        cameraMatrix=self.K, distCoeffs=None)
        E, mask = cv2.findEssentialMat(pts1_norm, pts2_norm, focal=1.0,
                    pp=(0, 0), method=cv2.RANSAC, prob=0.999, threshold=0.001)

        pts1_norm, pts2_norm = self.masked_points(pts1_norm, pts2_norm, mask)
        return E, pts1_norm, pts2_norm

    @staticmethod
    def masked_points(pts1, pts2, mask):
        """Get points that are masekd true
        Return:
            true pts1, true pts2
        """
        if mask is None:
            return pts1, pts2
        mask = mask.astype(np.bool)
        return pts1[mask], pts2[mask]

    def find_kp(self, im):
        """Detect keypoints
        """
        kp = self.detector.detect(im, None)
        return kp

    @staticmethod
    def match_kp(des1, des2, method):
        """Match keypoints
        Return:
            list of good matches
        """
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        if method == "SIFT":
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.knnMatch(des1, des2, k=2)
            matches_mask = [[0,0] for i in range(len(matches))]

            good_matches = []
            for i,(m, n) in enumerate(matches):
                if m.distance < 0.8*n.distance:
                    matches_mask[i]=[1,0]
                    good_matches.append(m)
        elif method == "ORB":
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            good_matches = matcher.match(des1, des2)

        return good_matches

    @staticmethod
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

    def recover_pose(self, E, pts1_norm, pts2_norm):
        """Find the proper rotation and translation from
            the Essential matrix, by cheirality constraint

        Args:
            E: Essential matrix, 3x3 numpy array
            pts1_norm: Nx1x2 numpy array of image coordinates in image 1,
                        already normalized with the camera matrix
            pts2_norm: same as above, except for image 2
            mask: Nx1 numpy array indicating which points are inliers
        Return:
            3x3 rotation matrix, 3x1 translation
        """

        U, _, Vh = np.linalg.svd(E)
        # Correct for determinants
        if np.linalg.det(U) < 0:
            U = -U
        if np.linalg.det(Vh) < 0:
            Vh = -Vh

        # Obtain candidate rotation and translation
        D = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 1]])
        Ra = np.matmul(U, np.matmul(D, Vh))
        Rb = np.matmul(U, np.matmul(D.T, Vh))
        tu = U[:, 2].reshape(-1, 1)

        # Construct projection matrices
        PA = np.concatenate((Ra, tu), axis=1)
        PI = np.zeros(shape=(3, 4))
        PI[:3, :3] = np.identity(3)

        # Twist transformation
        Ht = np.diag([1, 1, 1, -1]).astype(np.float)
        Ht[3, :3] = -2 * Vh[2, :]

        votes = [0, 0, 0, 0]
        for pt1, pt2 in zip(pts1_norm, pts2_norm):
            q = cv2.triangulatePoints(PI, PA, pt1.T, pt2.T)
            c1 = q[2] * q[3]
            c2 = np.matmul(PA, q)[2] * q[3]
            assert(c1 != 0 and c2 != 0)
            if c1 > 0 and c2 > 0:
                votes[0] += 1
                continue
            if c1 < 0 and c2 < 0:
                votes[1] += 1
                continue
            qc = np.matmul(Ht,  q)
            if q[2] * qc[3] > 0:
                votes[2] += 1
                continue
            else:
                votes[3] += 1

        idx = votes.index(max(votes))
        candidates = [(Ra, tu), (Ra, -tu), (Rb, tu), (Rb, -tu)]
        return candidates[idx]
