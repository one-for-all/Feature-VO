"""Implement representation of camera poses
"""
import numpy as np
from scipy.spatial.transform import Rotation 

class Pose(object):
    def __init__(self):
        self.T = np.identity(4)
    
    @staticmethod
    def from_t_q(t, q):
        """Construct pose from translation and quaternion
        """
        t = np.asarray(t)
        rotation = Rotation.from_quat(q)
        p = Pose()
        p.T[:3, 3] = t
        p.T[:3, :3] = rotation.as_dcm()
        return p

    def to_t_q(self):
        q = Rotation.from_dcm(self.R).as_quat()
        t = self.t
        return t, q

    @staticmethod
    def from_t_R(t, R):
        """Construct pose from translation and rotation matrix
        """
        p = Pose()
        p.T[:3, 3] = t
        p.T[:3, :3] = R
        return p 
    
    @staticmethod
    def from_T(T):
        """Construct pose directly from transform matrix
        """
        p = Pose()
        p.T = T
        return p

    def __str__(self):
        return str(self.T)
    
    def __repr__(self):
        return str(self.T)
    
    def inverse(self):
        """Get inverse of transform
        """
        T_inv = np.linalg.inv(self.T)
        pose = Pose.from_T(T_inv)
        return pose

    def __mul__(self, other):
        """Multiply poses as matrices
        """
        T = np.matmul(self.T, other.T)
        pose = Pose.from_T(T)
        return pose
    
    @property
    def t(self):
        """Get translation
        """
        return self.T[:3, 3]

    @t.setter
    def t(self, new_t):
        self.T[:3, 3] = new_t
    
    @property
    def R(self):
        """Get rotation matrix
        """
        return self.T[:3, :3]

    @R.setter
    def R(self, new_R):
        self.T[:3, :3] = new_R
