"""Utility functions for image processing
"""
import cv2


def read_rgb(path):
    """Read image in RGB
    """
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im