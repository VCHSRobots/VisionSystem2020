"""
filter.py - Finds the best ORB features in a series of training images
9/4/2019 Holiday Pettijohn
"""

import cv2
import time

import numpy as np

def findBestFeatures(images, detections_per_image=4, min_quality_count=None):
  """
  Returns the best features found in the given series of images
  Defines 'best' features as those which occur the most in the set of images
  """
