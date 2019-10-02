"""
stereo_detect.py - Detects object locations using stereo vision
9/24/2019 Holiday Pettijohn
"""

import cv2
import tensorflow as tf
from math import radians, tan
import numpy as np

import detect

import triangulation

def runSystem(left_cam_num, right_cam_num, model_location,
              camera_distance_x, camera_distance_y,
              left_cam_angle_x, right_cam_angle_x,
              left_cam_angle_y, right_cam_angle_y,
              left_vision_range_x, right_vision_range_x,
              left_vision_range_y, right_vision_range_y,
              detection_threshold = .8, display_detection=True):
  """
  left_cam_num, right_cam_num - Numbers of the two cameras to be used for detection
  model_location - String path to the directory where the model graph is stored
  """
  left_cam = cv2.VideoCapture(left_cam_num)
  right_cam = cv2.VideoCapture(right_cam_num)
  #Loads model, its category index, and outlines of its inputs and outputs
  model, category_index, image_tensor, tensor_dict = detect.loadModelDataFromDir(model_location)
  #Create session to be reused
  sess = tf.Session(graph=model)
  while True:
    left_ret, left_img = left_cam.read()
    right_ret, right_img = right_cam.read()
    if left_ret and right_ret:
      #If both cameras read successfully, detect objects and process the data
      img_batch = np.array([left_img, right_img])
      inference = detect.sdetectBatch(sess, image_tensor,
                                      tensor_dict, img_batch)
      best_detections = filterBestData(inference, detection_threshold=detection_threshold)
      stereo_results = stereoVision(best_detections, camera_distance_x, camera_distance_y,
                                    left_cam_angle_x, right_cam_angle_x, left_cam_angle_y,
                                    right_cam_angle_y, left_vision_range_x, right_vision_range_x,
                                    left_vision_range_y, right_vision_range_y)

def filterBestData(inference, detection_threshold = .8):
  """
  Sorts through inference data and finds the best detection for each type of target for each camera
  """
  detection_boxes = {}
  for cam_num, out_scores in enumerate(inference["detection_scores"]):
    best_scores = {}
    detection_boxes[cam_num] = {}
    for ind, score in enumerate(out_scores):
      class_name = inference["detection_classes"][cam_num][ind]
      if score >= detection_threshold:
        if class_name not in best_scores:
          best_scores[class_name] = score
          detection_boxes[cam_num][class_name] = inference["detection_boxes"][cam_num][ind]
        elif score > best_scores[class_name]:
          best_scores[class_name] = score
          detection_boxes[cam_num][class_name] = inference["detection_boxes"][cam_num][ind]
  return detection_boxes


def stereoVision(detection_boxes, camera_distance_x,
                 camera_distance_y, left_cam_angle_x, right_cam_angle_x,
                 left_cam_angle_y, right_cam_angle_y,
                 left_vision_range_x, right_vision_range_x,
                 left_vision_range_y, right_vision_range_y):
  """
  Detects objects with two cameras for stereo vision
  Arguments:
    detection_boxes - Detection boxes for each camera. Assumes this represents the results of at 
                      least two cameras
    camera_distance_x, camera_distance_y - Horizontal or vertical distance between the two cameras
                                           The unit this is given in is the unit in which results
                                           will be returned
    left_cam_angle, right_cam_angle - Angle at which the cameras are situated. Measured
                                      north of east in degrees
    left_vision_range_x, right_vision_range_x - Horizontal range of vision of each
                                                        camera, measured in degrees
    left_vision_range_y, right_vision_range_y - Vertical range of vision of each
                                                        camera, measured in degrees
  Returns:
    object_displacements - Dictionary of x-y-z target displacements from each detected object
                           relative to the origin at the leftmost camera
    camera_depths - Depth (z displacement) relative to each camera
  """
  #Checks which classes were detected by both cameras
  left_classes = set(detection_boxes[0].keys())
  right_classes = set(detection_boxes[1].keys())
  overlap = left_classes.intersection(right_classes)
  object_displacements = {}
  camera_depths = {}
  #Uses stereo vision to detect the object's depth location
  for target_name in list(overlap):
    ax, ay = getCenterPoint(detection_boxes[0][target_name])
    bx, by = getCenterPoint(detection_boxes[1][target_name])
    #Finds the x-z displacement between the left camera and target
    base_triangulation = triangulation.triangulate(ax, bx,
                                                   radians(left_cam_angle_x), radians(right_cam_angle_x),
                                                   radians(left_vision_range_x), radians(right_vision_range_x),
                                                   camera_distance_x)
    object_angle_1, object_angle_2, target_distance_1, target_distance_2, = base_triangulation
    left_x_displacement, left_z_displacement = triangulation.findTargetDistances(object_angle_1, radians(left_cam_angle_x), target_distance_1)
    right_x_displacement, right_z_displacement = triangulation.findTargetDistances(object_angle_2, radians(right_cam_angle_x), target_distance_2)
    left_depth, right_depth = triangulation.findCameraDepths(radians(left_vision_range_x), radians(right_vision_range_x),
                                                             object_angle_1, object_angle_2,
                                                             target_distance_1, target_distance_2)
    #Finds the y displacement between the cameras and target
    #Doesn't use triangulation... yet
    #Will not work if there is any y displacement between cameras
    #TODO: This needs testing
    y_angle = (ay-.5)*radians(left_vision_range_y)
    y_displacement = tan(y_angle)*left_depth
    object_displacements[target_name] = ((left_x_displacement, y_displacement, left_z_displacement),
                                         (right_x_displacement, y_displacement, right_z_displacement))
    camera_depths[target_name] = left_depth, right_depth
  return object_displacements, camera_depths

def getCenterPoint(detection_box):
  """
  Finds the center point of a detection box given the edge points of the box
  """
  #Compensates for the OpenCV defenition of 'positive y is down' in the second calculation
  x, y = (detection_box[0]+detection_box[2])/2, 1-((detection_box[1]+detection_box[3])/2)
  return x, y
