"""
triangulation.py - Runs object detection on two cameras of know distance apart for depth perception
8/1/2019 Holiday Pettijohn
"""

from math import sin, cos, pi

def triangulate(detection_location_1, detection_location_2,
                camera_1_angle, camera_2_angle, 
                vision_range_1, vision_range_2,
                camera_seperation_distance):
  """
  Solves for the distances and angles contained in the triangle
  c1/t/c2 as documented under Triangulation directory
  Angles will be returned in radians
  Distance will be returned in the unit the camera distances are given in
  """
  #Angles of vision relative to the cameras
  sight_angle_1 = vision_range_1*(1-detection_location_1)
  sight_angle_2 = vision_range_2*detection_location_2
  #Angle of object relative to the line connecting the two cameras
  vision_angle_1 = sight_angle_1+camera_1_angle
  vision_angle_2 = sight_angle_2+camera_2_angle
  target_angle = pi-(vision_angle_1+vision_angle_2)
  camera_target_distance_1 = camera_seperation_distance*sin(vision_angle_2)/sin(target_angle)
  camera_target_distance_2 = camera_seperation_distance*sin(vision_angle_1)/sin(target_angle)
  return sight_angle_1, sight_angle_2, camera_target_distance_1, camera_target_distance_2

def findTargetDistances(sight_angle, camera_angle, target_distance):
  """
  Finds the x/z or y/z distances from the camera to the target
  Documented under Triangulation directory
  """
  sight_angle_prime = pi/2-(sight_angle+camera_angle)
  z_distance = target_distance*cos(sight_angle_prime)
  x_y_distance = target_distance*sin(sight_angle_prime)
  return z_distance, x_y_distance

def findRelativeCameraDepths(vision_range_1, vision_range_2,
                             sight_angle_1, sight_angle_2,
                             target_distance_camera_1, target_distance_camera_2):
  #Keep these angles positive to keep returned distances positive
  center_of_vision_angle_1 = abs(sight_angle_1-vision_range_1/2)
  center_of_vision_angle_2 = abs(sight_angle_2-vision_range_2/2)
  depth_1 = target_distance_camera_1*cos(center_of_vision_angle_1)
  depth_2 = target_distance_camera_2*cos(center_of_vision_angle_2)
  return depth_1, depth_2
