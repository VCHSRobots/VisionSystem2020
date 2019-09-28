"""
vid_detect.py - Runs a video feed with running object detection
8/1/2019 Holiday Pettijohn
"""

import cv2

import tensorflow as tf

import time

import detect

import numpy as np

from math import atan2, tan, degrees, radians

def detectVideoStream(model_location, cam_num=0, detection_threshold=.4, vision_angle_range = (radians(100), radians(100))):
  #TODO: Turn this into an orientation system
  frames = 0
  time_passed = 0
  last_time = time.time()
  camera = cv2.VideoCapture(cam_num)
  model = detect.loadModel("{}/frozen_inference_graph.pb".format(model_location))
  sess = tf.Session(graph=model)
  image_tensor, tensor_dict = detect.getDetectorInput(model)
  category_index = detect.getVisualData("{}/label_map.pbtxt".format(model_location))
  detection_boxes = {}
  cv2.namedWindow("Detection Stream")
  _, image = camera.read()
  height, width, _ = image.shape
  while True:
    ret, image = camera.read()
    if ret:
      detection = detect.sdetect(sess, image_tensor, tensor_dict, image)
      for y, out_scores in enumerate(detection["detection_scores"]):
        for x, score in enumerate(out_scores):
          if score >= detection_threshold:
            if detection["detection_classes"][y][x] not in detection_boxes:
              detection_boxes[detection["detection_classes"][y][x]] = []
            detection_boxes[detection["detection_classes"][y][x]].append(detection["detection_boxes"][y][x])
      #visualization = detect.visualize(image, detection["detection_boxes"], detection["detection_scores"],
      #                                detection["detection_classes"], category_index)
      for object_num in detection_boxes.keys():
        for detection_box in detection_boxes[object_num]:
          box_width, box_height = detection_box[0]-detection_box[2], detection_box[1]-detection_box[3]
          #1' = .22 width & height
          #Distance is based solely off height for now,
          #Angles are measured east and north of the axis extruding straight from the camera's vision
          inches_dist = -3.1/box_width
          norm_detection_center = (detection_box[0]+detection_box[2])/2, (detection_box[1]+detection_box[3])/2
          y, x = 1-norm_detection_center[0], norm_detection_center[1]
          angles = getTargetAngles(inches_dist, x, y, vision_angle_range)
          print(round(x, 4), round(y, 4), round(inches_dist, 4), round(degrees(angles[0]), 4), round(degrees(angles[1]), 4))
          detection_center = int(norm_detection_center[0]*height), int(norm_detection_center[1]*width)
          lines = (((detection_center[1]+8, detection_center[0]), (detection_center[1]-8, detection_center[0])),
                  (((detection_center[1], detection_center[0]+8), (detection_center[1], detection_center[0]-8))))
          for line in lines:
            cv2.line(image, line[0], line[1], (0, 0, 255))
      cv2.imshow("Detection Stream", image)
      frames += 1
      ctime = time.time()
      time_passed += ctime-last_time
      last_time = ctime
      if time_passed >= 10:
        print("FPS: {}".format(frames/10))
        frames = 0
        time_passed = 0
      detection_boxes = {}
      if cv2.waitKey(1) & 0xff == ord("q"):
        break
  cv2.destroyAllWindows()

def getTargetAngles(distance, x_pos, y_pos, vision_angle_range, origin = (.5, .5)):
  """
  Gets the target's angles relative to the camera's origin
  Returns in x, y format
  """
  x_range = 2*distance*tan(vision_angle_range[0]/2)
  y_range = 2*distance*tan(vision_angle_range[1]/2)
  origin_distance_x = x_pos-origin[0]
  origin_distance_y = y_pos-origin[1]
  unit_x_dist = origin_distance_x*x_range
  unit_y_dist = origin_distance_y*y_range
  angles = atan2(unit_x_dist, distance), atan2(unit_y_dist, distance)
  return angles

if __name__ == "__main__":
  detectVideoStream("detectors/faster_rcnn_pin", 0)
