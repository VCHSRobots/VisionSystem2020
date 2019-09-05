"""
detect.py - Uses ORB to detect objects
8/7/2019 Holiday Pettijohn
"""

import cv2
import time

import numpy as np

def detect(detector, matcher, reference_image, detect_image, min_match_count=7, image_points=None):
  ref_kp, ref_des = detector.detectAndCompute(reference_image, None)
  detect_kp, detect_des = detector.detectAndCompute(detect_image, None)
  matches = matcher.knnMatch(ref_des, detect_des, k=2)
  #Checks for good matches with Lowe's ratio test
  good_matches = []
  #Checks for badly formatted matches. This may be an OpenCV bug
  for pair in matches:
    if len(pair) != 2:
      continue
    if pair[0].distance < .7*pair[1].distance:
      good_matches.append(pair[0])
  if len(good_matches)>=min_match_count:
    src_points = np.float32([ref_kp[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([detect_kp[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    perspective, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    if perspective is not None:
      if image_points is None:
        height, width = reference_image.shape[:2]
        image_points = np.float32([(0, 0), (0, height-1), (width-1, height-1), (width-1, 0)]).reshape(-1, 1, 2)
      transformed_points = cv2.perspectiveTransform(image_points, perspective)
      keypoints = (ref_kp, detect_kp)
      return perspective, transformed_points, good_matches, mask, keypoints
    else:
      return None, None, None, None, None
  else:
    return None, None, None, None, None
  
draw_params = dict(matchColor = (0,0,255), # draw matches in red color
                  singlePointColor = None,
                  matchesMask = None, # draw only inliers
                  flags = 2)

def visualize(detector, matcher, reference_image, detect_image, min_match_count=7, image_points=None, draw_matches=False):
  perspective, transformed_points, good_matches, mask, keypoints = detect(detector, matcher, reference_image, detect_image, min_match_count, image_points)
  if perspective is None:
    return None, None, None, None, None, detect_image
  ref_kp, detect_kp = keypoints
  outlined_detect_img = cv2.polylines(detect_image, [np.int32(transformed_points)],True,(0,0,255),3,cv2.LINE_AA)
  if draw_matches:
    draw_params["matchesMask"] = mask
    matched_img = cv2.drawMatches(reference_image, ref_kp, outlined_detect_img, detect_kp, good_matches, None, **draw_params)
  else:
    matched_img = outlined_detect_img
  return perspective, transformed_points, good_matches, mask, keypoints, matched_img

def vidDetect(detector, matcher, reference_image, min_match_count=7, image_points=None, cam_num=0, draw_matches=False, window_name="Orb Detection"):
  cam = cv2.VideoCapture(cam_num)
  last_update_time = time.time()
  frames = 0
  while True:
    _, image = cam.read()
    if image is None:
      print("image is none")
      continue
    perspective, _, good_matches, _, _, vis_image = visualize(detector, matcher, reference_image, image, min_match_count, image_points, draw_matches)
    if perspective is None:
      vis_image = image
    cv2.imshow(window_name, vis_image)
    if cv2.waitKey(1) & 0xff == ord("q"):
      break
    t = time.time()
    frames += 1
    if t-last_update_time >= 10:
      fps = frames/10
      print("fps: {}".format(fps))
      frames = 0
      last_update_time = t
  cv2.destroyAllWindows()

orb = cv2.ORB_create()
#6 points to flann index lsh algorithm
index_params = dict(algorithm = 6,
                table_number = 6, # 12
                key_size = 12,     # 20
                multi_probe_level = 1) #2
#This number can be adjusted higher for accuracy and lower for speed
search_params = dict(checks=400)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
ref_img = cv2.imread("train.jpg")
vidDetect(orb, matcher, ref_img)
