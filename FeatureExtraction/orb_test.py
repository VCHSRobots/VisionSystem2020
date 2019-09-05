"""
obj_test.py - A test of the object detectors from OpenCV
8/2/2019 Holiday Pettijohn
"""

import cv2
import time
import os

import numpy as np

train_img = "train.jpg"
test_img = "test.jpg"

detector = cv2.ORB_create()

def detectAndSave(detector, img, time=False):
  if time:
    start_time = time.time()
  kp, des = detector.detectAndCompute(img,None)
  disp = np.empty(shape=(1))
  disp = cv2.drawKeypoints(img, kp, outImage=disp, color=(0,0,255),flags=0)
  cv2.imwrite("file.png", disp)
  if time:
    return time.time()-start_time()

def drawFeatureMatch(detector, train_img, test_img, matcher=None, min_match_count=-1, save_file="save.png", perspective=None, good_matches=None, mask=None):
  #TODO: Remove perspective test arg
  if matcher == None:
    #6 points to flann index lsh algorithm
    index_params = dict(algorithm = 6,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
    search_params = {"checks": 50}
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
  train_kp, train_des = detector.detectAndCompute(train_img,None)
  test_kp, test_des = detector.detectAndCompute(test_img,None)
  if good_matches is None or mask is None:
    matches = matcher.knnMatch(train_des, test_des, k=2)
    #Checks for good matches with Lowe's ratio test
    good_matches = []
    #Checks for badly formatted matches. This may be an OpenCV bug
    for pair in matches:
      if len(pair) != 2:
        continue
      if pair[0].distance < .7*pair[1].distance:
        good_matches.append(pair[0])
  if len(good_matches)>=min_match_count:
    src_points = np.float32([train_kp[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([test_kp[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
    if perspective is None:
      perspective, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    else:
      _, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    mask = mask.ravel().tolist()
  else:
    print("Not enough good matches")
    return 0
  if perspective is not None:
    height, width = train_img.shape[:2]
    corners = np.float32([(0, 0), (0, height-1), (width-1, height-1), (width-1, 0)]).reshape(-1, 1, 2)
    transform = cv2.perspectiveTransform(corners, perspective)
    outlined_test_img = cv2.polylines(test_img, [np.int32(transform)],True,(0,0,255),3,cv2.LINE_AA)
    draw_params = dict(matchColor = (0,0,255), # draw matches in red color
                      singlePointColor = None,
                      matchesMask = mask, # draw only inliers
                      flags = 2)
    matched_img = cv2.drawMatches(train_img, train_kp, outlined_test_img, test_kp, good_matches, None, **draw_params)
    pos_filename = save_file
    ind = 0
    while os.path.isfile(pos_filename):
      pos_filename = save_file[:save_file.index(".")]+str(ind)+".png"
      ind += 1
    save_file = pos_filename
    print("Saving to {}".format(save_file))
    cv2.imwrite(save_file, matched_img)
    return perspective, good_matches, mask

#detectAndSave(detector, cv2.imread(train_img))
def test(perspective=None, good_matches=None, mask=None, save_file="save.png"):
  return drawFeatureMatch(detector, cv2.imread(train_img), cv2.imread(test_img), perspective=perspective, save_file=save_file, good_matches=good_matches, mask=mask)
test()
