"""
xml_to_tfr.py - Creates training record files from xml object location markups
7/27/2019 Holiday Pettijohn
"""

import argparse
import glob
import io
import pandas
import os

import tensorflow as tf
import xml.etree.ElementTree as ET

from collections import namedtuple
from object_detection.utils import dataset_util
from os import path
from PIL import Image

TrainData = namedtuple("TrainData", "filename size xmins xmaxes ymins ymaxes classes encoded_image int_labels image_format")
class_list = []

def makeRecords(directory, outpath):
  """
  Creates tfr records from xml files in a given directory
  Returns how many files were processed
  """
  absolute_path = absolutePath(outpath)
  writer = tf.io.TFRecordWriter(outpath)
  #Loads all data in directory
  data = loadData(directory)
  for file_data in data:
    record = makeExample(file_data)
    writer.write(record.SerializeToString())
  writer.close()
  absolute_path = absolute_path.replace("\\", "/")
  print("TFRecords successfully written to {}".format(absolute_path))
  #Returns number of xml/picture files
  return len(glob.glob("{}/*.xml".format(directory)))

def absolutePath(path):
  if not isAbsolutePath(path):
    path = os.path.join(os.getcwd(), path)
  return path

def isAbsolutePath(path):
  if ":/" in path or ":\\" in path:
    return True
  else:
    return False

def loadData(directory):
  """
  Makes a data frame with the xml data in the directory
  """
  data = []
  for filename in glob.glob("{}/*.xml".format(directory)):
    #Grabs data from each file in the directory
    filename = os.path.join(directory, filename)
    data.append(openXml(filename))
  return data

def checkForNewClasses(class_name):
  global class_list
  if not class_name in class_list:
    class_list.append(class_name)

def openXml(xml_filename):
  """
  Extracts xml data from a file and returns values needed to create training records
  """
  xmins, xmaxes, ymins, ymaxes, classes, int_labels = [], [], [], [], [], []
  directory = xml_filename[:xml_filename.rfind("\\")+1]
  tree = ET.parse(xml_filename)
  root = tree.getroot()
  filename = root.find("filename").text.encode("utf8")
  image_format = filename[filename.index(b".")+1:]
  location = os.path.join(directory, filename.decode())
  #Gets encoded image
  with tf.io.gfile.GFile(location, "rb") as f:
    encoded_image = f.read()
  size_root = root.find("size")
  width, height = int(size_root.find("width").text), int(size_root.find("height").text)
  size = width, height
  #Collects bounding box and class name data from file
  for obj in root.findall("object"):
    #Gets binding box container of bounding location
    box = obj.find("bndbox")
    #Normalizes object location data upon collection
    xmin, xmax = int(box.find("xmin").text)/width, int(box.find("xmax").text)/width
    ymin, ymax = int(box.find("ymin").text)/height, int(box.find("ymax").text)/height
    class_name = obj.find("name").text.encode("utf8")
    xmins.append(xmin)
    xmaxes.append(xmax)
    ymins.append(ymin)
    ymaxes.append(ymax)
    classes.append(class_name)
    checkForNewClasses(class_name)
    int_labels.append(class_list.index(class_name))
  train_data = TrainData(filename, size, xmins, xmaxes, ymins, ymaxes, classes, encoded_image, int_labels, image_format)
  return train_data

def makeExample(train_data):
  """
  Creates a TensorFlow training example from data
  """
  #Gets width and height
  width, height = train_data.size
  #Creates example
  example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(train_data.filename),
        'image/source_id': dataset_util.bytes_feature(train_data.filename),
        'image/encoded': dataset_util.bytes_feature(train_data.encoded_image),
        'image/format': dataset_util.bytes_feature(train_data.image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(train_data.xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(train_data.xmaxes),
        'image/object/bbox/ymin': dataset_util.float_list_feature(train_data.ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(train_data.ymaxes),
        'image/object/class/text': dataset_util.bytes_list_feature(train_data.classes),
        'image/object/class/label': dataset_util.int64_list_feature(train_data.int_labels),
    }))
  return example

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Generates protobuf training files for object detection model training")
  parser.add_argument("directory")
  parser.add_argument("--out_dir", "-out")
  args = parser.parse_args()
  if args.out_dir:
    makeRecords(args.directory, args.out_dir)
  else:
    makeRecords(args.directory, "record.record")
