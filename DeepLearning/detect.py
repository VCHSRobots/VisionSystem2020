"""
detect.py - Uses a trained model to detect objects given an image
7/24/2019 Holiday Pettijohn
"""

import tensorflow as tf
import numpy as np

from utils import visualization_utils as vis_util
from utils import label_map_util

keys = ["image_tensor", "detection_boxes", "detection_scores",
        "detection_classes", "num_detections", "detection_masks"]

#Note to self - You can reuse the same session and run it with different inputs
def detect(model, image):
  """
  Detects objects in an image with the given model
  """
  tensor_dict = {}
  with tf.Session(graph=model) as sess:
    image_tensor, tensor_dict = getDetectorInput(model)
    output = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
  return output

def detectBatch(model, images):
  """
  Detects objects in multiple images with the given model
  """
  tensor_dict = {}
  with tf.Session(graph=model) as sess:
    image_tensor, tensor_dict = getDetectorInput(model)
    image_tensor = tensor_dict.pop("image_tensor")
    output = sess.run(tensor_dict, feed_dict={image_tensor: images})
  return output

def sdetect(sess, image_tensor, tensor_dict, image):
  """
  Detects objects on an already-configured session and tensor dict
  """
  output = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, axis=0)})
  return output

def sdetectBatch(sess, image_tensor, tensor_dict, images):
  """
  Detects objects on an already-configured session and tensor dict
  """
  output = sess.run(tensor_dict, feed_dict={image_tensor: images})
  return output

def visualize(image, boxes, scores,
              classes, category_index,
              use_normalized_coordinates=True, line_thickness=4):
  img = vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index=category_index,
    use_normalized_coordinates=use_normalized_coordinates,
    line_thickness=line_thickness
  )
  return img

def getDetectorInput(model):
  tensor_dict = {}
  operations = model.get_operations()
  all_tensor_names = {output.name for operation in operations for output in operation.outputs}
  for key in keys:
    tensor_name = "{}:0".format(key)
    if tensor_name in all_tensor_names:
      tensor_dict[key] = model.get_tensor_by_name(tensor_name)
  image_tensor = tensor_dict.pop("image_tensor")
  return image_tensor, tensor_dict

def getVisualData(label_map_path):
  f = open(label_map_path)
  text = f.read()
  f.close()
  num_classes = text.count("item {\n\t")
  label_map = label_map_util.load_labelmap(label_map_path)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)
  return category_index

def loadModel(model_location):
  """
  Opens a model from a filepath
  """
  model = tf.Graph()
  with model.as_default():
    graph_def = tf.GraphDef()
    with tf.io.gfile.GFile(model_location, "rb") as f:
      serialized_graph = f.read()
      graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(graph_def, name="")
  return model
