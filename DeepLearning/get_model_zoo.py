"""
get_model_zoo.py - Gets the desired entries in the model zoo and extracts them
7/24/2019 Holiday Pettijohn
"""

import os
import glob
import tarfile
import six.moves.urllib as urllib

if not os.path.isdir("model_zoo"):
  os.mkdir("model_zoo")

url_base = 'http://download.tensorflow.org/models/object_detection/'
models = ["ssd_mobilenet_v1_coco_2017_11_17", "ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03",
          "ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18", "ssdlite_mobilenet_v2_coco_2018_05_09",
          "ssd_inception_v2_coco_2018_01_28", "faster_rcnn_inception_v2_coco_2018_01_28"]
opener = urllib.request.URLopener()

for model in models:
  model_path = "model_zoo/{}.tar.gz".format(model)
  opener = urllib.request.URLopener()
  print("Retrieving model from", url_base+model+".tar.gz")
  opener.retrieve(url_base+model+".tar.gz", model_path)
  print("Extracting tar file")
  tar = tarfile.open(model_path)
  print(model, "extracted to model zoo")
  tar.extractall(path="model_zoo/")
print("Model zoo successfully downloaded")
