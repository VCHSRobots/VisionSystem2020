"""
train.py - Trains a model with a dataset
7/24/2019 Holiday Pettijohn
"""

import argparse
import glob
import json
import subprocess
import os

import tensorflow as tf

import xml_to_tfr

def trainModel(model_location, train_dir="", test_dir="",
               config_location="default.config", out_dir="detector_training"):
  """
  Trains a model with the given directories with the train and test data
  Will continue training the provided model
  Arguments:
    model_location: Path to the directory of the model checkpoint to start training from
                    Should be absolute
    config_location: Path to a formatted config file to be referenced
                     The system includes a default config file if none is provided
    train_dir: Directory with training images and xml files
    test_dir: Directory with testing images and xml files
    out_dir: The directory where record and pbtext files will be written
  """
  #Makes output directory if it doesn't exist
  #Doesn't clobber existing directories
  num_dirs_with_name = 0
  check_dir = out_dir
  while os.path.exists(check_dir):
    check_dir = out_dir+str(num_dirs_with_name)
    num_dirs_with_name += 1
  out_dir = check_dir
  os.makedirs(out_dir)
  if not xml_to_tfr.isAbsolutePath(config_location):
    config_location = xml_to_tfr.absolutePath(config_location).replace("/", "\\")
  if not xml_to_tfr.isAbsolutePath(out_dir):
    out_dir = xml_to_tfr.absolutePath(out_dir).replace("/", "\\")
  #Converts xml files into training record files
  xml_to_tfr.makeRecords(train_dir, os.path.join(out_dir, "train.record"))
  num_test_images = xml_to_tfr.makeRecords(test_dir, os.path.join(out_dir, "test.record"))
  #Makes pbtext file for classes which were found in the xml files
  classes = xml_to_tfr.class_list
  makePbtext(classes, out=os.path.join(out_dir, "label_map.pbtxt"))
  #Modifies copy of config data as needed
  writeConfigData(config_location, model_location, len(classes), out_dir, num_test_images)
  out_config_location = xml_to_tfr.absolutePath(os.path.join(out_dir, "model.config")).replace("\\", "/")
  print("To view model training diagnostics, open a command promps as admin and type 'tensorboard --logdir=<PATH_TO_OUTPUT_DIRECTORY>'\nTraining preformance is graphed under the TotalLoss tab.")
  print("Press Enter to stop training and save inference graph from last checkpoint")
  #Trains the model
  first_temp_state = glob.glob(settings["train_data_location"]+"/*")
  train_location = os.path.join(settings["install"], "models\\research\\object_detection\\model_main.py")
  training = subprocess.Popen(["py", train_location, "--logtostderr", "--train_dir={}".format(out_dir), "--pipeline_config_path={}".format(out_config_location)])
  input()
  training.terminate()
  print("""If a traceback does not appear above, the inference graph wrote successfully. 
In case of failure, recover training progress by going to User/<Username>/AppData/Local/Temp and find the most recently modified directory.
Copy this directory to a convienent location and rename it if desired. Then, run the following command from your TensorFlow install directory:
py {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_prefix=<Checkpoint Folder>/model.ckpt-<Highest Number After model.ckpt> --output_directory={}""".format(os.path.join(settings["install"], "models\\research\\object_detection\\export_inference_graph.py"), config_location, out_dir))
  created_dirs = []
  created_objs = list(set(glob.glob(settings["train_data_location"]+"/*")).difference(first_temp_state))
  created_objs = [obj.replace("\\", "/") for obj in created_objs]
  for obj in created_objs:
    obj_name = obj[obj.rfind("/")+1:]
    if os.path.isdir(obj) and obj_name.startswith("tmp"):
      created_dirs.append(obj)
  training_data_location = created_dirs[-1]
  freezeInferenceGraph(out_config_location, out_dir, training_data_location)
  input("Press Enter to exit")

def makePbtext(classes, out):
  """
  Makes the pbtext file which lists the object detection classes and their indexes
  """
  text = ""
  for ind, object_class in enumerate(classes):
    inner_text = "\n\tid: {}\n\tname: '{}'\n".format(ind+1, object_class.decode())
    text += "item {" + inner_text + "}\n"
  f = open(out, "w")
  f.write(text)
  f.close()

def writeConfigData(config_location, model_location, num_classes, out_dir, num_test_images):
  """
  Writes config data using a file as reference
  """
  f = open(config_location)
  config = f.readlines()
  f.close()
  config = fillField(config, "num_classes:", num_classes)
  #Adds the model.ckpt to the filepath
  #This allows the training program to pick up several files with model.ckpt in their extension
  model_location = os.path.join(model_location, "model.ckpt")
  #Makes sure file paths are written with /, not \
  config = fillField(config, "fine_tune_checkpoint:", model_location.replace("\\", '/'))
  #Finds the section which concerns training records
  train_start, train_stop = findGroup(config, "train_input_reader:")
  config = fillField(config, "input_path:", os.path.join(out_dir, "train.record").replace("\\", '/'), start=train_start, stop=train_stop)
  config = fillField(config, "label_map_path:", os.path.join(out_dir, "label_map.pbtxt").replace("\\", '/'), start=train_start, stop=train_stop)
  config = fillField(config, "num_examples:", num_test_images)
  #Finds the section which concerns test images
  eval_start, eval_stop = findGroup(config, "eval_input_reader:")
  config = fillField(config, "input_path:", os.path.join(out_dir, "test.record").replace("\\", '/'), start=eval_start, stop=eval_stop)
  config = fillField(config, "label_map_path:", os.path.join(out_dir, "label_map.pbtxt").replace("\\", '/'), start=eval_start, stop=eval_stop)
  text = ""
  for line in config:
    text += line
  f = open(os.path.join(out_dir, "model.config"), "w")
  f.write(text)
  f.close()

def fillField(lines, field, value, start = 0, stop = None, step = 1):
  """
  Searches through lines for a field and fills in that field with the given value
  Preserves indentation levels
  """
  index = -1
  for ind, line in enumerate(lines[start:stop:step]):
    if field in line:
      index = (ind*step)+start
  if index is not -1:
    spaces = countIndent(lines[index])
    #Puts quotes around string variables
    if type(value) == str:
      value = '"{}"'.format(value)
    lines[index] = "{}{} {}\n".format(" "*spaces, field, value)
  else:
    raise ValueError("File does not have field {}".format(field[:-1]))
  return lines

def countIndent(string):
  """
  Finds how many spaces are in a string before alpanumeric characters appear
  """
  spaces = 0
  for c in string:
    if c == " ":
      spaces += 1
    else:
      break
  return spaces

def findGroup(lines, name):
  """
  Finds the lines contained in the brackets of a given group's name
  """
  name_found = False
  start_ind, stop_ind = -1, -1
  depth = 0
  for ind, line in enumerate(lines):
    if name in line:
      name_found = True
    if name_found:
      if "{" in line:
        if depth == 0:
          start_ind = ind
          depth = 1
        else:
          depth += 1
      if "}" in line:
        if depth == 1:
          stop_ind = ind
          break
        else:
          depth -= 1
  return start_ind, stop_ind

def freezeInferenceGraph(config_location, out_dir, training_data_location):
  checkpoint_file = glob.glob("{}/model.*.meta".format(training_data_location))[-1].replace("\\", "/")
  checkpoint_file = checkpoint_file[checkpoint_file.rfind("/")+1:]
  last_step = checkpoint_file.replace("model.ckpt-", "").replace(".meta", "")
  print("Saving inference graph at step {} from data at {}".format(last_step, checkpoint_file))
  subprocess.call(["py", os.path.join(settings["install"], "models\\research\\object_detection\\export_inference_graph.py"), "--input_type=image_tensor", "--pipeline_config_path={}".format(config_location), "--trained_checkpoint_prefix={}/model.ckpt-{}".format(training_data_location, last_step), "--output_directory={}".format(out_dir)])

def readSettingsFile():
  global settings
  f = open("settings.json")
  settings = json.load(f)
  f.close()

settings = {}

readSettingsFile()

if __name__ == "__main__":
  #Command line argument handling
  parser = argparse.ArgumentParser(description="Trains an object detection model")
  parser.add_argument("--model_name", type=str, help="path to a model to continue training")
  parser.add_argument("--model_location", type=str, help="path to a model to continue training")
  parser.add_argument("--train_dir", type=str, help="path to a model to continue training")
  parser.add_argument("--test_dir", type=str, help="path to training images and xml files")
  parser.add_argument("-out", "--out_dir", type=str, help="path to testing images and xml files")
  parser.add_argument("-config", "--config_location", type=str, help="path to a config file to base the model on")
  args = parser.parse_args()
  model_name = args.model_name if args.model_name else settings["model_name"] if "model_name" in settings else None
  if model_name:
    model_location = os.path.join("model_zoo", model_name)
  else:
    model_location = args.model_location if args.model_location else settings["model_location"]
  train_dir = args.train_dir if args.train_dir else settings["train_dir"]
  test_dir = args.test_dir if args.test_dir else settings["test_dir"]
  out_dir = args.out_dir if args.out_dir else settings["out_dir"]
  config_location = args.config_location if args.config_location else settings["config_location"]
  trainModel(model_location=model_location, train_dir=train_dir, test_dir=test_dir, config_location=config_location, out_dir=out_dir)
