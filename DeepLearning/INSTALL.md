These installation instructions for Python TensorFlow are based off the documentation at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md, https://www.tensorflow.org/install/gpu, and https://www.tensorflow.org/install/pip.
To install the TensorFlow Python framework for model training:
Install CUDA Components:
  Install CUDA Version 10.0 (https://developer.nvidia.com/cuda-10.0-download-archive)
  Click the following link to the CUDnn download page (https://developer.nvidia.com/rdp/cudnn-download). You will need to create a developer account with your email address. After you do so, download the most recent avalible CUDnn for CUDA 10.0. Extract the resulting zip file to your install location, then move the cuda file from the main extracted file to the root of your installation. Finally, add the location of the cuda and cuda\bin directiory to your path. 
If you are on Windows, download the precompiled protobuf binary for your operating system from https://github.com/protocolbuffers/protobuf/releases/ and add the bin folder therin to your PATH.
If on Linux, run
  sudo apt update
  sudo apt install protobuf-compiler
Run the INSTALL script for your operating system (.bat for Windows and .sh for Linux).
If on Linux, add the following lines to your ~/.bashrc file:
  export PYTHONPATH=$PYTHONPATH;<Install Directory>/models;<Install Directory>/models/research/;<Install Directory>/models/research/slim;
  export PATH=$PATH;$PYTHONPATH
  export TENSORFLOW_INSTALL=<Absolute Path To Install Directory>/models
You may need reboot your system or run the previous line manually to add the given directories to your Python Path
If you are on Windows, add the following directories to path:
  <Install Directory>/models
  <Install Directory>/models/research/
  <Install Directory>/models/research/slim
Also, add these directories to enviornment variable PYTHONPATH (create it if it doesn't exist):
  <Install Directory>/models/research/
  <Install Directory>/models/research/slim

To test your installation, run the following terminal/command line command from your models/research direcitory:
  python object_detection/builders/model_builder_test.py
If the above line fails, follow the debugging steps in the provided error messages
You should now be able to import tensorflow in python. To learn more about TensorFlow, go to your models/research/object_detection directiory type "jupyter notebook object_detection_tutorial.ipynb". Refer to the TensorFlow API documentation (https://www.tensorflow.org/api_docs/python/tf) for further instructions.
To train an object detection model, refer to TRAINING.md in the Vision System folder

Troubleshooting:
  If the notebook fails to run, check that cudart64_100.dll and cudnn64_7.dll are in directories directly reffered to by your path variable.
