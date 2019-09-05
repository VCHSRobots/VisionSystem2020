Setup:
To train an object detection model, copy the default_settings.json file to settings.json
The training script will default to detecting settings.json, but will fall back to default settings if it does not exist
Make sure that tensorflow_install value points to the models folder installed by the setup script. If on Windows, be sure it uses backslashes ('\') not foward slashes. This does not apply to most of the other fields.
The model_name setting allows you to change which model you will start training from. The current configuration should be fine, but you can change it to another model in the model_zoo folder
If you want to point to your own model, change the model_name key to model_location and set the variable to the path to your model.
The train_dir and test_dir variables point at the training image and testing image directories respectively. The default settings assume you will place the train and test directories in your install directory.
The out_dir is the directory which configuration and training files will be written to. By default, this will be named detector_training.
The config_location variable points at the default config file. This should not need to be changed, but if you want to change training configuration such as learning rate, it is best to edit the default.config file by hand instead of swapping it out for another one, as other config files may cause problems with training.
  If you do want to use another configuration file, you can find files at <install>\DeepLearning\models\research\object_detection\samples\configs. Take note that none of them have been tested.
The train_data_location will need your username in the <Username> field. This is needed to access the checkpoint files made by the training script.
Labelimg:
Once your settings are configured, download the labelimg application from https://github.com/tzutalin/labelImg/releases/tag/v1.8.1. Extract and install it in your preferred location.
Gathering Images:
It is recommended to gather about 20 training images of your desired object, although more is usually better. Also gather test images which will be used to evaluate model preformance during training.
Place each batch of images in their respective directories. You are now ready to start labeling them. 
Image Tips: You may want to have images with similar objects in the frame to tune the detector to find the correct ones. You should also have images with different lighting and other interferance, as such interference will almost certainly be present on the playing field.
Labeling Images:
Use the labelimg application to open your training image directory and create bounding boxes around each of the images. Be sure to save the xml data files to the same directory as the images.
Once all of the images have been labeled, run the tftrain.py script. This should automatically detect all object classes you have made in your images and train your detector to recognize them.
When you press enter to stop training, the python program should save a frozen inference graph of the last checkpoint recorded by the trainer. If you see a traceback message, meaning something has gone wrong with the process, follow the onscreen directions.
Note:
  The automatic inference graph saving feature only works on Windows due to checkpoint save locations. It may be modified for Linux in the future.
