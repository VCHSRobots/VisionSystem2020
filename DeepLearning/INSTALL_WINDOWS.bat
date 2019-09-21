py get_model_zoo.py
pip install tensorflow-gpu
pip install virtualenv
pip install Cython
pip install contextlib2
pip install pillow
pip install lxml
pip install jupyter
pip install matplotlib
pip install keras
git clone https://github.com/cocodataset/cocoapi.git
git clone https://github.com/tensorflow/models.git
rename cocoapi-master cocoapi
rename models-master models
py fix_cocoapi.py
cd cocoapi\PythonAPI
py setup.py build_ext install
cd ..
cd ..
copy cocoapi\PythonAPI\pycocotools\ models\research\
cd models\research
protoc object_detection\protos\*.proto --python_out=.
python setup.py build
python setup.py install
pip install tensorflow-gpu
echo "Installation sucessfull. Press Enter to end."
PAUSE
