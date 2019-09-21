sudo apt update
python get_model_zoo.py
sudo apt install python3-dev python3-pip
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install --user Cython
pip install --user contextlib2
pip install --user jupyter
pip install --user matplotlib
pip install --user keras
git clone https://github.com/cocodataset/cocoapi.git
git clone https://github.com/tensorflow/models.git
rn cocoapi-master cocoapi
rn models-master models
py fix_cocoapi.py
cd cocoapi/PythonAPI
python setup.py build_ext --inplace
cd ..
cd ..
cp -r cocoapi/PythonAPI/pycocotools tensorflow/models/research/
cd tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
python setup.py build
python setup.py install
read -p "Installation successful. Press Enter to end..."
