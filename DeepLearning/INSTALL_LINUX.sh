sudo apt update
python get_model_zoo.py
sudo apt install libatlas-base-dev libopenblas-dev libblas-dev hdf5-dev liblapack-dev gfortran
sudo apt install python3-dev python3-pip
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
pip install Cython contextlib2 jupyter matplotlib keras
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
cd tensorflow/ /research
protoc object_detection/protos/*.proto --python_out=.
python setup.py build
python setup.py install
wget -O tensorflow_gpu-py3-none-any.whl https://files.pythonhosted.org/packages/76/04/43153bfdfcf6c9a4c38ecdb971ca9a75b9a791bb69a764d652c359aca504/tensorflow_gpu-1.14.0-cp36-cp36m-manylinux1_x86_64.whl
pip install tensorflow_gpu-py3-none-any.whl
read -p "Installation successful. Press Enter to end..."
