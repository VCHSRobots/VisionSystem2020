These installation instructions are based off the documentation at https://devtalk.nvidia.com/default/topic/1048776/jetson-nano/official-tensorflow-for-jetson-nano-/
They will only work for the Jetson Nano with JetPack 4.2S

Run bash INSTALL_LINUX.sh
Add PYTONPATH="$PYTHONPATH:<Install Directory>/models/research:<Install Directory>/models/research/slim:<Install Directory>/models/research/object_detection" to ~/.profile