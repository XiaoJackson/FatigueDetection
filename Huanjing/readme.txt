我在编译过程中用到的命令~
System:uname -a        arch

Opencv:sudo apt-get install python3-opencv 
	version:3.2.0     cv2.__version__

Pytorch:1.9.0 torch.__version__

torchvision:0.10.0 

onnxruntime:sudo apt-get install -y python3-dev build-essential cmake git
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
git checkout rel-1.8.1
./build.sh --config Release --update --build --build_wheel --use_openmp --arm
pip3 install build/Linux/Release/dist/*.whl
version:1.8.1



git clone --branch 1.1.3 https://github.com/alibaba/MNN.git
cd MNN
mkdir build
cd build
cmake .. -DMNN_BUILD_CONVERTER=true -DMNN_BUILD_FOR_LINUX_HOST_PYTHON=ON -DMNN_OPENMP=ON -DMNN_AAPL_FMWK_BUILD=OFF -DMNN_SEP_BUILD=OFF -DMNN_SUPPORT_BF16=OFF -DMNN_SUPPORT_TFLITE_QUAN=ON -DMNN_OPENGL=ON -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_SYSTEM_NAME=Linux
make -j$(nproc)
rm -rf build
python build_wheel.py --version 1.1.3 



git clone --branch 1.1.3 https://github.com/alibaba/MNN.git
cd MNN
git checkout 1.1.3
./schema/generate.sh
mkdir pymnn_build
cd pymnn/pip_package
python3 build_deps.py
python3 build_wheel.py --version 1.1.3
python3 build_wheel.py --version 2.8.1
pip install pymnn/pip_package/dist/MNN-1.1.3-cp37-cp37m-linux_aarch64.whl
pip install pymnn/pip_package/dist/MNN-2.8.1-cp37-cp37m-linux_aarch64.whl
from . import tools



git clone https://gitcode.com/alibaba/MNN.git

