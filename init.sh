#!/usr/bin/env bash
set -e  # stop on error

ENV_NAME=openvla
PYTHON_VERSION=3.10

echo "Creating conda env..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch CUDA 11.8 build..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu118

echo "Cloning OpenVLA..."
git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .

echo "Installing CUDA toolkit (for flash-attn build)..."
conda install -c nvidia cuda-toolkit=11.8 -y
conda install nvidia::cuda-nvcc==11.7.64 -y
conda install conda-forge::cuda-nvcc==12.0.76 -y

export PATH=$CONDA_PREFIX/bin:$PATH

pip install packaging ninja
ninja --version

echo "Installing FlashAttention..."
pip install flash-attn==2.5.5 --no-build-isolation --no-cache-dir

cd ..

echo "Cloning LIBERO..."
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
touch libero/__init__.py
pip install -e .
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial

cd ../openvla

echo "Installing LIBERO requirements..."
pip install -r experiments/robot/libero/libero_requirements.txt

pip install numpy==1.26.4 tensorflow==2.15.0

echo "Downloading modified RLDS dataset..."
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds.git

echo "Setup complete."