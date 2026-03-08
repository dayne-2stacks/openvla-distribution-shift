set -e  # stop on error

ENV_NAME=openvla
PYTHON_VERSION=3.10

conda activate $ENV_NAME


echo "Installing FlashAttention..."
pip install flash-attn==2.5.5 --no-build-isolation --no-cache-dir

echo "Cloning LIBERO..."
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
touch libero/__init__.py
pip install -e .
echo "n" | python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial

cd ../openvla

echo "Installing LIBERO requirements..."
pip install -r experiments/robot/libero/libero_requirements.txt

pip install numpy==1.26.4 tensorflow==2.15.0

echo "Downloading modified RLDS dataset..."
git clone https://huggingface.co/datasets/openvla/modified_libero_rlds.git

echo "Setup complete."