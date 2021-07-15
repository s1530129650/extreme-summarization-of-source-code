conda create -n conv_cs python=3.6 -y
source ~/miniconda3/etc/profile.d/conda.sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate conv_cs
conda install pytorch=0.4.1 torchvision=0.2.1 cudatoolkit=11.0 -c pytorch
pip install torchtext==0.2.3
conda install numpy
pip install six