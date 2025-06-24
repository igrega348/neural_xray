# neural_xray
Exploring the potential for neural rendering methods in x-ray tomography

## How to run the code
    
1. Get access to a machine with Nvidia GPU (local, or cloud; see platform-specific instructions below)
2. Clone the repository with submodules
```
git clone --recurse-submodules https://github.com/igrega348/neural_xray.git 
```


### Lambda
- Need to use identity file. E.g. using VS Code, the .ssh config file would contain path to private key
```
Host 192.222.[XX].[XX]
  HostName 192.222.[XX].[XX]
  IdentityFile C:\Users\[John Smith]\.ssh\lambda-id 
  User ubuntu
```
or use basic ssh command with -i flag
```
ssh ubuntu@192.222.[XX].[XX] -i C:\Users\[John Smith]\.ssh\lambda-id 
```
- install miniconda; use platform-specific installation file. E.g. for Arm-based cluster:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
sh Miniconda3-latest-Linux-aarch64.sh
```
Advisable to install onto the filesystem which pertains between sessions (e.g. in our example, the
path to miniconda folder is `/home/ubuntu/nerfstudio-drive/miniconda3`).

- set up an environment and activate
```
conda create --name nerfstudio -y python=3.9
conda activate nerfstudio
pip install --upgrade pip
conda install -c "nvidia/label/cuda-12.6.0" cuda-toolkit
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install ninja 
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
cd neural_xray/nerfstudio
pip install --upgrade setuptools
pip install -e .
cd ../nerfstudio-xray/nerf-xray/
pip install -e .
```
