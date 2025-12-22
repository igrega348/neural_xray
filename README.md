# High-speed X-ray tomography for 4D imaging

[![Paper](https://img.shields.io/badge/Paper-PNAS-blue)](https://www.pnas.org/doi/10.1073/pnas.2521089122)

## Motivation

X-ray computed tomography (X-CT) is an established method for 3d characterization of objects with applications ranging from medical imaging to industrial component inspection.
While in this project we focus on materials engineering, the developed techniques can be used in many other domains.

![Motivation](https://github.com/Neural-Xray/nerfxray/blob/main/assets/motivation.gif?raw=true)
_Left: Detailed brain X-CT [1]. Middle: X-CT of a gear with a set screw. Right: Strain inside a lattice sample._

Interrupted in-situ X-CT has been employed to study material response to deformation: for instance it revealed new insights into fracture behaviour of lattice materials [2] or surprising new behaviours in rubber elasticity [3].
The principle of the method is to interrupt deformation sequence at several stages and obtain a _tomogram_ at quasi-static conditions.
The acquisition of a tomogram requires the collection of many (~3000) projections and subsequent tomographic reconstruction.
The exposure time in a lab-based X-CT system is on the order of 1 second, therefore tomogram acquisition takes about 1 hour (which has to be repeated at every deformation stage).
Due to the number of projections needed for conventional tomographic reconstruction, it has not been posible to obtain 3d reconstruction of _dynamic, high-speed_ deformations.
In this work we address this limitation by developing a framework based on neural rendering in which we combine high-fidelity X-CT obtained at the start and end of deformation with few simultaneous projections taken during deformation to obtain a full 3d spatio-temporal reconstruction.

![Illustration of framework](https://github.com/user-attachments/assets/4cbde9e6-7c46-479d-ac39-bfd2c9b29999)
_Illustration of the framework._


## Setting up
    
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


## References

[1] Brain CT animation adapted from Human Organ Atlas ([doi:10.15151/ESRF-DC-572252655](http://doi.org/10.15151/ESRF-DC-572252655), [https://human-organ-atlas.esrf.eu/datasets/572252538](https://human-organ-atlas.esrf.eu/datasets/572252538)) under CC-BY-4.0 license.

[2] Shaikeea, A.J.D., Cui, H., O’Masta, M. et al. The toughness of mechanical metamaterials. Nat. Mater. 21, 297–304 (2022). [doi.org/10.1038/s41563-021-01182-1](https://doi.org/10.1038/s41563-021-01182-1).

[3] Wang, Z., Das, S., Joshi, A. et al. 3D observations provide striking findings in rubber elasticity. PNAS, 121, 24 (2024). [doi.org/10.1073/pnas.2404205121](https://doi.org/10.1073/pnas.2404205121)
