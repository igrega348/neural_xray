# %%
from pathlib import Path
from typing import Optional
from enum import Enum
import tyro
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from nerf_xray.objects import Object, VoxelGrid
# %%
class DTYPES(Enum):
    UINT8 = np.uint8
    UINT16 = np.uint16
    UINT32 = np.uint32
    UINT64 = np.uint64
    INT8 = np.int8
    INT16 = np.int16
    INT32 = np.int32
    INT64 = np.int64
    FLOAT32 = np.float32
    FLOAT64 = np.float64
# %%
def load_obj(ref_path: Path, resolution: Optional[int]=None, dtype: Optional[DTYPES]=None):
    if ref_path.suffix in ['.npy', '.npz', '.yaml']:
        assert resolution is None
        assert dtype is None
        vol = Object.from_file(ref_path)
    elif ref_path.suffix == '.raw':
        assert resolution is not None
        assert dtype is not None
        vol = np.fromfile(ref_path, dtype=dtype).astype(dtype)
        vol = vol.reshape(resolution, resolution, resolution) # ZYX as needed for torch
        vol = torch.from_numpy(vol).float()
        vol = VoxelGrid(vol)
    else:
        raise ValueError(f'Unsupported file format {ref_path.suffix}')
    if isinstance(vol, VoxelGrid):
        print(f'VoxelGrid of shape {vol.rho.shape} loaded')
    return vol
# %%
def main(
    obj_path: Path, 
    ref_path: Path, 
    eval_resolution: int = 200,
    obj_resolution: Optional[int] = None,
    obj_dtype: Optional[DTYPES] = None,
    ref_resolution: Optional[int] = None,
    ref_dtype: Optional[DTYPES] = None,
):
    print(f'Loading object from {obj_path}')
    obj = load_obj(obj_path, obj_resolution, obj_dtype)

    print(f'Loading reference object from {ref_path}')
    vol = load_obj(ref_path, ref_resolution, ref_dtype)

    pos = torch.linspace(0, 1, eval_resolution)
    pos = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'), dim=-1)
    density = obj.t_density(pos.view(-1, 3)).view(eval_resolution, eval_resolution, eval_resolution)
    ref_density = vol.t_density(pos.view(-1, 3)).view(eval_resolution, eval_resolution, eval_resolution)
    
    # plot slices as sanity check
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(ref_density[:,:,eval_resolution//2])
    axs[0].set_title('Recon')
    axs[1].imshow(density[:,:,eval_resolution//2].cpu().numpy())
    axs[1].set_title('Target')
    plt.savefig(ref_path.parent/'slices_eval.png')
    plt.close()

    y = ref_density.flatten()
    x = density.flatten()

    density_loss = torch.nn.functional.mse_loss(y, x).item()

    density_n = (x - x.min()) / (x.max() - x.min())
    pred_dens_n = (y - y.min()) / (y.max() - y.min())
    scaled_density_loss = torch.nn.functional.mse_loss(pred_dens_n, density_n).item()
    
    mux = x.mean()
    muy = y.mean()
    dx = x-mux
    dy = y-muy
    normed_correlation = torch.sum(dx*dy) / torch.sqrt(dx.pow(2).sum() * dy.pow(2).sum())
    loss_dict = {
        'volumetric_loss': density_loss, 
        'scaled_volumetric_loss': scaled_density_loss,
        'normed_correlation': normed_correlation.item()
        }
    print(loss_dict)
    # save to file
    (ref_path.parent/'eval_loss.json').write_text(json.dumps(loss_dict, indent=2))
# %%
if __name__=='__main__':
    tyro.cli(main)