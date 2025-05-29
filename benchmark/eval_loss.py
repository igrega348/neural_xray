# %%
from pathlib import Path
from typing import Optional, Tuple, List
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
def load_obj(
        ref_path: Path, 
        resolution: Optional[Tuple[int, int, int]]=None, 
        dtype: Optional[DTYPES]=None
):
    if ref_path.suffix in ['.npy', '.npz', '.yaml']:
        assert resolution is None
        assert dtype is None
        vol = Object.from_file(ref_path)
    elif ref_path.suffix == '.raw':
        assert resolution is not None
        assert dtype is not None
        dtype = dtype.value
        vol = np.fromfile(ref_path, dtype=dtype)
        assert len(vol) == np.prod(resolution), f'Expected {np.prod(resolution)} elements but got {len(vol)} in {ref_path}'
        vol = vol.reshape(resolution[2], resolution[1], resolution[0]) # ZYX as needed for torch
        # vol = vol.swapaxes(0,2)
        vol = torch.from_numpy(vol.astype(float))
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
    out_dir: Optional[Path] = None,
    eval_resolution: int = 200,
    obj_resolution: Optional[Tuple[int, int, int]] = None,
    obj_dtype: Optional[DTYPES] = None,
    ref_resolution: Optional[Tuple[int, int, int]] = None,
    ref_dtype: Optional[DTYPES] = None,
    extent: Optional[List[Tuple[float, float]]] = None,
):
    print(f'Loading object from {obj_path}')
    obj = load_obj(obj_path, obj_resolution, obj_dtype)

    print(f'Loading reference object from {ref_path}')
    vol = load_obj(ref_path, ref_resolution, ref_dtype)

    if out_dir is None:
        out_dir = obj_path.parent

    pos = torch.linspace(-1, 1, eval_resolution)
    if extent is not None:
        assert len(extent) == 3
        xpos = torch.linspace(extent[0][0], extent[0][1], eval_resolution)
        ypos = torch.linspace(extent[1][0], extent[1][1], eval_resolution)
        zpos = torch.linspace(extent[2][0], extent[2][1], eval_resolution)
    else:
        xpos = pos
        ypos = pos
        zpos = pos
    pos = torch.stack(torch.meshgrid(xpos, ypos, zpos, indexing='ij'), dim=-1)
    density = obj.density(pos.view(-1, 3)).view(eval_resolution, eval_resolution, eval_resolution)
    ref_density = vol.density(pos.view(-1, 3)).view(eval_resolution, eval_resolution, eval_resolution)
    # normalise to zero mean and unit variance
    density_n = density - density.mean()
    density_n = density_n / density_n.std()
    pred_density_n = ref_density - ref_density.mean()
    pred_density_n = pred_density_n / pred_density_n.std()
    
    # plot slices as sanity check
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(ref_density[:,:,eval_resolution//2])
    axs[0].set_title('Target')
    axs[1].imshow(density[:,:,eval_resolution//2].cpu().numpy())
    axs[1].set_title('Reconstruction')
    axs[2].imshow(((density_n-pred_density_n)[:,:,eval_resolution//2].cpu().numpy()), cmap='bwr')
    axs[2].set_title('Difference')
    plt.savefig(out_dir/'slices_eval.png')
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
    print(f'Saving loss to {out_dir/"eval_loss.json"}')
    (out_dir/'eval_loss.json').write_text(json.dumps(loss_dict, indent=2))
# %%
if __name__=='__main__':
    tyro.cli(main)