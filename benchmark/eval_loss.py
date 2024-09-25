# %%
from pathlib import Path
import tyro
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from nerf_xray.objects import Object
# %%
def main(obj_path: Path, volume_grid_path: Path, resolution: int = 200):
    if not isinstance(obj_path, Path):
        obj_path = Path(obj_path)
    if obj_path.is_dir():
        obj_path = list(obj_path.glob('*.yaml'))
        assert len(obj_path) == 1, f'Found {len(obj_path)} yaml files in {obj_path}'
        obj_path = obj_path[0]
    if not isinstance(volume_grid_path, Path):
        volume_grid_path = Path(volume_grid_path)
    assert 'yaml' in obj_path.suffix
    obj = Object.from_yaml(obj_path)
    if volume_grid_path.suffix == '.npy':
        vol = np.load(volume_grid_path)
        resolution = vol.shape[0]
    elif volume_grid_path.suffix == '.npz':
        vol = np.load(volume_grid_path)['vol']
        resolution = vol.shape[0]
    else:
        assert volume_grid_path.suffix == '.raw'
        vol = np.fromfile(volume_grid_path, dtype='uint16').astype(np.float32)
        vol = vol.reshape(resolution, resolution, resolution).swapaxes(0, 2) # xyz

    pos = torch.linspace(0, 1, resolution)
    pos = torch.stack(torch.meshgrid(pos, pos, pos, indexing='ij'), dim=-1)
    density = obj.t_density(pos.view(-1, 3)).view(resolution, resolution, resolution)
    
    # plot slices as sanity check
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(vol[:,:,100])
    axs[0].set_title('Recon')
    axs[1].imshow(density[:,:,100].cpu().numpy())
    axs[1].set_title('Target')
    plt.savefig(volume_grid_path.parent/'slices_eval.png')
    plt.close()

    y = torch.from_numpy(vol).float().flatten() / 255.0
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
    (volume_grid_path.parent/'eval_loss.json').write_text(json.dumps(loss_dict, indent=2))
# %%
if __name__=='__main__':
    tyro.cli(main)