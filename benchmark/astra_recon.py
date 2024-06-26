# %%
import astra
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2 as cv
import tyro
from typing import Tuple, Callable, Optional, Literal
import json
from rich.progress import track
from utils import PrintTableMetrics

def load_image(
        dname: Path,
        filepath: Path,
        image_downscale: float
) -> np.ndarray:
    im = cv.imread((dname/filepath).as_posix(), cv.IMREAD_GRAYSCALE)
    im = cv.resize(im, None, fx=1/image_downscale, fy=1/image_downscale)
    im = 1 - im.astype(np.float32)/255
    # flip upside down
    im = np.flipud(im)
    return im

def load_projections(
        path: Path, 
        filter_name: Callable, 
        image_downscale: float = 1.0, 
        Lscale: Optional[float] = 1.0
) -> Tuple[np.ndarray, dict]:
    path = Path(path)
    if path.is_dir():
        fn = list(path.glob('transforms*.json'))
        assert len(fn) == 1
        jsonfile = fn[0]
        dname = path
    elif 'json' in path.suffix:
        jsonfile = path
        dname = path.parent
    else:
        raise ValueError('Invalid path')
    meta = json.loads(jsonfile.read_text())
   
    image_filenames = []
    images = []
    poses = []

    fnames = []
    frame_indices = []
    for i, frame in enumerate(meta["frames"]):
        filepath = Path(frame["file_path"])
        if filter_name(filepath.stem):
            fnames.append(filepath.as_posix())
            frame_indices.append(i)
    # sort the frames by fname
    inds = np.argsort(fnames)
    inds = [frame_indices[i] for i in inds]
    print(f'Loading {len(inds)} images')
    frames = [meta["frames"][ind] for ind in inds]
    del inds

    for frame in track(frames, description='Loading images'):
        filepath = Path(frame["file_path"])
        image_filenames.append(filepath.as_posix())
        poses.append(np.array(frame["transform_matrix"]))
        im = load_image(dname, filepath, image_downscale)
        images.append(im)
    poses = np.stack(poses, axis=0)
    images = np.stack(images, axis=1)

    eye = np.array([0,0,0,1])
    eye = np.einsum('...ij,j', poses, eye) # eye to world coordinates
    assert np.allclose(eye[:,2], 0)
    R = np.linalg.norm(eye[:,:2], axis=1) # distance from eye to origin
    assert np.allclose(R, R[0]) # should be the same for all frames
    R = R[0]
    theta = np.arctan2(eye[:,1], eye[:,0])
    theta[theta < 0] += 2*np.pi

    fl_x = float(meta['fl_x']) # in pixels
    fl_y = float(meta['fl_y']) # in pixels
    assert fl_x == fl_y
    w = float(meta['w']) # in pixels
    h = float(meta['h']) # in pixels
    assert w == h
    f = fl_x / (w/2)
    # put detector at 2*R from source
    detector_size = 2* 2*R / f
    
    geom_data = {}
    geom_data['DetectorPixelsX'] = w
    geom_data['DetectorPixelsY'] = h
    geom_data['SrcToObject'] = R
    geom_data['SrcToDetector'] = 2*R
    geom_data['DetectorPixelSizeX'] = detector_size / w
    geom_data['DetectorPixelSizeY'] = detector_size / h

    proj_geom = {
        'type':'cone',
        'DetectorSpacingX':geom_data['DetectorPixelSizeX']*image_downscale*Lscale, # L
        'DetectorSpacingY':geom_data['DetectorPixelSizeY']*image_downscale*Lscale, # L
        'DetectorRowCount':int(geom_data['DetectorPixelsX']/image_downscale), # -
        'DetectorColCount':int(geom_data['DetectorPixelsY']/image_downscale), # -
        'ProjectionAngles':theta, # - 
        'DistanceOriginSource':geom_data['SrcToObject']*Lscale, # L
        'DistanceOriginDetector':(geom_data['SrcToDetector']-geom_data['SrcToObject'])*Lscale # L
    }

    return images, proj_geom, image_filenames

def normalize_reorder(rec: np.ndarray) -> np.ndarray:
    rec = (rec - rec.min()) / (rec.max() - rec.min()) * 255
    rec = np.transpose(rec, (1,2,0)) # z x y -> x y z
    rec = np.flip(rec, axis=0) # flip x
    return rec

def save_slice(rec: np.ndarray, output_dir: Path):
    plt.figure()
    plt.imshow(rec[:,:,rec.shape[2]//2])
    plt.savefig(output_dir/f'slice.png')
    plt.close()

def main(
        input_dir: Path,
        output_dir: Path,
        Lscale: Optional[float] = None,
        image_downscale: float = 1.0,
        resolution: int = 256,
        imin: int = 0,
        imax: int = 1 << 26,
        istep: int = 1,
        algorithm: Optional[Literal['SIRT3D_CUDA','CGLS3D_CUDA']] = 'SIRT3D_CUDA'
):
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output folder: {output_dir}')

    if Lscale is None:
        Lscale = float(resolution/2)
    
    def filter_name(x):
        # if 'train' not in x:
        #     return False
        i = int(x.split("_")[-1])
        return i >= imin and i<=imax and (i - imin) % istep == 0
    # filter_name = lambda x: True

    proj_data, proj_geom, image_filenames = load_projections(
        input_dir, filter_name, image_downscale, Lscale
    )

    vol_geom = astra.create_vol_geom(resolution, resolution, resolution)

    # Create empty volume
    cube = np.zeros((vol_geom['GridSliceCount'], vol_geom['GridRowCount'], vol_geom['GridColCount'])) # todo check row col order
  
    # Create projection data from this
    proj_id = astra.create_sino3d_gpu(cube, proj_geom, vol_geom, returnData=False)
    astra.data3d.store(proj_id, proj_data)

    # Display a single projection image
    plt.figure(1)
    plt.imshow(proj_data[:,0,:], cmap='gray')
    plt.savefig(output_dir/'projection.png')
    plt.close()

    # Create a data object for the reconstruction
    rec_id = astra.data3d.create('-vol', vol_geom)

    # Set up the parameters for a reconstruction algorithm using the GPU
    cfg = astra.astra_dict(algorithm)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {'GPUindex': 0} # can be useful for multi GPU machines

    # Create the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)

    # Run the algorithm
    neach = 2
    nmax = 50
    print(f'Running {neach} iterations per step')
    residual_error = []
    t = PrintTableMetrics(['Iteration', 'Error', 'de/rng'])
    de_rng = 0
    for i in range(nmax):
        # Run a single iteration
        astra.algorithm.run(alg_id, neach)
        residual_error.append(astra.algorithm.get_res_norm(alg_id))

        # save slice
        rec = astra.data3d.get(rec_id)
        rec = normalize_reorder(rec)
        save_slice(rec, output_dir)

        t.update({'Iteration': i, 'Error': residual_error[-1]})
        # check convergence
        if len(residual_error) > 1:
            rng = max(residual_error) - min(residual_error)
            de = residual_error[-1] - residual_error[-2]
            de_rng = de / rng
            # improvement by less than threshold
            if de > 0 or -de_rng < 0.001:
                break
        t.update({'Iteration': i, 'Error': residual_error[-1], 'de/rng': de_rng})

    # Get the result and save
    rec = astra.data3d.get(rec_id)
    rec = normalize_reorder(rec)
    _tmp = rec.swapaxes(0,2).astype(np.float32)
    _tmp.tofile(output_dir/f'vol_zyx.raw') 
    np.save(output_dir/f'vol_zyx.npy', _tmp)
    del _tmp

    # save slice
    save_slice(rec, output_dir)

    plt.figure(4)
    plt.plot(neach*np.arange(len(residual_error)), residual_error)
    plt.savefig(output_dir/f'convergence.png')
    plt.close()

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(rec_id)
    astra.data3d.delete(proj_id)

    # save proj_geom to config.json file
    proj_geom['ProjectionAngles'] = proj_geom['ProjectionAngles'].tolist()
    proj_geom['image_filenames'] = image_filenames
    proj_geom['algorithm'] = algorithm
    (output_dir/'config.json').write_text(json.dumps(proj_geom, indent=2))

if __name__ == '__main__':
    tyro.cli(main)