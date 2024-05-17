# %%
import astra
import numpy as np
import matplotlib.pyplot as plt
import pylab
from pathlib import Path
import cv2 as cv
from rich.progress import track
# %%
ims = []
modulo = 8
for f in Path('../../nerf_data/synthetic/balls/images').glob('train*png'):
  im_id = int(f.stem.split('_')[-1])
  if im_id % modulo != 0:
    continue
  im = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
  im = cv.resize(im, None, fx=0.25, fy=0.25)
  im = 1 - im.astype(np.float32)/255
  ims.append(im)
proj_data = np.stack(ims, axis=1)
print(proj_data.shape)
# %%
vol_geom = astra.create_vol_geom(256, 256, 256)

angles = np.arange(proj_data.shape[1])*modulo * 2*np.pi/256
# angles = np.linspace(0, 2*np.pi, proj_data.shape[1], False)
# proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 128, 192, angles)
L = 1000
detsize = 2*L*np.tan(22.5*np.pi/180)
det_count = ims[0].shape[0]
proj_geom = astra.create_proj_geom('cone', detsize/det_count, detsize/det_count, det_count, det_count, angles, L/2, L/2)
# proj_geom = astra.create_proj_geom('cone', 2, 2, 128, 128, angles, 1000, 500)

# Create a simple hollow cube phantom
cube = np.zeros((256,256,256))
cube[17:113,17:113,17:113] = 1
cube[33:97,33:97,33:97] = 0

# Create projection data from this
# proj_id, proj_data = astra.create_sino3d_gpu(cube, proj_geom, vol_geom)
proj_id = astra.create_sino3d_gpu(cube, proj_geom, vol_geom, returnData=False)
astra.data3d.store(proj_id, proj_data)

# Display a single projection image
pylab.gray()
pylab.figure(1)
pylab.imshow(proj_data[:,0,:])
# %%

# Create a data object for the reconstruction
rec_id = astra.data3d.create('-vol', vol_geom)

# Set up the parameters for a reconstruction algorithm using the GPU
cfg = astra.astra_dict('SIRT3D_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id


# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)

# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
# niters = 150
# astra.algorithm.run(alg_id, niters)
nIters = 150
residual_error = np.zeros(nIters)
for i in track(range(nIters), description='Reconstructing'):
  # Run a single iteration
  astra.algorithm.run(alg_id, 1)
  residual_error[i] = astra.algorithm.get_res_norm(alg_id)


# Get the result
rec = astra.data3d.get(rec_id)
pylab.figure(2)
pylab.imshow(rec[:,:,128])

pylab.figure(4)
pylab.plot(residual_error)
pylab.show()

out_dir = Path(f'./reconstructed/slices_{proj_data.shape[1]}')
out_dir.mkdir(exist_ok=True, parents=True)
for i in range(0,rec.shape[2],rec.shape[2]//64):
  # plt.imsave(str(out_dir/f'slice_{i:03d}.png'), rec[:,:,i], cmap='gray')
  cv.imwrite(str(out_dir/f'slice_{i:03d}.png'), ((rec[:,:,i]-rec.min()) / (rec.max()-rec.min())*255).astype(np.uint8))

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)

# %%
