from pathlib import Path
import numpy as np
import cv2 as cv
import tyro

def main(input: Path):
    assert input.is_file(), f'Input file {input} does not exist'
    if input.suffix == '.npz':
        data = np.load(input)['vol']
    elif input.suffix == '.npy':
        data = np.load(input)
    else:
        raise ValueError(f'Invalid input file {input}')
    print(f'Loaded {data.size} elements of type {data.dtype}')
    nx, ny, nz = data.shape

    # Show 3 slices, one from each axis. Create sliders to change the slice.
    cv.namedWindow('x-slice', cv.WINDOW_NORMAL)
    cv.namedWindow('y-slice', cv.WINDOW_NORMAL)
    cv.namedWindow('z-slice', cv.WINDOW_NORMAL)
    cv.namedWindow('control', cv.WINDOW_NORMAL)

    # keep track of circles
    global circles
    circles = {
        'x': None,
        'y': None,
        'z': None
    }

    def update_slices(val):
        global circles

        x = cv.getTrackbarPos('x', 'control')
        y = cv.getTrackbarPos('y', 'control')
        z = cv.getTrackbarPos('z', 'control')
        slices = {
            'x': data[x,:,::-1].T,
            'y': data[:,y,::-1].T,
            'z': data[:,::-1,z].T
        }
        for k,v in slices.items():
            im = cv.medianBlur(v, 5)
            h,w = im.shape
            _circles = cv.HoughCircles(im, cv.HOUGH_GRADIENT, 1, h / 8,
                                    param1=100, param2=10,
                                    minRadius=0, maxRadius=0)
            circles[k] = _circles

            if _circles is not None:
                v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)
                _circles = np.uint16(np.around(_circles))
                for i in _circles[0, :]:
                    center = (i[0], i[1])
                    # circle outline
                    radius = i[2]
                    cv.circle(v, center, radius, (255, 0, 255), 3)

            cv.imshow(f'{k}-slice', v)

    def on_click_x(event, y, z, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            z = nz - z - 1
            cv.setTrackbarPos('y', 'control', y)
            cv.setTrackbarPos('z', 'control', z)
            update_slices(0)
            fit_sphere()

    def on_click_y(event, x, z, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            z = nz - z - 1
            cv.setTrackbarPos('x', 'control', x)
            cv.setTrackbarPos('z', 'control', z)
            update_slices(0)
            fit_sphere()

    def on_click_z(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            y = ny - y - 1
            cv.setTrackbarPos('x', 'control', x)
            cv.setTrackbarPos('y', 'control', y)
            update_slices(0)
            fit_sphere()

    def fit_sphere():
        x = cv.getTrackbarPos('x', 'control')
        y = cv.getTrackbarPos('y', 'control')
        z = cv.getTrackbarPos('z', 'control')
        for k,v in circles.items():
            if v is None: continue
            for p0, p1, r in v[0]:
                if k == 'x':
                    p1 = nz - p1
                    xc, yc, zc = x, p0, p1
                elif k == 'y':
                    p1 = nz - p1
                    xc, yc, zc = p0, y, p1
                elif k == 'z':
                    p1 = ny - p1
                    xc, yc, zc = p0, p1, z
                else: raise ValueError(f'Invalid slice {k}')

                # is the position of click inside the circle?
                if (xc-x)**2 + (yc-y)**2 + (zc-z)**2 <= r**2:
                    # yes, print the circle
                    print(f'{k}-slice: circle at ({xc}, {yc}) with radius {r}')

    cv.createTrackbar('x', 'control', 0, data.shape[0]-1, update_slices)
    cv.createTrackbar('y', 'control', 0, data.shape[1]-1, update_slices)
    cv.createTrackbar('z', 'control', 0, data.shape[2]-1, update_slices)
    cv.setMouseCallback('x-slice', on_click_x)
    cv.setMouseCallback('y-slice', on_click_y)
    cv.setMouseCallback('z-slice', on_click_z)
    update_slices(0)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=='__main__':
    tyro.cli(main)