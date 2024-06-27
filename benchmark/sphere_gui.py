from pathlib import Path
import numpy as np
import cv2 as cv
import tyro

def main(
    input: Path,
    zxy: bool = False
):
    assert input.is_file(), f'Input file {input} does not exist'
    if input.suffix == '.npz':
        data = np.load(input)['vol']
    elif input.suffix == '.npy':
        data = np.load(input)
    else:
        raise ValueError(f'Invalid input file {input}')
    if zxy:
        data = data.swapaxes(0,2)
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
        circle_data = {}
        for k,v in circles.items():
            if v is None: continue
            for p0, p1, r in v[0]:
                if k == 'x':
                    p1 = nz - p1
                    yc, zc = p0, p1
                    if (y-yc)**2 + (z-zc)**2 <= r**2:
                        circle_data[k] = {k: x, 'ycx': yc, 'zcx': zc, 'rx': r}
                elif k == 'y':
                    p1 = nz - p1
                    xc, zc = p0, p1
                    if (x-xc)**2 + (z-zc)**2 <= r**2:
                        circle_data[k] = {k: y, 'xcy': xc, 'zcy': zc, 'ry': r}
                elif k == 'z':
                    p1 = ny - p1
                    xc, yc = p0, p1
                    if (x-xc)**2 + (y-yc)**2 <= r**2:
                        circle_data[k] = {k: z, 'xcz': xc, 'ycz': yc, 'rz': r}
                else: raise ValueError(f'Invalid slice {k}')

        if len(circle_data) < 3:
            return
        
        # find centre
        xc = (circle_data['y']['xcy'] + circle_data['z']['xcz']) / 2
        yc = (circle_data['x']['ycx'] + circle_data['z']['ycz']) / 2
        zc = (circle_data['x']['zcx'] + circle_data['y']['zcy']) / 2
        # find radius
        deltax = x - xc
        deltay = y - yc
        deltaz = z - zc
        rx = np.sqrt(deltax**2 + circle_data['x']['rx']**2)
        ry = np.sqrt(deltay**2 + circle_data['y']['ry']**2)
        rz = np.sqrt(deltaz**2 + circle_data['z']['rz']**2)
        R = np.mean([rx, ry, rz])
        print(f'Centre: {xc}, {yc}, {zc}, Radius: {R:.3g}')
        # normalized coordinates
        xc = xc / nx * 2 - 1
        yc = yc / ny * 2 - 1
        zc = zc / nz * 2 - 1
        assert nx == ny == nz
        R = R / nx * 2
        print(f'\tNormalised: ({xc:.3g}, {yc:.3g}, {zc:.3g}), {R:.3g}')

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