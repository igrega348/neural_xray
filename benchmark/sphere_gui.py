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

    # Show 3 slices, one from each axis. Create sliders to change the slice.
    cv.namedWindow('x-slice', cv.WINDOW_NORMAL)
    cv.namedWindow('y-slice', cv.WINDOW_NORMAL)
    cv.namedWindow('z-slice', cv.WINDOW_NORMAL)
    cv.namedWindow('control', cv.WINDOW_NORMAL)

    def on_change(val):
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
            circles = cv.HoughCircles(im, cv.HOUGH_GRADIENT, 1, h / 8,
                                    param1=100, param2=10,
                                    minRadius=0, maxRadius=0)

            if circles is not None:
                v = cv.cvtColor(v, cv.COLOR_GRAY2BGR)
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle outline
                    radius = i[2]
                    cv.circle(v, center, radius, (255, 0, 255), 3)

            cv.imshow(f'{k}-slice', v)

    cv.createTrackbar('x', 'control', 0, data.shape[0]-1, on_change)
    cv.createTrackbar('y', 'control', 0, data.shape[1]-1, on_change)
    cv.createTrackbar('z', 'control', 0, data.shape[2]-1, on_change)
    on_change(0)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__=='__main__':
    tyro.cli(main)