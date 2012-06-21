import time
import os
import cv2
import numpy as np
import vidfeat
import random


def save_boxes(path, frame, boxes):
    try:
        os.makedirs(path)
    except OSError:
        pass
    st = time.time()
    for x, y in enumerate(boxes):
        cur_path = '%s/%f-%.5d.png' % (path, st, x)
        patch = cv2.resize(frame[y[0]:y[2], y[1]:y[3]], (32, 32))
        cv2.imwrite(cur_path, patch)


def sample_boxes(sz, density=1., sizes=(32, 64, 128)):
    """
    Args:
        sz: Size (width, height)
        num_boxes: Number of boxes
        size: Box size

    Returns:
        Numpy array of shape num_boxes x 4 where each is tl_y, tl_x, br_y, br_x
    """
    out = []
    for size in sizes:
        sz = np.asarray(sz, dtype=np.int)
        num_boxes = int(np.prod(sz / float(size) * density))
        print('sz:%s num_boxes:%s size:%s' % (sz, num_boxes, size))
        if num_boxes <= 0:
            continue
        try:
            tls = np.dstack([np.random.randint(0, sz[0] - size, num_boxes),
                             np.random.randint(0, sz[1] - size, num_boxes)])[0]
        except ValueError:
            pass
        else:
            out.append(np.hstack([tls, tls + np.array([size, size])]))
    try:
        return np.vstack(out)
    except ValueError:
        if not out:
            return np.array([], dtype=np.int).reshape(0, 4)
        print(out)


def remove_tiling(frame):
    num_rows = 0
    while np.all(frame[-(num_rows + 2), :, :] == frame[-1, :, :]):
        num_rows += 1
    print('NR:%d' % num_rows)
    if num_rows >= 1:
        print('Removing tiling')
        try:
            os.makedirs('remove_tiling/')
        except OSError:
            pass
        p = 'remove_tiling/%f' % random.random()
        cv2.imwrite(p + '.png', frame)
        frame = frame[:-(num_rows + 1), :, :]
        cv2.imwrite(p + '-clean.png', frame)
    return frame


def write_boxes():
    for label, frame in vidfeat.load_label_frames('/aladdin_data_cropped/person/'):
        if label == 1:  # Some of the people have a tiling artifact on the bottom
            frame = remove_tiling(frame)
        boxes = sample_boxes(frame.shape[:2])
        save_boxes('/aladdin_data_cropped/boxes/%d' % label, frame, boxes)
        print(label)
