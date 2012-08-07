import numpy as np
import cv2


def chip_square(chips=None, num_per_side=15, chip_size=40):
    assert num_per_side % 2 == 1
    c = num_per_side / 2
    xmesh, ymesh = np.meshgrid(range(num_per_side), range(num_per_side))
    d = ((xmesh - c) ** 2 + (ymesh - c) ** 2).ravel()
    square = np.zeros((num_per_side * chip_size, num_per_side * chip_size, 3), dtype=np.uint8)
    for chip, ind in zip(chips, np.argsort(d.ravel())):
        chip = cv2.resize(chip, (chip_size, chip_size))
        y, x = ymesh.flat[ind] * chip_size, xmesh.flat[ind] * chip_size
        square[y:y + chip_size, x:x + chip_size, :] = chip
    return square

import glob

base = 'exemplars_similar_cropped'
for sz in [11, 21, 41]:
    square = chip_square((cv2.imread(x) for x in sorted(glob.glob(base + '/*'), reverse=True)), sz)
    cv2.imwrite('square-pos-%d.jpg' % sz, square)
    square = chip_square((cv2.imread(x) for x in sorted(glob.glob(base + '/*'))), sz)
    cv2.imwrite('square-neg-%d.jpg' % sz, square)
