import imfeat
import cv2
import numpy as np


PATCH_SIZE = 40
CELLS = 8
SBIN = 4
compute = imfeat.HOGLatent(sbin=SBIN, blocks=1)


def compute_patch(image):
    assert image.shape[0] == image.shape[1]
    assert image.shape[0] % PATCH_SIZE == 0
    while image.shape[0] != PATCH_SIZE:
        image = cv2.resize(image, (image.shape[1] / 2, image.shape[0] / 2))
    f = compute(image, ravel=False)
    return np.ascontiguousarray(f[1:-1, 1:-1, :].ravel())


def _image_patch_features_base(image, inner_func, scales=6, normalize_box=False):
    orig_height, orig_width = image.shape[:2]
    for scale in [2 ** x for x in range(scales)]:
        if scale > 1:
            height, width = np.array(image.shape[:2]) / 2
            image = image[:height * 2, :width * 2, :]
            if min(width, height) < 1:
                return
            image = cv2.resize(image, (width, height))
        if image is None:  # NOTE(brandyn): It is too small
            return
        if normalize_box:
            norm_vec = 1. / np.asfarray([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        else:
            norm_vec = np.ones(4, dtype=np.int) * scale
        any_boxes = False
        for box, f in inner_func(image, scale):
            any_boxes = True
            yield norm_vec * box, f
        if not any_boxes:
            return


def image_patch_features_random(image, density=1, **kw):

    def _inner(image, scale):
        for x in _sample_boxes(image.shape[:2], density=density, patch_size=PATCH_SIZE):
            yield x, compute_patch(image[x[0]:x[2], x[1]:x[3], :])
    return _image_patch_features_base(image, _inner, **kw)


def image_patch_features_dense(image, cell_skip=32, **kw):

    def _inner(image, scale):
        cur_cell_skip = cell_skip / scale ** 2
        f = compute(image, ravel=False)
        for row in range(0, f.shape[0] - CELLS, cur_cell_skip):
            for col in range(0, f.shape[1] - CELLS, cur_cell_skip):
                y = (row) * SBIN
                x = (col) * SBIN
                fs = np.ascontiguousarray(f[row + 1:row + CELLS - 1, col + 1:col + CELLS - 1, :].ravel())
                yield np.array([y, x, y + PATCH_SIZE, x + PATCH_SIZE]), fs

    return _image_patch_features_base(image, _inner, **kw)


def test_patchs(patch_func, image):
    for box, f in patch_func(image):
        f2 = compute_patch(image[box[0]:box[2], box[1]:box[3], :])
        np.testing.assert_equal(f, f2)


def _sample_boxes(image_size, density, patch_size):
    """
    Args:
        image_size: Size (width, height)
        num_boxes: Number of boxes
        size: Box size

    Returns:
        Numpy array of shape num_boxes x 4 where each is tl_y, tl_x, br_y, br_x
    """
    out = []

    image_size = np.asarray(image_size, dtype=np.int)
    num_boxes = int(np.prod(image_size / float(patch_size) * density))
    print('image_size:%s num_boxes:%s patch_size:%s' % (image_size, num_boxes, patch_size))
    try:
        tls = np.dstack([np.random.randint(0, image_size[0] - patch_size, num_boxes),
                         np.random.randint(0, image_size[1] - patch_size, num_boxes)])[0]
    except ValueError:
        pass
    else:
        out.append(np.hstack([tls, tls + np.array([patch_size, patch_size])]))
    try:
        return np.vstack(out)
    except ValueError:
        return np.array([], dtype=np.int).reshape(0, 4)

if __name__ == '__main__':
    im = cv2.imread('lena.ppm')
    test_patchs(image_patch_features_random, im)
    test_patchs(image_patch_features_dense, im)
