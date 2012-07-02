import imfeat
import cv2
import numpy as np


compute = imfeat.HOGLatent(sbin=4, blocks=1)


def _image_patch_features_base(image, inner_func, scales=6, normalize_box=False):
    orig_height, orig_width = image.shape[:2]
    for scale in [2 ** x for x in range(scales)]:
        if scale > 1:
            height, width = np.array(image.shape[:2]) / 2
            image = image[:height * 2, :width * 2, :]
            image = cv2.resize(image, (width, height))
        if image is None:  # NOTE(brandyn): It is too small
            return
        if normalize_box:
            norm_vec = 1. / np.asfarray([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
        else:
            norm_vec = np.ones(4, dtype=np.int)
        norm_vec *= scale
        for box, f in inner_func(image):
            yield norm_vec * box, f


def image_patch_features_random(image, patch_size=32, density=1, **kw):

    def _inner(image):
        for x in _sample_boxes(image.shape[:2], density=density, patch_size=patch_size):
            yield x, compute(image[x[0]:x[2], x[1]:x[3], :])
    return _image_patch_features_base(image, **kw)


def image_patch_features_dense(image, cell_skip=6, patch_size=32, cells=6, sbin=4, **kw):

    def _inner(image):
        f = compute(image, ravel=False)
        for row in range(0, f.shape[0] - cells, cell_skip):
            for col in range(0, f.shape[1] - cells, cell_skip):
                y = min(row * sbin, image.shape[0])
                x = min(col * sbin, image.shape[1])
                fs = np.ascontiguousarray(f[x:x + cells, y:y + cells, :].ravel())
                yield np.array([y, x, y + patch_size, x + patch_size]), fs

    return _image_patch_features_base(image, **kw)


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
