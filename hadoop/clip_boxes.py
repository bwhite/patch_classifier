import hadoopy
import imfeat
import numpy as np
import cPickle as pickle
import feature
import os
import cv2


class Mapper(object):

    def __init__(self):
        self.image_box_fns = pickle.load(open('image_box_fns.pkl'))  # [image_id] = [box_fns]
        self.type = os.environ.get('TYPE', 'image')
        if self.type not in ('image', 'feature', 'box'):
            raise ValueError('Unknown type[%s]' % self.type)

    def map(self, image_id, image_binary):
        try:
            boxes = self.image_box_fns[image_id]
        except KeyError:
            pass
        else:
            image = imfeat.image_fromstring(image_binary)
            size_array = np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
            scale_boxes = {}
            for box, fn in boxes:
                fbox = size_array * box
                scale = int(np.round(np.log2((fbox[2] - fbox[0]) / feature.PATCH_SIZE)))
                scale_check = int(np.round(np.log2((fbox[3] - fbox[1]) / feature.PATCH_SIZE)))
                if scale != scale_check:
                    raise ValueError('Box is not square.')
                scale_boxes.setdefault(scale, []).append((box, fn))
            # Order boxes and fn's by scale
            for scale in range(max(scale_boxes.keys()) + 1):
                if scale > 0:
                    height, width = np.array(image.shape[:2]) / 2
                    image = image[:height * 2, :width * 2, :]
                    if min(width, height) < 1:
                        raise ValueError('Image is too small')
                    image = cv2.resize(image, (width, height))
                if image is None:  # NOTE(brandyn): It is too small
                    raise ValueError('Image is too small')
                try:
                    boxes = scale_boxes[scale]
                except KeyError:
                    continue
                size_array = np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
                for box, fn in boxes:
                    box = np.round(size_array * box).astype(np.int)
                    print(box)
                    if self.type == 'image':
                        image_box = np.ascontiguousarray(image[box[0]:box[2], box[1]:box[3], :])
                        yield fn, imfeat.image_tostring(image_box, 'png')
                    elif self.type == 'feature':
                        image_box = np.ascontiguousarray(image[box[0]:box[2], box[1]:box[3], :])
                        yield fn, feature.compute_patch(image_box)
                    elif self.type == 'box':
                        image2 = image.copy()
                        cv2.rectangle(image2, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 10)
                        yield fn, imfeat.image_tostring(image2, 'jpg')
                    else:
                        raise ValueError(self.type)

if __name__ == '__main__':
    hadoopy.run(Mapper)
