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
        assert self.type in ('image', 'feature', 'box')

    def map(self, image_id, image_binary):
        try:
            boxes = self.image_box_fns[image_id]
        except KeyError:
            pass
        else:
            image = imfeat.image_fromstring(image_binary)
            size_array = np.array([image.shape[0], image.shape[1], image.shape[0], image.shape[1]])
            for box, fn in boxes:
                box = np.round(size_array * box).astype(np.int)
                print(box)
                image_box = np.ascontiguousarray(image[box[0]:box[2], box[1]:box[3], :])
                if self.type == 'image':
                    yield fn, imfeat.image_tostring(image_box, 'png')
                elif self.type == 'feature':
                    yield fn, feature.compute_patch(image_box)
                elif self.type == 'box':
                    image2 = image.copy()
                    cv2.rectangle(image2, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 10)
                    yield fn, imfeat.image_tostring(image2, 'jpg')
                else:
                    raise ValueError(self.type)

if __name__ == '__main__':
    hadoopy.run(Mapper)
