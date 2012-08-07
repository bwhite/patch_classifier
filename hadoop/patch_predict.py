import numpy as np
import hadoopy
import os
import feature
import imfeat
import time
import cv2
import cPickle as pickle


def resize(image, max_side=512):
    if np.max(image.shape[:2]) > max_side:
        height, width = (max_side * np.array(image.shape[:2]) / np.max(image.shape[:2])).astype(np.int)
        print(image.shape)
        print('Resizing to (%d, %d), from(%s)' % (height, width, image.shape))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


class Mapper(object):

    def __init__(self):
        self.ids, self.coefs, self.intercepts = zip(*[(x, y, z) for x, (y, z) in pickle.load(open(os.environ['EXEMPLARS']))])
        #self.ids, self.coefs, self.intercepts = zip(*[(x, y, z) for x, (y, z) in hadoopy.readtb(os.environ['EXEMPLARS'])])
        self.coefs = np.ascontiguousarray(self.coefs)
        self.intercepts = np.ascontiguousarray(self.intercepts)
        print('NumExemplars[%d] Coefs[%s] Intercepts[%s]' % (len(self.ids), self.coefs.shape, self.intercepts.shape))

    def map(self, image_id, image_binary):
        image = resize(imfeat.image_fromstring(image_binary))
        print(image.shape)
        st = time.time()
        box_num = -1
        for box_num, (box, f) in enumerate(feature.image_patch_features_dense(image, normalize_box=True)):
            yield (image_id, box.tolist()), np.dot(self.coefs, f.reshape((f.size, 1))).ravel() + self.intercepts
        hadoopy.counter('stats', 'num_boxes', box_num + 1)
        print('ImageTime[%f]' % (time.time() - st))

if __name__ == '__main__':
    hadoopy.run(Mapper, jobconfs=['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                  'mapred.compress.map.output=true',
                                  'mapred.output.compress=true',
                                  'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                  'mapred.task.timeout=6000000'])
