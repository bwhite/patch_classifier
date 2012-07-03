import hadoopy
import imfeat
import cv2
import numpy as np
import os
import feature
import numpy as np

class Mapper(object):

    def __init__(self):
        pass

    def map(self, image_id, image_binary):
        image = imfeat.image_fromstring(image_binary)
        for box, f in feature.image_patch_features_dense(image):
            yield (image_id, box), f


if __name__ == '__main__':
    hadoopy.run(Mapper, jobconfs=['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                  'mapred.compress.map.output=true',
                                  'mapred.output.compress=true',
                                  'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
