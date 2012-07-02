import hadoopy
import imfeat
import cv2
import numpy as np
import os
import feature
import numpy as np


#diff = np.linalg.norm(feature.compute(cv2.resize(image[box[0] + wy:box[2] + wy, box[1] + wx:box[3] + wx, :], (32, 32), interpolation=cv2.INTER_AREA)) - f)
#print('diff[%f]' % diff)

class Mapper(object):

    def __init__(self):
        pass

    def map(self, image_id, image_binary):
        image = imfeat.image_fromstring(image_binary)
        for box, f in feature.image_patch_features_random(image):
            yield (image_id, box), f


if __name__ == '__main__':
    hadoopy.run(Mapper, jobconfs=['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                  'mapred.compress.map.output=true',
                                  'mapred.output.compress=true',
                                  'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
