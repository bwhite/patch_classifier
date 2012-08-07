import patch_predict
import numpy as np
import hadoopy


class Mapper(patch_predict.Mapper):

    def __init__(self):
        super(Mapper, self).__init__()
        self.levels = 4
        self.num_bins = 2 ** self.levels
        self.pyramid = np.zeros((len(self.ids), self.num_bins, self.num_bins), dtype=np.int32)
        self.num_boxes = 0

    def map(self, image_id, image_binary):
        for (image_id, box), confs in super(Mapper, self).map(image_id, image_binary):
            self.num_boxes += 1
            cy = (box[2] + box[0]) / 2 * self.num_bins
            cx = (box[1] + box[3]) / 2 * self.num_bins
            inds = (confs >= 0).nonzero()[0]
            hadoopy.counter('STATS', 'num_pos', inds.size)
            hadoopy.counter('STATS', 'num_neg', confs.size - inds.size)
            hadoopy.counter('STATS', 'total', confs.size)
            if inds.size:
                self.pyramid[inds, cy, cx] += 1

    def close(self):
        yield 0, (self.pyramid, float(self.num_boxes))


class Reducer(object):

    def __init__(self):
        pass

    def reduce(self, key, pyramid_num_boxes):
        pyramid_out = 0
        num_boxes_out = 0
        for pyramid, num_boxes in pyramid_num_boxes:
            pyramid_out += pyramid
            num_boxes_out += num_boxes
        yield key, (pyramid_out, num_boxes_out)

if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer, jobconfs=['mapred.task.timeout=6000000',
                                           'mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                           'mapred.compress.map.output=true',
                                           'mapred.output.compress=true',
                                           'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
