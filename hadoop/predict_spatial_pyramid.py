import patch_predict
import numpy as np
import hadoopy
#import scipy as sp
#import scipy.sparse
import random
#import time
#import kernels
#import scipy.spatial.distance


class Mapper(patch_predict.Mapper):

    def __init__(self):
        super(Mapper, self).__init__()
        self.levels = 4
        self.num_bins = np.array([2 ** x for x in range(self.levels)], dtype=np.int32)
        self.num_bins_sqr = self.num_bins * self.num_bins
        self.bin_weight = np.zeros(np.sum(self.num_bins_sqr))
        prev = 0
        for x in self.num_bins_sqr:
            self.bin_weight[prev:prev + x] += x
            prev += x
        self.bin_weight /= np.sum(self.bin_weight)
        self.bin_weight /= float(self.levels)
        self.num_inputs = 1
        print('BinWeight[%s]' % str(self.bin_weight))

    def map(self, image_id, image_binary):
        if self.num_inputs <= 0:
            return
        self.num_inputs -= 1
        pyramid = np.zeros((len(self.ids), np.sum(self.num_bins_sqr)), dtype=np.int32)
        num_boxes = 0
        for (image_id, box), confs in super(Mapper, self).map(image_id, image_binary):
            num_boxes += 1
            cy = (box[2] + box[0]) / 2
            cx = (box[1] + box[3]) / 2
            offset = 0
            cur_bins = []
            for l in range(self.levels):
                cur_bins.append(offset + int(cy * self.num_bins[l]) * self.num_bins[l] + int(cx * self.num_bins[l]))
                offset += self.num_bins_sqr[l]
            #if num_boxes < 1000 and num_boxes % 100:
            #    print((box, cy, cx, cur_bins))
            inds = (confs >= 0).nonzero()[0]
            hadoopy.counter('STATS', 'num_pos', inds.size)
            hadoopy.counter('STATS', 'num_neg', confs.size - inds.size)
            hadoopy.counter('STATS', 'total', confs.size)
            if inds.size:
                for cur_bin in cur_bins:
                    pyramid[inds, cur_bin] += 1
        hadoopy.counter('STATS', 'sz-%s' % str(pyramid.shape))
        if np.any(pyramid):
            pyramid = pyramid * (self.bin_weight / float(num_boxes))
            for exemplar_num, row in enumerate(pyramid):
                yield exemplar_num, (image_id, row)


class Reducer(object):

    def __init__(self):
        pass

    def reduce(self, exemplar_num, id_rows):
        out = np.hstack([x[1].ravel()
                         for x in sorted(id_rows, key=lambda x: x[0])])
        hadoopy.counter('STATS', 'sz-%s' % str(out.shape))
        yield exemplar_num, out
        

if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer, jobconfs=['mapred.task.timeout=6000000',
                                           'mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                           'mapred.compress.map.output=true',
                                           'mapred.output.compress=true',
                                           'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
