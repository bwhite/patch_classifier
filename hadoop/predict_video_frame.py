import patch_predict
import numpy as np
import hadoopy
from hadoopy_helper.decorators import max_inputs


class Mapper(patch_predict.Mapper):

    def __init__(self):
        super(Mapper, self).__init__()
        self.levels = 5
        self.num_bins = 2 ** self.levels

    @max_inputs(10)
    def map(self, event_video, data):
        frame = data['frame']
        data['event'] = event_video[0]
        data['video'] = event_video[1]
        pyramid = np.zeros((self.num_bins, self.num_bins), dtype=np.int32)
        num_boxes = 0
        coord = lambda x: int(np.round(x * self.num_bins))
        for (_, box), confs in super(Mapper, self).map(None, frame):
            num_boxes += 1
            cy0, cx0, cy1, cx1 = map(coord, box)
            cy1 += 1
            cx1 += 1
            cell_value = 1. / ((cy1 - cy0) * (cx1 - cx0))
            inds = (confs >= 0).nonzero()[0]
            hadoopy.counter('STATS', 'num_pos', inds.size)
            hadoopy.counter('STATS', 'num_neg', confs.size - inds.size)
            hadoopy.counter('STATS', 'total', confs.size)
            if inds.size:
                pyramid[cy0:cy1, cx0:cx1] += cell_value * inds.size
        yield data, (pyramid, num_boxes * len(self.ids))

if __name__ == '__main__':
    hadoopy.run(Mapper, jobconfs=['mapred.task.timeout=6000000',
                                  'mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                  'mapred.compress.map.output=true',
                                  'mapred.output.compress=true',
                                  'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
