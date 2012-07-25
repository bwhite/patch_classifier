import numpy as np
import hadoopy
import os
import heapq
import feature
import imfeat
import time
import cPickle as pickle
    

class Mapper(object):

    def __init__(self):
        self.max_hard = int(os.environ['MAX_HARD'])
        self.ids, self.coefs, self.intercepts = zip(*[(x, y, z) for x, (y, z) in pickle.load(open(os.environ['EXEMPLARS']))])
        #self.ids, self.coefs, self.intercepts = zip(*[(x, y, z) for x, (y, z) in hadoopy.readtb(os.environ['EXEMPLARS'])])
        self.coefs = np.ascontiguousarray(self.coefs)
        self.intercepts = np.ascontiguousarray(self.intercepts)
        self.preds = [[] for x in self.coefs]
        output_format = os.environ.get('OUTPUT_FORMAT', 'score')
        if output_format == 'score':
            self.output_formatter = lambda x: x[0]
        elif output_format == 'score_feat':
            self.output_formatter = lambda x: x[0:] + x[3:]
        elif output_format == 'score_image_box':
            self.output_formatter = lambda x: x[:3]
        else:
            raise ValueError('Unknown OUTPUT_FORMAT[%s]' % output_format)
        print('NumExemplars[%d]' % len(self.preds))

    def map(self, image_id, image_binary):
        image = imfeat.image_fromstring(image_binary)
        print(image.shape)
        st = time.time()
        box_num = -1
        for box_num, (box, f) in enumerate(feature.image_patch_features_dense(image, normalize_box=True)):
            scores = np.dot(self.coefs, f.reshape((f.size, 1))) + self.intercepts
            pred_common = [image_id, box.tolist(), f.tolist()]
            for score, preds in zip(scores, self.preds):
                pred = self.output_formatter([float(score[0])] + pred_common)
                if len(preds) >= self.max_hard:
                    heapq.heappushpop(preds, pred)
                else:
                    heapq.heappush(preds, pred)
        hadoopy.counter('stats', 'num_boxes', box_num + 1)
        print('ImageTime[%f]' % (time.time() - st))

    def close(self):
        for x in zip(self.ids, self.preds):
            yield x


class Reducer(object):

    def __init__(self):
        self.max_hard = int(os.environ['MAX_HARD'])

    def reduce(self, exemplar_id, predss):
        best_preds = []
        for preds in predss:
            for pred in preds:
                if len(best_preds) >= self.max_hard:
                    heapq.heappushpop(best_preds, pred)
                else:
                    heapq.heappush(best_preds, pred)
        yield exemplar_id, best_preds


if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer, jobconfs=['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                           'mapred.compress.map.output=true',
                                           'mapred.output.compress=true',
                                           'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                           'mapred.task.timeout=6000000'])
