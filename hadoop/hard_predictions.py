import numpy as np
import hadoopy
import os
import heapq
import feature
import imfeat
import time
    

class Mapper(object):

    def __init__(self):
        self.max_hard = int(os.environ['MAX_HARD'])
        self.exemplar_coefs, self.exemplar_intercepts = map(np.ascontiguousarray,
                                                            zip(*[x[1] for x in hadoopy.readtb(os.environ['EXEMPLARS'])]))
        self.exemplar_preds = [[] for x in self.exemplar_coefs]
        self.output_formatter = lambda x: x[0]
        print('NumExemplars[%d]' % len(self.exemplar_preds))

    def map(self, image_id, image_binary):
        image = imfeat.image_fromstring(image_binary)
        print(image.shape)
        st = time.time()
        for box, f in feature.image_patch_features_dense(image, normalize_box=True):
            scores = np.dot(self.exemplar_coefs, f.reshape((f.size, 1))) + self.exemplar_intercepts
            pred_common = [image_id, box.tolist(), f.tolist()]
            for score, preds in zip(scores, self.exemplar_preds):
                pred = self.output_formatter([float(score[0])] + pred_common)
                if len(preds) >= self.max_hard:
                    heapq.heappushpop(preds, pred)
                else:
                    heapq.heappush(preds, pred)
        print('ImageTime[%f]' % (time.time() - st))

    def close(self):
        for x in enumerate(self.exemplar_preds):
            yield x


class Reducer(object):

    def __init__(self):
        self.max_hard = int(os.environ['MAX_HARD'])

    def reduce(self, exemplar_num, predss):
        best_preds = []
        for preds in predss:
            for pred in preds:
                if len(best_preds) >= self.max_hard:
                    heapq.heappushpop(best_preds, pred)
                else:
                    heapq.heappush(best_preds, pred)
        yield exemplar_num, best_preds


if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer, jobconfs=['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                           'mapred.compress.map.output=true',
                                           'mapred.output.compress=true',
                                           'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
