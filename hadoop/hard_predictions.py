import numpy as np
import hadoopy
import os
import heapq
import feature
import imfeat


class Mapper(object):

    def __init__(self):
        self.max_hard = int(os.environ['MAX_HARD'])
        self.exemplar_coefs, self.exemplar_intercepts = map(np.ascontiguousarray,
                                                            zip(*[x[1] for x in hadoopy.readtb(os.environ['EXEMPLARS'])]))
        self.exemplar_preds = [[] for x in self.exemplar_coefs]

    def map(self, image_id, image_binary):
        image = imfeat.image_fromstring(image_binary)
        for box, f in feature.image_patch_features_dense(image, cell_skip=32, normalize_box=True):
            scores = np.dot(self.exemplar_coefs, f.reshape((f.size, 1))) + self.exemplar_intercepts
            pred_common = [image_id, box.tolist(), f.tolist()]
            for score, preds in zip(scores, self.exemplar_preds):
                pred = [score] + pred_common
                if len(preds) >= self.max_hard:
                    heapq.heappushpop(preds, pred)
                else:
                    heapq.heappush(preds, pred)

    def close(self):
        for x in enumerate(self.exemplar_preds):
            yield x


class Reducer(object):

    def __init__(self):
        self.max_hard = int(os.environ['MAX_HARD'])
        #neg_feats = np.vstack([x[1] for x in hadoopy.readtb(os.environ['NEG_FEATS'])])
        #self.neg_feats_one_pos = np.vstack([np.zeros_like(neg_feats[0]), neg_feats])
        #self.labels = np.hstack([np.ones(1), np.zeros(len(neg_feats))])

    def reduce(self, exemplar_num, negss):
        best_negs = []
        for negs in negss:
            for neg in negs:
                if len(negs) >= self.max_hard:
                    heapq.heappushpop(best_negs, neg)
                else:
                    heapq.heappush(best_negs, neg)
        yield exemplar_num, best_negs


if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer, jobconfs=['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                           'mapred.compress.map.output=true',
                                           'mapred.output.compress=true',
                                           'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
