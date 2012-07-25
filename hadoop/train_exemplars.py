import numpy as np
import hadoopy
import sklearn.svm
import os
import cPickle as pickle


class Mapper(object):

    def __init__(self):
        neg_feats = pickle.load(open(os.environ['NEG_FEATS']))
        self.neg_feats_one_pos = np.vstack([np.zeros_like(neg_feats[0]), neg_feats])
        self.labels = np.hstack([np.ones(1), np.zeros(len(neg_feats))])

    def map(self, image_id_box, feature):
        self.neg_feats_one_pos[0, :] = feature
        c = sklearn.svm.LinearSVC(verbose=10, C=10**20).fit(self.neg_feats_one_pos, self.labels)
        yield [image_id_box[0], image_id_box[1], 0.], (c.coef_.ravel(), float(c.intercept_[0]))


if __name__ == '__main__':
    hadoopy.run(Mapper)
