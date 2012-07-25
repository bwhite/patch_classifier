import numpy as np
import hadoopy
import sklearn.svm
import os
import cPickle as pickle


class Mapper(object):

    def __init__(self):
        pass

    def map(self, key, value):
        """
        If a hard negative feature
        key: exemplar_id
        value: (score, feature)

        If an exemplar positive feature
        key: exemplar_id
        value: feature
        """
        input_file = os.environ['map_input_file']
        if input_file.find('hard_neg_clip') != -1:
            yield list(key)[:2], (0, value)
        else:
            yield list(key)[:2], (1, value)


class Reducer(object):

    def __init__(self):
        neg_feats = pickle.load(open(os.environ['NEG_FEATS']))
        self.num_neg_feats = len(neg_feats)
        self.max_hard = int(os.environ['MAX_HARD'])
        self.neg_feats_one_pos = np.zeros((neg_feats.shape[0] + 1 + self.max_hard, neg_feats.shape[1]))
        self.neg_feats_one_pos[1:len(neg_feats) + 1, :] = neg_feats
        self.labels = np.hstack([np.ones(1), np.zeros(len(neg_feats) + self.max_hard)])

    def reduce(self, key, values):
        """
        If a hard negative feature
        key: exemplar_id
        value: (0, feature)

        If an exemplar positive feature
        key: exemplar_id
        value: (1, feature)
        """
        hard_feats = 0
        pos_feat = 0
        for val_type, value in values:
            if val_type:  # Positive exemplar
                assert pos_feat == 0
                self.neg_feats_one_pos[0, :] = value
                pos_feat += 1
            else:
                assert hard_feats < self.max_hard
                self.neg_feats_one_pos[1 + self.num_neg_feats + hard_feats] = value[-1]
                hard_feats += 1
        print('[%r]PosFeat[%d] HardFeats[%d] NegFeats[%d]' % (key, pos_feat, hard_feats, self.num_neg_feats))
        assert pos_feat
        num_feats = 1 + self.num_neg_feats + hard_feats
        c = sklearn.svm.LinearSVC(verbose=10, C=10**20).fit(self.neg_feats_one_pos[:num_feats, :], self.labels[:num_feats])
        yield [key[0], key[1], 0.], (c.coef_.ravel(), float(c.intercept_[0]))


if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer)
