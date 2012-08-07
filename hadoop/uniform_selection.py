#!/usr/bin/env python
# (C) Copyright 2012 Brandyn A. White
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Random sampling of key/value pairs
"""

__author__ = 'Brandyn A. White <bwhite@cs.umd.edu>'
__license__ = 'GPL V3'

import hadoopy
import os
import bisect
import random
import cPickle as pickle
import train_exemplars
import calibrate_thresholds
try:
    # We often need these to be frozen along with the package
    import numpy as np
except ImportError:
    pass


class LeakyPriorityQueue(object):

    def __init__(self, capacity):
        self._sorted_keys = [float('inf')]  # (random)
        self._key_values = {}  # [key] = item
        self._capacity = capacity

    def add(self, key, item):
        if key < self._sorted_keys[-1]:
            bisect.insort(self._sorted_keys, key)
            self._key_values[key] = item
        if len(self._sorted_keys) > self._capacity:
            try:
                del self._key_values[self._sorted_keys[-1]]
            except KeyError:  # Ignore inf
                pass
            self._sorted_keys = self._sorted_keys[:-1]

    def items(self):
        return self._key_values.iteritems()

    def items_sorted(self):
        for x in self._sorted_keys:
            try:
                yield x, self._key_values[x]
            except KeyError:
                pass

    def values(self):
        return self._key_values.itervalues()


class Mapper(train_exemplars.Mapper):

    def __init__(self):
        self._pq = LeakyPriorityQueue(int(os.environ['SAMPLE_SIZE']))
        self._neg_feats = np.vstack(pickle.load(open(os.environ['NEG_VAL_FEATS'])))
        self._pos_feats = np.vstack(pickle.load(open(os.environ['POS_VAL_FEATS'])))
        super(Mapper, self).__init__()

    def map(self, exemplar_id, feature):
        """Take in points and output a random sample

        Args:
            key: Exemplar ID
            value: feature
        """
        (image_id, box, _), (coef, intercept) = super(Mapper, self).map(exemplar_id, feature).next()
        pred = lambda f: np.dot(f, coef).ravel() + intercept
        print(self._neg_feats.shape)
        print(coef.shape)
        print(intercept)
        neg_pred = pred(self._neg_feats)
        pos_pred = pred(self._pos_feats)
        thresh, score = calibrate_thresholds.fpr_threshold(pos_pred, neg_pred)
        # Scores are integral, but they need to be unique, add a random decimal
        self._pq.add(-score + random.random(), ([image_id, box, score], (coef, intercept - thresh)))

    def close(self):
        """
        Yields:
            A tuple in the form of (key, value)
            key: Random float
            value: (Input key, Input value)
        """
        return self._pq.items()


class Reducer(object):

    def __init__(self):
        self._pq = LeakyPriorityQueue(int(os.environ['SAMPLE_SIZE']))

    def reduce(self, score, key_vals):
        """

        Args:
            score:
            key_vals: Iterator of (Input key, Input value)

        Yields:
            A tuple in the form of (key, value)
            key: Input key
            value: Input value
        """
        for kv in key_vals:
            self._pq.add(score, kv)

    def close(self):
        return self._pq.values()

if __name__ == "__main__":
    hadoopy.run(Mapper, Reducer)
