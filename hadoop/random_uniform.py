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

    def values(self):
        return self._key_values.itervalues()


class Mapper(object):

    def __init__(self):
        self._pq = LeakyPriorityQueue(int(os.environ['SAMPLE_SIZE']))

    def map(self, key, value):
        """Take in points and output a random sample

        Args:
            key: Opaque (can be anything)
            value: Opaque (can be anything)
        """
        self._pq.add(random.random(), (key, value))

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

    def reduce(self, rand, key_vals):
        """Take in a series of points and output a random sample

        Args:
            rand: Random float
            key_vals: Iterator of (Input key, Input value)

        Yields:
            A tuple in the form of (key, value)
            key: Input key
            value: Input value
        """
        for kv in key_vals:
            self._pq.add(rand, kv)

    def close(self):
        return self._pq.values()
        

if __name__ == "__main__":
    hadoopy.run(Mapper, Reducer)
