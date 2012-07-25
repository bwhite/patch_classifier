import hadoopy
import numpy as np


def mapper(k, v):
    yield k, v


def reducer(k, vs):
    for v in vs:
        yield k, v

if __name__ == '__main__':
    hadoopy.run(mapper, reducer)
