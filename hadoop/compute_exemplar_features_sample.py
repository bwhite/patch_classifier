from compute_exemplar_features import Mapper
import hadoopy
import os


class Reducer(object):

    def __init__(self):
        self.max_outputs = int(os.environ['MAX_OUTPUTS'])
        self.num_output = 0

    def reduce(self, key, values):
        if self.num_output < self.max_outputs:
            self.num_output += 1
            yield key, values.next()

if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer, Reducer, jobconfs=['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec',
                                                    'mapred.compress.map.output=true',
                                                    'mapred.output.compress=true',
                                                    'mapred.output.compression.codec=org.apache.hadoop.io.compress.GzipCodec'])
