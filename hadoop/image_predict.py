import patch_predict
import numpy as np
import hadoopy


class Mapper(patch_predict.Mapper):

    def __init__(self):
        super(Mapper, self).__init__()

    def map(self, image_id, image_binary):
        all_confs = []
        for (image_id, box), confs in super(Mapper, self).map(image_id, image_binary):
            all_confs.append(confs)
        print(all_confs[0].shape)
        print(len(all_confs))
        for x, y in zip(self.ids, np.vstack(all_confs).T):
            yield x, (image_id, y.ravel())


class Reducer(object):

    def __init__(self):
        pass

    def reduce(self, exemplar_id, image_id_confs):
        # TODO: Use secondary sort design pattern
        yield exemplar_id, np.hstack([x[1] for x in sorted(image_id_confs, key=lambda x: x[0])])

if __name__ == '__main__':
    hadoopy.run(Mapper, Reducer)
