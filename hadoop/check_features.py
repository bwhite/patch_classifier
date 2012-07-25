import hashlib
import hadoopy
import hadoopy_helper
import cPickle as pickle
import logging
logging.basicConfig(level=logging.INFO)


def hash_features(path):
    for x, y in hadoopy.readtb(path):
        yield x, hashlib.md5(y.tostring()).hexdigest()


def main():
    dense_path = 'exemplarbank/output/1341790878.92/pos'
    image_path = 'exemplarbank/data/sun_labelme_person/1-tr'
    image_box_fns = {}
    id_box_features = list(hash_features(dense_path))
    print id_box_features[0]
    for (image_id, box), feature in id_box_features:
        image_box_fns.setdefault(image_id, []).append((box, (image_id, box)))
    with open('image_box_fns.pkl', 'w') as fp:
        pickle.dump(image_box_fns, fp, -1)
    with hadoopy_helper.hdfs_temp() as hdfs_output:
        hadoopy.launch_frozen(image_path, hdfs_output, 'clip_boxes.py', files=['image_box_fns.pkl'], remove_output=True,
                              cmdenvs=['TYPE=feature'])
        id_box_features2 = list(hash_features(hdfs_output))
        with open('compare.pkl', 'w') as fp:
            pickle.dump((id_box_features, id_box_features2), fp, -1)
main()
