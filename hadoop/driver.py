# TODO
# Compute validation predictions using exemplars
# Score exemplars and compute thresholds
# Output top matches per exemplar


import hadoopy
import imfeat
import numpy as np
import cv2
import hashlib
import glob
import time
import logging
import feature
import sklearn
import random

logging.basicConfig(level=logging.INFO)


def cleanup_image(image):
    image = remove_tiling(image)
    # if the image is > 1024, make the largest side 1024
    max_side = 1024
    if np.max(image.shape[:2]) > max_side:
        height, width = (max_side * np.array(image.shape[:2]) / np.max(image.shape[:2])).astype(np.int)
        image = cv2.resize(image, (width, height))
        print('Resizing to (%d, %d)' % (height, width))
    print(image.shape)
    print(image.flags)
    return image


def load_data_iter(local_inputs):
    # Push labeled samples to HDFS
    unique_ids = set()
    for fn in local_inputs:
        image_data = imfeat.image_tostring(cleanup_image(cv2.imread(fn)), '.jpg')
        image_id = hashlib.md5(image_data).hexdigest()
        if image_id not in unique_ids:
            unique_ids.add(image_id)
            yield image_id, image_data


def setup_data(local_inputs, hdfs_input, images_per_file=10):
    cnt = 0
    out = []
    for x in load_data_iter(local_inputs):
        out.append(x)
        if len(out) > images_per_file:
            hadoopy.writetb(hdfs_input + '/%d' % cnt, out)
            cnt += 1
            out = []
    if out:
        hadoopy.writetb(hdfs_input + '/%d' % cnt, out)


def remove_tiling(frame):
    nums = []
    # Bottom
    num = 0
    while np.all(frame[-(num + 2), :, :] == frame[-1, :, :]):
        num += 1
    if num >= 1:
        frame = frame[:-(num + 1), :, :]
    nums.append(num)
    # Top
    num = 0
    while np.all(frame[num + 1, :, :] == frame[0, :, :]):
        num += 1
    if num >= 1:
        frame = frame[num:, :, :]
    nums.append(num)
    # Right
    num = 0
    while np.all(frame[:, -(num + 2), :] == frame[:, -1, :]):
        num += 1
    if num >= 1:
        frame = frame[:, :-(num + 1), :]
    nums.append(num)
    # Left
    num = 0
    while np.all(frame[:, num + 1, :] == frame[: 0, :]):
        num += 1
    if num >= 1:
        frame = frame[:, num:, :]
    nums.append(num)
    if np.sum(nums):
        print(nums)
    return frame




def workflow(hdfs_input, hdfs_output):
    # TRAINING
    # Compute random negative feature samples (enough to easily fit in memory)
    hadoopy.launch_frozen(hdfs_input + '0-tr', hdfs_output + 'neg', 'compute_exemplar_features.py')
    # Compute random positive exemplars (enough to easily fit in memory)  (TODO: Next step is to prefilter at this point)
    hadoopy.launch_frozen(hdfs_input + '1-tr', hdfs_output + 'pos', 'compute_exemplar_features.py')
    # Randomly sample positives to produce an initial set of exemplars
    hadoopy.launch_frozen(hdfs_output + 'pos', hdfs_output + 'pos_sample', 'random_uniform.py', cmdenvs=['SAMPLE_SIZE=1000'])
    # Randomly sample negatives to produce a set that all exemplars will use initially
    hadoopy.launch_frozen(hdfs_output + 'neg', hdfs_output + 'neg_sample', 'random_uniform.py', cmdenvs=['SAMPLE_SIZE=10000'])

    # Train initial classifiers and serialize them
    hadoopy.launch_frozen(hdfs_output + 'pos_sample', hdfs_output + 'exemplars-0', 'train_exemplars.py', cmdenvs=['NEG_FEATS=%s' % (hadoopy.abspath(hdfs_output + 'neg_sample'))])
    
    # Train initial classifier using sampled positives and negatives, find hard negatives for each positive
    #hadoopy.launch_frozen(hdfs_input + '0-v', hdfs_output + 'exemplars-14', 'train_exemplars_hard.py', cmdenvs=['MAX_HARD=1000',
    #'EXEMPLARS=%s' % hadoopy.abspath(hdfs_output + 'exemplars-0')], num_reducers=10)
    # CALIBRATION
    # Predict on pos/neg sets
    hadoopy.launch_frozen(hdfs_output + '0-v', hdfs_output + 'exemplars-0', 'hard_predictions.py', cmdenvs=['NEG_FEATS=%s' % (hadoopy.abspath(hdfs_output + 'neg_sample'))])
    hadoopy.launch_frozen(hdfs_output + '0-v', hdfs_output + 'exemplars-0', 'hard_predictions.py', cmdenvs=['NEG_FEATS=%s' % (hadoopy.abspath(hdfs_output + 'neg_sample'))])
    # Calibrate threshold using pos/neg validation set #1

    # CLUSTERING
    # Predict on positive validation set #2, produce sparse spatial pyramid binned detections and reduce (collect rows)
    # Compute similarity matrix for sparse detections
    # Compute hierarchical clustering of similarity matrix
    pass


def main():
    local_input, hdfs_input = '/home/brandyn/playground/aladdin_data_cropped/person/', 'exemplarbank/data/aladdin_person_classes/'
    neg_local_inputs = glob.glob('%s/%d/*' % (local_input, 0))
    pos_local_inputs = glob.glob('%s/%d/*' % (local_input, 1))
    random.shuffle(neg_local_inputs)
    random.shuffle(pos_local_inputs)
    print(len(neg_local_inputs))
    print(len(pos_local_inputs))
    train_ind = int(.5 * len(neg_local_inputs))
    setup_data(neg_local_inputs[:train_ind], hdfs_input + '0-tr')
    setup_data(neg_local_inputs[train_ind:], hdfs_input + '0-v')
    train_ind = int(.5 * len(pos_local_inputs))
    setup_data(pos_local_inputs[:train_ind], hdfs_input + '1-tr')
    setup_data(pos_local_inputs[train_ind:], hdfs_input + '1-v')
    
    #setup_data(local_input + '0', hdfs_input + '0')
    #setup_data(local_input + '1', hdfs_input + '1')
    hdfs_output = 'exemplarbank/output/%s/' % '1340929620.410318'
    workflow(hdfs_input, hdfs_output)


if __name__ == '__main__':
    main()
