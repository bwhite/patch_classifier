# TODO
# Add hard negatives so that exemplars are more descriminative, right now they are too "soft"
# Add full positive images to validation set so that the exemplars aren't so easily hacked

import picarus
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
import cPickle as pickle
import os
import shutil
import hadoopy_helper

logging.basicConfig(level=logging.INFO)

LAUNCH_HOLDER = None
def toggle_launch():
    global LAUNCH_HOLDER
    if LAUNCH_HOLDER is None:
        LAUNCH_HOLDER = hadoopy.launch, hadoopy.launch_frozen, hadoopy.launch_local
        hadoopy.launch = hadoopy.launch_frozen = hadoopy.launch_local = lambda *x, **y: {}
    else:
        hadoopy.launch, hadoopy.launch_frozen, hadoopy.launch_local = LAUNCH_HOLDER
        LAUNCH_HOLDER = None


def cleanup_image(image):
    if image is None:
        raise ValueError('Bad image')
    image = remove_tiling(image)
    # if the image is > 1024, make the largest side 1024
    if np.min(image.shape[:2]) < 2 * feature.PATCH_SIZE:
        print('Skipping [%s]' % (image.shape[:2],))
        raise ValueError('Image too small')
    max_side = 512
    if np.max(image.shape[:2]) > max_side:
        height, width = (max_side * np.array(image.shape[:2]) / np.max(image.shape[:2])).astype(np.int)
        print(image.shape)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        print('Resizing to (%d, %d)' % (height, width))
    return image


def load_data_iter(local_inputs):
    # Push labeled samples to HDFS
    unique_ids = set()
    for fn in local_inputs:
        try:
            image_data = imfeat.image_tostring(cleanup_image(cv2.imread(fn)), '.jpg')
        except (ValueError, IndexError):
            continue
        image_id = hashlib.md5(image_data).hexdigest()
        if image_id not in unique_ids:
            unique_ids.add(image_id)
            yield image_id, image_data


def setup_data(local_inputs, hdfs_input, images_per_file=2):
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
    try:
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
        while np.all(frame[:, num + 1, :] == frame[:, 0, :]):
            num += 1
        if num >= 1:
            frame = frame[:, num:, :]
    except IndexError:
        print(frame.shape)
        print(num)
        raise
    nums.append(num)
    if np.sum(nums):
        print(nums)
    return frame


def initial_train(hdfs_input, hdfs_output):
    hadoopy.launch_frozen(hdfs_input + '0-tr', hdfs_output + 'neg', 'compute_exemplar_features.py', remove_output=True)
    hadoopy.launch_frozen(hdfs_input + '1-tr', hdfs_output + 'pos', 'compute_exemplar_features.py', remove_output=True)
    # Randomly sample positives to produce an initial set of exemplars
    hadoopy.launch_frozen(hdfs_output + 'pos', hdfs_output + 'pos_sample0', 'random_uniform.py',
                          cmdenvs=['SAMPLE_SIZE=1000'], remove_output=True)
    hadoopy.launch_frozen(hdfs_output + 'pos_sample0', hdfs_output + 'pos_sample', 'split.py', num_reducers=100, remove_output=True)
    # Randomly sample negatives to produce a set that all exemplars will use initially
    hadoopy.launch_frozen(hdfs_output + 'neg', hdfs_output + 'neg_sample', 'random_uniform.py',
                          cmdenvs=['SAMPLE_SIZE=100000'], remove_output=True)
    # Train initial classifiers and serialize them
    with open('neg_feats.pkl', 'w') as fp:
        pickle.dump(np.vstack([x[1].ravel() for x in hadoopy.readtb(hdfs_output + 'neg_sample')]),
                    fp, -1)
    hadoopy.launch_frozen(hdfs_output + 'pos_sample', hdfs_output + 'exemplars-0', 'train_exemplars.py', cmdenvs=['NEG_FEATS=neg_feats.pkl'], remove_output=True)
    exemplar_out = sorted(hadoopy.readtb(hdfs_output + 'exemplars-0'), key=lambda x: x[0])
    with open('exemplars.pkl', 'w') as fp:
        pickle.dump(exemplar_out, fp, -1)


def hard_train(hdfs_input, hdfs_output):
    hadoopy.launch_frozen(hdfs_input + '0-tr', hdfs_output + 'hard_neg', 'hard_predictions.py', cmdenvs=['EXEMPLARS=exemplars.pkl',
                                                                                                         'MAX_HARD=100',
                                                                                                         'OUTPUT_FORMAT=score_image_box'],
                          num_reducers=10, files=['exemplars.pkl'], remove_output=True)

    def _inner():
        with open('image_box_fns.pkl', 'w') as fp:
            image_box_fns = {}
            for (image_id, box, score), negs in hadoopy.readtb(hdfs_output + 'hard_neg'):
                for score2, image_id2, box2 in negs:
                    image_box_fns.setdefault(image_id2, []).append((box2, [image_id, box, score]))
            pickle.dump(image_box_fns, fp, -1)
        del image_box_fns
    _inner()
    hadoopy.launch_frozen(hdfs_input + '0-tr', hdfs_output + 'hard_neg_clip', 'clip_boxes.py', files=['image_box_fns.pkl'], remove_output=True, cmdenvs=['TYPE=feature'])
    hadoopy.launch_frozen([hdfs_output + 'pos_sample',
                           hdfs_output + 'hard_neg_clip'], hdfs_output + 'exemplars-1', 'train_exemplars_hard.py',
                          cmdenvs=['NEG_FEATS=neg_feats.pkl', 'MAX_HARD=200'],
                          remove_output=True, num_reducers=10)
    exemplar_out = sorted(hadoopy.readtb(hdfs_output + 'exemplars-1'), key=lambda x: x[0])
    with open('exemplars.pkl', 'w') as fp:
        pickle.dump(exemplar_out, fp, -1)


def calibrate(hdfs_input, hdfs_output):
    # Predict on pos/neg sets
    hadoopy.launch_frozen(hdfs_input + '1-v', hdfs_output + 'val_pos', 'image_predict.py', cmdenvs=['EXEMPLARS=exemplars.pkl'], remove_output=True, num_reducers=10, files=['exemplars.pkl'])
    hadoopy.launch_frozen(hdfs_input + '0-v', hdfs_output + 'val_neg', 'image_predict.py', cmdenvs=['EXEMPLARS=exemplars.pkl'], remove_output=True, num_reducers=10, files=['exemplars.pkl'])
    # Calibrate threshold using pos/neg validation set #1
    hadoopy.launch_frozen([hdfs_output + 'val_neg', hdfs_output + 'val_pos', hdfs_output + 'exemplars-1'], hdfs_output + 'exemplars-2', 'calibrate_thresholds.py', num_reducers=100, remove_output=True)
    exemplar_out = sorted(hadoopy.readtb(hdfs_output + 'exemplars-2'), key=lambda x: x[0])
    with open('exemplars.pkl', 'w') as fp:
        pickle.dump(exemplar_out, fp, -1)


def output_exemplars(hdfs_input, hdfs_output):
    with open('image_box_fns.pkl', 'w') as fp:
        image_box_fns = {}
        for (image_id, box, score), _ in hadoopy.readtb(hdfs_output + 'exemplars-2'):
            image_box_fns.setdefault(image_id, []).append((box, 'exemplar-%.5d-%s-%s.png' % (score, image_id, box)))
        pickle.dump(image_box_fns, fp, -1)
    hadoopy.launch_frozen(hdfs_input + '1-tr', hdfs_output + 'exemplars-1-clip', 'clip_boxes.py', files=['image_box_fns.pkl'], remove_output=True, cmdenvs=['TYPE=box'])
    try:
        shutil.rmtree('exemplars')
    except OSError:
        pass
    os.makedirs('exemplars')
    for x, y in hadoopy.readtb(hdfs_output + 'exemplars-1-clip'):
        open('exemplars/%s' % (x,), 'w').write(y)


def cluster(hdfs_input, hdfs_output):
    hadoopy.launch_frozen(hdfs_input + '1-v', hdfs_output + 'val_pred_pos', 'predict_spatial_pyramid.py', cmdenvs=['EXEMPLARS=exemplars.pkl'], remove_output=True, files=['exemplars.pkl'], num_reducers=50)
    with open('labels.pkl', 'w') as fp:
        pickle.dump(list(hadoopy_helper.jobs.unique_keys(hdfs_output + 'val_pred_pos')), fp, -1)
    picarus.classify.run_compute_kernels(hdfs_output + 'val_pred_pos', hdfs_output + 'val_pred_pos_kern', 'labels.pkl', 'labels.pkl', remove_output=True, num_reducers=20, jobconfs=['mapred.child.java.opts=-Xmx256M'], cols_per_chunk=500)
    picarus.classify.run_assemble_kernels(hdfs_output + 'val_pred_pos_kern', hdfs_output + 'val_pred_pos_kern2', remove_output=True)


def workflow(hdfs_input, hdfs_output):
    initial_train(hdfs_input, hdfs_output)
    hard_train(hdfs_input, hdfs_output)
    calibrate(hdfs_input, hdfs_output)
    output_exemplars(hdfs_input, hdfs_output)
    cluster(hdfs_input, hdfs_output)


    # TODO Use the calibrated offset and current offset to determine what threshold to predict on the previous hard prediction run
    # TODO Check that the libsvm prediction is thresholded at 0
    
    # Predict on positive validation set #2, produce sparse spatial pyramid binned detections and reduce (collect rows)
    # Compute similarity matrix for sparse detections
    # Compute hierarchical clustering of similarity matrix
    pass


def exemplar_boxes(hdfs_input, hdfs_output):
    exemplar_name = 'a2bcdfa40fc25f4104899f7e84fc9667-[0.0821917808219178, 0.38961038961038963, 0.1506849315068493, 0.5194805194805195]'
    #exemplar_name = '61471dbbb05b7380f84994ba81e24fe2-[0.0, 0.0, 0.20618556701030927, 0.6779661016949152]'
    #exemplar_name = '6c140327d841f9546a957cfca5d5b557-[0.24615384615384617, 0.0, 0.4512820512820513, 0.38461538461538464]'
    st = time.time()
    exemplar_out = hadoopy.abspath(hdfs_output + 'exemplar_boxes/%s' % st) + '/'
    for kv in hadoopy.readtb(hdfs_output + 'exemplars-2'):
        (image_id, box, score), _ = kv
        if exemplar_name == '%s-%s' % (image_id, box):
            print('Found it')
            hadoopy.writetb(exemplar_out + 'exemplar', [kv])
            break
    hadoopy.launch_frozen(hdfs_input + '1-v', exemplar_out + 'val_pos', 'hard_predictions.py', cmdenvs=['EXEMPLARS=exemplars.pkl', 'MAX_HARD=200', 'OUTPUT_FORMAT=score_image_box'], files=['exemplars.pkl'],
                          num_reducers=10)
    hadoopy.launch_frozen(hdfs_input + '0-v', exemplar_out + 'val_neg', 'hard_predictions.py', cmdenvs=['EXEMPLARS=exemplars.pkl', 'MAX_HARD=200', 'OUTPUT_FORMAT=score_image_box'], files=['exemplars.pkl'],
                          num_reducers=10)
    with open('image_box_fns.pkl', 'w') as fp:
        image_box_fns = {}
        pos_boxes = [(score, image_id, box, 1) for score, image_id, box in sorted(hadoopy.readtb(exemplar_out + 'val_pos').next()[1])]
        neg_boxes = [(score, image_id, box, 0) for score, image_id, box in sorted(hadoopy.readtb(exemplar_out + 'val_neg').next()[1])]
        for num, (score, image_id, box, pol) in enumerate(sorted(pos_boxes + neg_boxes, reverse=True)):
            image_box_fns.setdefault(image_id, []).append((box, 'exemplar-%.5d-%d-%f.png' % (num, pol, score)))
        pickle.dump(image_box_fns, fp, -1)
    hadoopy.launch_frozen([hdfs_input + '1-v', hdfs_input + '0-v'], exemplar_out + 'boxes', 'clip_boxes.py', files=['image_box_fns.pkl'], remove_output=True)
    out_dir = 'exemplar-%s/' % st
    os.makedirs(out_dir)
    for x, y in hadoopy.readtb(exemplar_out + 'boxes'):
        open(out_dir + x, 'w').write(y)


def main():
    local_input, hdfs_input = '/home/brandyn/playground/sun_labelme/person/', 'exemplarbank/data/sun_labelme_person/'
    #local_input, hdfs_input = '/home/brandyn/playground/aladdin_data_cropped/person/', 'exemplarbank/data/aladdin_person/'
    neg_local_inputs = glob.glob('%s/%d/*' % (local_input, 0))
    pos_local_inputs = glob.glob('%s/%d/*' % (local_input, 1))
    random.shuffle(neg_local_inputs)
    random.shuffle(pos_local_inputs)
    print(len(neg_local_inputs))
    print(len(pos_local_inputs))
    train_ind = int(.25 * len(neg_local_inputs))
    #setup_data(neg_local_inputs[:train_ind], hdfs_input + '0-tr')
    #setup_data(neg_local_inputs[train_ind:], hdfs_input + '0-v')
    #setup_data(pos_local_inputs[:train_ind], hdfs_input + '1-tr')
    #setup_data(pos_local_inputs[train_ind:], hdfs_input + '1-v')
    hdfs_output = 'exemplarbank/output/%s/' % '1341790878.92'  # time.time()
    #exemplar_boxes(hdfs_input, hdfs_output)
    workflow(hdfs_input, hdfs_output)


if __name__ == '__main__':
    main()
