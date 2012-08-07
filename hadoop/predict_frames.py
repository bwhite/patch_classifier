import hadoopy
import logging
import numpy as np
import os
import shutil
import cPickle as pickle
import cv2
import imfeat
import matplotlib.pyplot as mp
logging.basicConfig(level=logging.INFO)

COLORS = (mp.cm.hot(np.arange(256))[:, :3][:, ::-1] * 255).astype(np.uint8)
print(COLORS.shape)

def main():
    exemplars = sorted(pickle.load(open('exemplars.pkl')), key=lambda x: x[0][2], reverse=True)[:100]
    with open('exemplars_best.pkl', 'w') as fp:
        pickle.dump(exemplars, fp, -1)
    hdfs_output = 'exemplarbank/output/%s/' % '1341790878.92'
    #hadoopy.launch_frozen('/user/brandyn/aladdin_results/keyframe/9/keyframe', hdfs_output + 'frame_pred', 'predict_video_frame.py', cmdenvs=['EXEMPLARS=exemplars_best.pkl', 'CELL_SKIP=1'], remove_output=True, files=['exemplars_best.pkl'])
    local_out = 'frame_preds/'
    try:
        shutil.rmtree(local_out)
    except OSError:
        pass
    os.makedirs(local_out)
    for num, (data, (pyramid, num_boxes)) in enumerate(hadoopy.readtb(hdfs_output + 'frame_pred')):
        if np.sum(pyramid):
            pyramid_norm = pyramid / float(num_boxes)
            pyramid_prob = np.sqrt(pyramid / float(np.max(pyramid)))
            p = np.sum(pyramid_norm)
            f = imfeat.image_fromstring(data['frame'])
            pyramid_prob_frame = cv2.resize(pyramid_prob, (f.shape[1], f.shape[0]))
            pyramid_prob_frame_color = COLORS[(pyramid_prob_frame * 255).astype(np.int), :]
            alpha = .5
            beta = alpha * pyramid_prob_frame
            beta = beta.reshape((beta.shape[0], beta.shape[1], 1))
        else:
            beta = 0.
        f = ((1 - beta) * f + beta * pyramid_prob_frame_color).astype(np.uint8)
        print(p)
        open(local_out + '%f-%d.jpg' % (p, num), 'w').write(imfeat.image_tostring(f, 'jpg'))


if __name__ == '__main__':
    main()
