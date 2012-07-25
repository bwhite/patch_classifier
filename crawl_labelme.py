import vision_data
import cv2
import numpy as np

for z, (x, y) in enumerate(vision_data.LabelMe().object_rec_boxes(objects=['blurry', 'blur'])):
    if np.min(y.shape[:2]) < 50:
        continue
    print(z)
    cv2.imwrite('labelme-%.5d-%s.png' % (z, x), y)
