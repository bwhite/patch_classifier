import vision_data
import cv2

for z, (x, y) in enumerate(vision_data.LabelMe().object_rec_boxes(objects=['person', 'people', 'human', 'man',
                                                                           'woman', 'pedestrian', 'boy', 'girl', 'baby', 'kid'])):
    print(z)
    cv2.imwrite('/home/brandyn/playground/aladdin_data_cropped/person/1/labelme-%.5d-%s.png' % (z, x), y)
