import hadoopy
import imfeat

for z, (fn, image_binary) in enumerate(hadoopy.readtb('/user/brandyn/alggen_data/sun397/train/')):
    print(fn)
    open('sun-%.5d.jpg' % z, 'w').write(image_binary)
