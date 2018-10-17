'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''
import sys 
#sys.path.append('/home/ubuntu/caffe/python') 
# import caffe 

import os
import glob
import random
import numpy as np
#import Image,numpy
import cv2

import caffe
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
   # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    #img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    #img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

train_lmdb = '/home/b13107/myproject/cafecode/input/train_lmdb'
validation_lmdb = '/home/b13107/myproject/cafecode/input/validation_lmdb'

os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)


cafe_train_data = [img for img in glob.glob("../input/train/cafe/*")]
monica_train_data = [img for img in glob.glob("../input/train/monica/*")]
chandler_train_data = [img for img in glob.glob("../input/train/chandler/*")]

# test_data = [img for img in glob.glob("../input/test1/*jpg")]

#Shuffle train_data
random.shuffle(cafe_train_data)
random.shuffle(monica_train_data)
random.shuffle(chandler_train_data)

print((monica_train_data))
print(cafe_train_data)
print(chandler_train_data)
print(len(monica_train_data))


print 'Creating train_lmdb'

# in_db = lmdb.open(train_lmdb, map_size=int(1e12))
in_db = lmdb.open(train_lmdb, map_size=int(1e9))
with in_db.begin(write=True) as in_txn:
    i=0
    tmp=0
    for in_idx, img_path in enumerate(cafe_train_data):
        
	#if in_idx %  5 == 0:
        #    continue
        
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # if 'cat' in img_path:
        #     label = 0
        # else:
        #     label = 1
        label = 0
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(i+in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(i+in_idx) + ':' + img_path
        tmp=i+in_idx
    i=tmp+1

    for in_idx, img_path in enumerate(chandler_train_data):
        #if in_idx %  5 == 0:
        #    continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # if 'cat' in img_path:
        #     label = 0
        # else:
        #     label = 1
        label = 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(i+in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(i+in_idx) + ':' + img_path
	tmp=i+in_idx
    i=tmp+1


    for in_idx, img_path in enumerate(monica_train_data):
        #if in_idx %  5 == 0:
        #    continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # if 'cat' in img_path:
        #     label = 0
        # else:
        #     label = 1
        label = 2
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(i+in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(i+in_idx) + ':' + img_path
	

in_db.close()

# done training
# print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e9))
# in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    i=0
    tmp=0
    for in_idx, img_path in enumerate(cafe_train_data):
        tmp=i+in_idx
        if in_idx %  4 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # if 'cat' in img_path:
        #     label = 0
        # else:
        #     label = 1
        label = 0
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(i+in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(i+in_idx) + ':' + img_path
    i=tmp+1

    for in_idx, img_path in enumerate(chandler_train_data):
        tmp=i+in_idx
	if in_idx %  4 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # if 'cat' in img_path:
        #     label = 0
        # else:
        #     label = 1
        label = 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(i+in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(i+in_idx) + ':' + img_path
    i=tmp+1

    for in_idx, img_path in enumerate(cafe_train_data):
        tmp=i+in_idx
        if in_idx %  4 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        # if 'cat' in img_path:
        #     label = 0
        # else:
        #     label = 1
        label = 2
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(i+in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(i+in_idx) + ':' + img_path
    i=tmp+1


#        j = Image.fromarray(img.astype(numpy.uint8))
 #       j.save('img2.png')

in_db.close()




# in_db = lmdb.open(train_lmdb, map_size=int(1e9))
# print(len(in_db))
# print '\nCreating validation_lmdb'

# in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
# with in_db.begin(write=True) as in_txn:
#     for in_idx, img_path in enumerate(train_data):
#         if in_idx % 6 != 0:
#             continue
#         img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#         img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
#         if 'cat' in img_path:
#             label = 0
#         else:
#             label = 1
#         datum = make_datum(img, label)
#         in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
#         print '{:0>5d}'.format(in_idx) + ':' + img_path
# in_db.close()

print '\nFinished processing all images'
