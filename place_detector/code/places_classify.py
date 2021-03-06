import sys
import cv2
import numpy as np
import os
import glob
import lmdb
import random

#sys.path.insert(0, '/home/amit/Documents/caffe/python')

sys.path.insert(0, '/home/aswin/tools/caffe/python')
sys.path.append('/home/aswin/tools/anaconda2')

import caffe
from caffe.proto import caffe_pb2
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.externals import joblib

model_file = 'model_svm_linear.pkl'
save_file = 'new_out.txt'



input_image_file = sys.argv[1]
print(input_image_file)

# taking input from directory
#type = sys.argv[2]
input=[img for img in glob.glob('../input/test/*')]
random.shuffle(input)
print(type(input))

#exit(0)

#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('../input/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

print('read mean image')
print(mean_array)
# exit(0)

model_file = '/home/b13218/caffe/models/places205VGG16/snapshot_iter_765280.caffemodel'
deploy_prototxt = '/home/b13218/caffe/models/places205VGG16/deploy_10.prototxt'
net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
layer = 'fc6'
if layer not in net.blobs:
    raise TypeError("Invalid layer name: " + layer)

#
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1,3,224,224)
features=np.zeros((0,4096))
print(features)
for i in range(len(input)):
	#img = caffe.io.load_image(input_image_file)
	img = caffe.io.load_image(input[i])
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	output = net.forward()

	# np.save(open('out_put.npy', 'w'), net.blobs[layer].data[0])
	l=net.blobs[layer].data

	print(input[i])
#	print(net.blobs[layer].data)
	print(i)
#	print(len(l.shape))
#	print(type(l))
#	b=np.ones((1,4096))
	features=np.concatenate((features,net.blobs[layer].data))
#	print(b)
	#features=np.concatenate(features,b)
#	with open(output_file, 'w') as f:
 #   		np.savetxt(f, net.blobs[layer].data[0], fmt='%.4f', delimiter='\n')


# np.save(open('feature/'+'test.npy','w'),features)
# np.array(input).dump(open('feature/'+'test_images.npy','w'))

# svm loading
print('Model Loading')
clf = joblib.load(model_file)
print('Model Loaded')
Y = clf.predict(features)
print(Y)
print(type(Y))

print('byeeeeeee')
np.savetxt(save_file,Y,fmt="%s")
