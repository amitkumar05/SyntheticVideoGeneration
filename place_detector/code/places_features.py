import sys
import cv2
import numpy as np
import os
import glob
import lmdb
import random
#sys.path.insert(0, '/home/amit/Documents/caffe/python')
import caffe
from caffe.proto import caffe_pb2

input_image_file = sys.argv[1]
output_file = sys.argv[2]
print(input_image_file)
print(output_file)

# taking input from directory
type = sys.argv[2]
input=[img for img in glob.glob('../input/train/'+sys.argv[1]+'/*')]
random.shuffle(input)
print(input)

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

model_file = '/home/b13107/myproject/caffe/models/places205VGG16/snapshot_iter_765280.caffemodel'
deploy_prototxt = '/home/b13107/myproject/caffe/models/places205VGG16/deploy_10.prototxt'
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
train_samples=len(input)/4
train_samples=train_samples*3
for i in range(train_samples):
	#img = caffe.io.load_image(input_image_file)
	img = caffe.io.load_image(input[i])
	net.blobs['data'].data[...] = transformer.preprocess('data', img)
	output = net.forward()

	# np.save(open('out_put.npy', 'w'), net.blobs[layer].data[0])
	l=net.blobs[layer].data

#	print('bi')
#	print(net.blobs[layer].data)
	print(i)
	pred=output['prob']
#	print('hi')
#	print(input[i])
#	print(pred.argmax())
#	print('hi')	
#	print(l)
#	print(len(l.shape))
#	print(type(l))
#	b=np.ones((1,4096))
	features=np.concatenate((features,net.blobs[layer].data))
#	print(b)
	#features=np.concatenate(features,b)
#	with open(output_file, 'w') as f:
 #   		np.savetxt(f, net.blobs[layer].data[0], fmt='%.4f', delimiter='\n')
np.save(open('feature/'+'train'+'_'+sys.argv[1]+'.npy','w'),features)
print('started validation')
# validation data pickling starts
features=np.zeros((0,4096))
for i in xrange(train_samples,len(input)):
        #img = caffe.io.load_image(input_image_file)
        img = caffe.io.load_image(input[i])
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        output = net.forward()
	print(i)
        # np.save(open('out_put.npy', 'w'), net.blobs[layer].data[0])
        l=net.blobs[layer].data
        features=np.concatenate((features,net.blobs[layer].data))
np.save(open('feature/'+'validation'+'_'+sys.argv[1]+'.npy','w'),features)

