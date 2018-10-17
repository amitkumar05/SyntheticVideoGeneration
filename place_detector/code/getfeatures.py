import sys
import cv2
import numpy as np
import os
import glob
import lmdb
sys.path.insert(0, '/home/amit/Documents/caffe/python')
import caffe
from caffe.proto import caffe_pb2

input_image_file = sys.argv[1]
output_file = sys.argv[2]
print(input_image_file)
print(output_file)


#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/home/amit/cafecode/input/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

print('read mean image')

model_file = '/home/amit/Documents/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
deploy_prototxt = '/home/amit/Documents/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
net = caffe.Net(deploy_prototxt, model_file, caffe.TEST)
layer = 'fc6'
if layer not in net.blobs:
    raise TypeError("Invalid layer name: " + layer)



#
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))
transformer.set_raw_scale('data', 255.0)

net.blobs['data'].reshape(1,3,227,227)

img = caffe.io.load_image(input_image_file)

net.blobs['data'].data[...] = transformer.preprocess('data', img)

output = net.forward()

np.save(open('out_put.npy', 'w'), net.blobs[layer].data[0])

with open(output_file, 'w') as f:
    np.savetxt(f, net.blobs[layer].data[0], fmt='%.4f', delimiter='\n')
