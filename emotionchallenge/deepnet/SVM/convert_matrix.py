import sys,os
import glob
import numpy as np
directory = sys.argv[1]
input = glob.glob(directory+'/*.*')
#random.shuffle(input)
print(input)
print(type(input))
input = sorted(input)
print(input)


x = np.loadtxt(input[0])
print(len(x))
x = [x]
for i in xrange(1,len(input)):
	y=[np.loadtxt(input[i])]
	x=np.concatenate((x,y),axis=0)
print(x.shape)
np.savetxt(directory+'_mat.txt',x)
