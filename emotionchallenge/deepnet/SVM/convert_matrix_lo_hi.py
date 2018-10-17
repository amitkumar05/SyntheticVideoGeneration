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

lo = 0
hi = 8000

lo = 8000
hi = 15969

x = np.loadtxt(input[lo])
print(len(x))
x = [x]
for i in xrange(lo+1,hi):
	y=[np.loadtxt(input[i])]
	x=np.concatenate((x,y),axis=0)
print(x.shape)
np.savetxt(directory+'_mat_test.txt',x)
