# ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' video.mp4

import os,sys
import numpy as np 
import shutil
import random,math
import glob,cv2
import cv2,numpy
# from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim


add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

dir = 'ttt'
# hog = cv2.HOGDescriptor()

for i in range(245,255):
	file1 = dir+'/'+add_nulls(i,6)+'.jpg'
	# im = cv2.imread(file1)
	# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	# im = cv2.resize(im,(200,200))

# hog started

# 	im = cv2.imread(file1,0)
# 	print(im.shape)
# 	winSize = (64,64)
# 	blockSize = (16,16)
# 	blockStride = (8,8)
# 	cellSize = (8,8)
# 	nbins = 9
# 	derivAperture = 1
# 	winSigma = 4.
# 	histogramNormType = 0
# 	L2HysThreshold = 2.0000000000000001e-01
# 	gammaCorrection = 0
# 	nlevels = 64
# 	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
# 	                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
# 	#compute(img[, winStride[, padding[, locations]]]) -> descriptors
# 	winStride = (8,8)
# 	padding = (8,8)
# 	locations = ((10,20),)
# 	hist = hog.compute(im,winStride,padding,locations)




# #hog finished


#  	# h = hog.compute(im)
#  	print(hist.shape)
#  	exit(0)


#hog finished


	for j in range(245,350):
		# print(file)
		file2 = dir+'/'+add_nulls(j,6)+'.jpg'


		# img1 = cv2.imread(file1)
		# img2 = cv2.imread(file2)

		# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		# img1 = cv2.resize(img1,(200,200))
		# img2 = cv2.resize(img2,(200,200))


		img1 = cv2.imread(file1,0)
		img2 = cv2.imread(file2,0)
		# print(img1.shape,img2.shape)
		winSize = (64,64)
		blockSize = (16,16)
		blockStride = (8,8)
		cellSize = (8,8)
		nbins = 9
		derivAperture = 1
		winSigma = 4.
		histogramNormType = 0
		L2HysThreshold = 2.0000000000000001e-01
		gammaCorrection = 0
		nlevels = 64
		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
		                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
		#compute(img[, winStride[, padding[, locations]]]) -> descriptors
		winStride = (8,8)
		padding = (8,8)
		locations = ((10,20),)
		hist1 = hog.compute(img1,winStride,padding,locations)
		hist2 = hog.compute(img2,winStride,padding,locations)

		# print(img1.shape,img2.shape)
		

		dist = numpy.linalg.norm(hist1-hist2)
		s = ssim(img1,img2)


		print(file1,file2,s,dist)
		cv2.imwrite('img.png',img1)
