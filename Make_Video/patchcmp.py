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


def f(pat):
	avg = 0
	# print(pat)
	for i in range(pat.shape[0]):
		for j in range(pat.shape[1]):
			avg = avg+pat[i][j]
	# return avg/(pat.shape[0]*pat.shape[1])
	return avg
	# print('size = ',pat.shape)
	# return 1


def cmp_patch(p1,p2):
	x1 = f(p1)
	x2 = f(p2)
	# print(x1,x2)
	return abs(x1-x2)


def check(k,l,img1,img2):

	m = 1e10
	for i in range(-1,2):
		for j in range(-1,2):
			if(i == 0 and j==0):
				continue
			x = cmp_patch(img1[10*k:10*(k+1),10*l:10*(l+1)],img2[10*(k+i):10*(k+i+1),10*(l+j):10*(l+j+1)])
			# print('x = ',x)
			m=min(x,m)
	return m

def his_cmp(img1,img2):
	# hist1 = cv2.calcHist([img1],[0],None,[25],[0,256])
	# hist2 = cv2.calcHist([img2],[0],None,[25],[0,256])
	# print(hist)
	hist1 = cv2.calcHist([img1], [0, 1, 2], None, [10, 10, 10],
		[0, 256, 0, 256, 0, 256])
	hist1 = cv2.normalize(hist1).flatten()
	print('shape = ',hist1.shape)

	hist2 = cv2.calcHist([img2], [0, 1, 2], None, [10, 10, 10],
		[0, 256, 0, 256, 0, 256])
	hist2 = cv2.normalize(hist2).flatten()
	# print(hist)
	# print(hist.shape)
	# dist = numpy.linalg.norm(hist1-hist2)
	d = cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_CORREL)
	return d
	# exit(0)

for i in range(245,300,5):
	file1 = dir+'/'+add_nulls(i,6)+'.jpg'
	print(file1)
	for j in range(245,500):
		file2 = dir+'/'+add_nulls(j,6)+'.jpg'


		# img1 = cv2.imread(file1)
		# img2 = cv2.imread(file2)

		# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

		# img1 = cv2.resize(img1,(200,200))
		# img2 = cv2.resize(img2,(200,200))


		img1 = cv2.imread(file1)
		img2 = cv2.imread(file2)
		# print(img1.shape,img2.shape)
		
		img1 = cv2.resize(img1,(200,200))
		img2 = cv2.resize(img2,(200,200))


		score = 0



		# for k in range(20):
		# 	for l in range(20):
		# 		x =  check(k,l,img1,img2)
		# 		# print('min = ' ,x)
		# 		score = score+x

		score = his_cmp(img1,img2)
		print('score = ',file1,file2,score,float(score)/(256.0*400.0))
		
		


		# dist = numpy.linalg.norm(hist1-hist2)

		# cv2.imwrite('img.png',img1)
