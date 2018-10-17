# ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' video.mp4

import os,sys
import numpy as np 
import shutil
import random,math
import glob,cv2
import cv2,numpy
# from skimage.measure import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim
from skimage import data
from skimage.feature import match_template


add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

dir = 'ncc'
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

	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
	template = p1
	img = p2

	method = methods[0]
	# print(method)
	# Apply template Matching
	res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
	# 4 and 5 min value
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		similar = min_val
		# print('hi1')
	else:
		similar = max_val
		# print('hi2')

	return similar


def check(k,l,img1,img2):

	x = cmp_patch(img1[10*k:10*(k+1),10*l:10*(l+1)],img2)
			# print('x = ',x)
	return x



for i in range(0,2):
	file1 = dir+'/'+add_nulls(i,6)+'.jpg'
	print(file1)
	for j in range(0,10):
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
		
		img1 = cv2.resize(img1,(200,200))
		img2 = cv2.resize(img2,(200,200))


		score = 0
		for k in range(20):
			for l in range(20):
				x =  check(k,l,img1,img2)
				# print('min = ' ,x)
				score = score+x


		print('score = ',file1,file2,score,float(score)/(400.0))
		
		


		# dist = numpy.linalg.norm(hist1-hist2)

		# cv2.imwrite('img.png',img1)
