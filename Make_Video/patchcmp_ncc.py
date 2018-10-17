import os,sys
import numpy as np 
import shutil
import random,math
import glob,cv2
import cv2,numpy
from skimage.measure import compare_ssim as ssim
from skimage import data
from skimage.feature import match_template

add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)
# dir = 'ncc'

def f(pat):
	avg = 0
	for i in range(pat.shape[0]):
		for j in range(pat.shape[1]):
			avg = avg+pat[i][j]
	return avg

def cmp_patch(p1,p2):

	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
	template = p1
	img = p2
	method = methods[0]
	res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		similar = min_val
	else:
		similar = max_val
	return similar

def check(k,l,img1,img2):
	x = cmp_patch(img1[10*k:10*(k+1),10*l:10*(l+1)],img2)
	return x

def gen_path(file):
	f=file.split('_')
	frame = str(add_nulls(int(f[1]),6))
	return ('../Parsed_Episodes/'+f[0]+'/'+frame)

def find_sim(file1,file2):
	file1 = gen_path(file1)
	file2 = gen_path(file2)
	print(file1,file2)
	# exit(0)
	return 0
	img1 = cv2.imread(file1,0)
	img2 = cv2.imread(file2,0)
	img1 = cv2.resize(img1,(200,200))
	img2 = cv2.resize(img2,(200,200))

	score = 0
	for k in range(20):
		for l in range(20):
			x =  check(k,l,img1,img2)
			score = score+x

	print('score = ',file1,file2,score,float(score)/(400.0))
	return score

non_speaking = np.load('cafe_nonspeaking.npy').item()
speaking = np.load('cafe_speaking.npy').item()
# print(speaking)
# print(len(speaking),len(non_speaking))
# speaking_img_dict = {}
# nonspeaking_img_dict = {}

# for k in non_speaking.keys():
# 	l = non_speaking[k]
# 	for i in range(len(l)):
# 		folder = l[i][0]
# 		img_no = str(l[i][1])
# 		img = folder+'_'+img_no
# 		nonspeaking_img_dict[img] = l[i][3]


# for k in speaking.keys():
# 	l = speaking[k]
# 	for i in range(len(l)):
# 		folder = l[i][0]
# 		img_no = str(l[i][1])
# 		img = folder+'_'+img_no
# 		speaking_img_dict[img] = l[i][3]

# print(len(speaking_img_dict),len(nonspeaking_img_dict))

# np.save('speaking_img_dict.npy',speaking_img_dict)
# np.save('nonspeaking_img_dict.npy',nonspeaking_img_dict)

nonspeaking_img_dict = np.load('nonspeaking_img_dict.npy').item()
speaking_img_dict = np.load('speaking_img_dict.npy').item()

# print(len(speaking_img_dict),len(nonspeaking_img_dict.keys()))




# x = np.zeros((10000,10000))
# np.save('tmp.npy',x)
d={}
speaking_skip = 34
nonspeaking_skip = 47

for img1 in speaking_img_dict.keys():
	k = img1.split('_')
	ep1 = k[0]
	frame1 = int(k[1])+speaking_skip
	# print(ep1,frame1)
	end_frame = ep1+'_'+str(frame1)
	d[end_frame] = {}

	for key in non_speaking.keys():
		l = non_speaking[key]
		d[end_frame][key] = {}
		all_list = []
		for i in range(len(l)):
			folder = l[i][0]
			img_no = str(l[i][1])
			img2 = folder+'_'+img_no
			similarity = find_sim(img1,img2)
			all_list.append([img2,similarity])

		d[end_frame][key]['non_speaking']=all_list

	# doing for non_speaking
	
	for key in speaking.keys():
		l = speaking[key]
		all_list = []
		for i in range(len(l)):
			folder = l[i][0]
			img_no = str(l[i][1])
			img2 = folder+'_'+img_no
			similarity = find_sim(img1,img2)
			all_list.append([img2,similarity])

		d[end_frame][key]['speaking']=all_list

	np.save('tmp.npy',d)
	exit(0)



for img1 in nonspeaking_img_dict.keys():
	k = img1.split('_')
	ep1 = k[0]
	frame1 = int(k[1])+nonspeaking_skip
	# print(ep1,frame1)
	end_frame = ep1+'_'+str(frame1)
	d[end_frame] = {}

	for key in non_speaking.keys():
		l = non_speaking[key]
		d[end_frame][key] = {}
		all_list = []
		for i in range(len(l)):
			folder = l[i][0]
			img_no = str(l[i][1])
			img2 = folder+'_'+img_no
			similarity = find_sim(img1,img2)
			all_list.append([img2,similarity])

		d[end_frame][key]['non_speaking']=all_list

	# print(d[end_frame].keys())
	# exit(0)
	# doing for non_speaking
	
	for key in speaking.keys():
		l = speaking[key]
		all_list = []
		for i in range(len(l)):
			folder = l[i][0]
			img_no = str(l[i][1])
			img2 = folder+'_'+img_no
			similarity = find_sim(img1,img2)
			all_list.append([img2,similarity])

		d[end_frame][key]['speaking']=all_list


	np.save('tmp.npy',d)


	exit(0)

