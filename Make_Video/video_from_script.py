# -*- coding: utf-8 -*-
# @Author: amit
# @Date:   2017-05-22 16:32:23
# @Last Modified by:   amit
# @Last Modified time: 2017-05-22 21:42:58

import os,sys
import numpy as np 
import shutil
import random,math
import glob,cv2
import cv2,numpy
from skimage.measure import compare_ssim as ssim
from skimage import data
from skimage.feature import match_template

script_file = 'script.npy'
similarity_file = 'smaller_similarity.npy'
cafe_speaking_file = 'cafe_speaking.npy'
cafe_nonspeaking_file = 'cafe_non_speaking.npy'

# script = np.load(script_file)
script = [
			['100000','speaking','central_perk'],
			['100000','speaking','central_perk'],
			['100000','speaking','central_perk'],	
		 ]


similarity = np.load(similarity_file).item()
cafe_speaking = np.load(cafe_speaking_file).item()
cafe_nonspeaking = np.load(cafe_nonspeaking_file).item()


print('loaded all files')


#t -> [last_frame_added,people_list,speaking/non-speaking,place(central cafe)]
tmp_dir = 'tmp/'
video_dir = 'video/'
add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

def find_next_frame(end_frame,s):
	people = s[0]
	activity = s[1]
	place = s[2]
	l = similarity[end_frame][people][activity]
	b = sorted(l,key=lambda x:-x[1])
	# for i in range(len(l)):
	# take first n and then do randomization and pick one
	if(len(b) == 0):
		print('List_empty . enter some other combination of people')
		return -1
	return b[0]


delta_speaking = 35
delta_nonspeaking = 48
num_video = 2

def add_delta(start_frame,delta):
	ep = start_frame.split('_')[0]
	frame = int(start_frame.split('_')[1])+delta
	end_frame = ep+add_nulls(frame,6)
	return end_frame

def save_video(start_frame,delta):
	os.system('rm '+tmp_dir+'*')
	ep = start_frame.split('_')[0]+'/'
	frame = int(start_frame.split('_')[1])
	for i in range(delta):
		next_frame = add_nulls(frame+i,6)
		source = '../Parsed_Episodes/'+ep+next_frame+'.jpg'
		dest = tmp_dir+next_frame+'.jpg'
		# print(source,dest)
		shutil.copy(source,dest)



def gen_video(end_frame):
	for i in range(1,len(script)):
		# start_frame to find based on end_frame
		print('end_frame = ',end_frame)
		next_candidate,prob = find_next_frame(end_frame,script[i])
		if(activity=='speaking'):
			end_frame = add_delta(next_candidate,delta_speaking-1)
			save_video(next_candidate,delta_speaking)
		else:
			end_frame = add_delta(next_candidate,delta_nonspeaking-1)
			save_video(next_candidate,delta_nonspeaking)


# def find_first_frame(s):
# 	people = s[0]
# 	activity = s[1]
# 	place = s[2]
# 	for i in range(num_video):
# 		if(activity == 'speaking'):
# 			l=cafe_speaking[people]
# 			rand_index = int(np.random.rand()*len(l))
# 			start_frame = l[rand_index][0]
# 			print('i = 'start_frame)
# 			end_frame = add_delta(start_frame,delta_speaking)
# 			gen_video(end_frame)
# 		else:
# 			l=cafe_nonspeaking[people]
# 			rand_index = int(np.random.rand()*len(l))
# 			start_frame = l[rand_index]
# 			print('i = 'start_frame)
# 			end_frame = add_delta(start_frame,delta_nonspeaking)
# 			gen_video(end_frame)
# 	#convert to video

def find_first_frames(s):
	print(s)
	people = s[0]
	activity = s[1]
	place = s[2]
	for i in range(num_video):
		if(activity == 'speaking'):
			l=cafe_speaking[people]
		else:
			l=cafe_nonspeaking[people]
		rand_index = int(np.random.rand()*len(l))
		start_frame = l[rand_index][0]+'_'+l[rand_index][1]
		end_frame = l[rand_index][0]+'_'+l[rand_index][2]
		print('i = ',end_frame)
		# gen_video(end_frame)
		# save_video(start_frame,l[rand_index][2] - l[rand_index][1])


	#convert to video

find_first_frames(script[0])