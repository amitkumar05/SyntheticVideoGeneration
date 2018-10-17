# ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' video.mp4

import os,sys
import numpy as np 
import shutil
import random,math
from math import log10 as log


l = 0  # stands for the place
v_len = 48	# length of video sequence
z=30000		# mx numbers of frames

people = [1,0,0,0,0,0]
thres = math.pow(0.2,v_len)

num_of_vid = 2
num_videos = 29000

face_file = 'results_face_recogn/s07e03.npy'
place_file = 'Results_Episodes/s07e03.npy'


face = np.load(face_file).item()
place = np.load(place_file).item()['place_prob']

add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

def find_max(startframe_prob,z):
	index = 0
	m = 1e-300

	for i in range(1,z):
		if(startframe_prob[i] > startframe_prob[index]):
			index = i

	return index

def copy_frames(index,v_len):
	os.system('rm tmp3/*')
	for i in range(index+1,index+1+v_len):
			source = add_nulls(i,6)+'.jpg'
			dest = 'tmp3/'+source
			source = 's07e03/'+source
			print(source,dest)
			shutil.copy(source,dest)

def prob_change(total_prob,index,v_len):
	for i in range(index-v_len,index+v_len):
		total_prob[i] = 0
	return total_prob


def max_prob(faces,c):
	n = faces[0]
	m = 0.001 					# changed to 0.001 from 0.01
	for i in range(n):
		m = max(m,faces[1][i][4+c])
	return m


def lg(x):
	x = x*1000
	if(x < 1):
		x = 1
	return log(x)

# print(place[4564])
# print(face['000100.jpeg'])
# for i in range(1000,2000):
# 	print(i,place[i])
# exit(0)


startframe_prob = np.zeros(len(place)-v_len)

for i in range(v_len):
	startframe_prob[0]=startframe_prob[0]*place[i][l]
# for i in range(1,len(place)-vlen):
for i in range(1,z):
	startframe_prob[i]=(startframe_prob[i-1]*place[i+v_len-1][l])/place[i-1][l]
	# r = add_nulls(i,6)
	# print(r+'.jpg',startframe_prob[i])


total_prob = np.zeros(len(place)-v_len)

for i in range(v_len):

	total_prob[0]=total_prob[0]+place[i][l]
# for i in range(1,len(place)-vlen):
for i in range(1,z):
	total_prob[i]=(total_prob[i-1]*place[i+v_len-1][l])/place[i-1][l]
	



# people starts and place finishes

total_prob = np.zeros((len(place)-v_len,6))
for i in range(v_len):
	img = add_nulls(i+1,6)+'.jpeg'
	# print(face[img])
	p_prob = lg(place[i][l])
	for j in range(6):
		c_prob = lg(max_prob(face[img],j))
		total_prob[0][j]=total_prob[0][j]+c_prob+p_prob

	# print(i,total_prob[0])

for i in xrange(1,z):
	curr_img = add_nulls(i+v_len,6)+'.jpeg'
	prev_img = add_nulls(i,6)+'.jpeg'
	curr_p_prob = lg(place[i+v_len-1][l])
	prev_p_prob = lg(place[i-1][l])
	for j in range(6):
		curr_c_prob = lg(max_prob(face[curr_img],j))
		prev_c_prob = lg(max_prob(face[prev_img],j))
		# if(c_prob ==0 or p_prob ==0):
		# print('hello   = ',c_prob,p_prob)
		total_prob[i][j]=total_prob[i-1][j]+(curr_c_prob-prev_c_prob)+(curr_p_prob-prev_p_prob)


	print(i,total_prob[i])
exit(0)
	
for k in range(num_videos):
	index = find_max(startframe_prob,thres)
	if(index == -1 or no_of_vid == 0):
		break
	# print(index)
	# print('index = ',index,startframe_prob[index])
	# index = 835
	# change prob of neighbouring frames
	flag = 1
	for i in range(6):
		print('hello ',total_prob[index][i],thres)
		if(people[i]==1 and total_prob[index][i]< thres):
			flag = 0
	if(flag == 0):
		for i in range(6):
			total_prob[index][i] = 0
		print('index breaked due to less threshold of one person')
		continue

	total_prob = prob_change(total_prob,index,v_len)
	rand = random.randint(0, 100000)

	copy_frames(index,v_len)
	command = "ffmpeg -framerate 24 -pattern_type glob -i 'tmp3/*.jpg' Videos/"+str(rand)+"_chandler_video.mp4"
	print(command)
	os.system(command)
	no_of_vid = no_of_vid-1
	





