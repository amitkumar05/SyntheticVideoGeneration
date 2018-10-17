# ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' video.mp4

import os,sys
import os.path
import numpy as np 
import shutil
import random,math
from math import log10 as log


files = os.listdir('Results_Episodes')
file_name = 'cafe_nonspeaking.npy'
total_vid = 0
all_sub_video = []

d= {}
if(os.path.exists(file_name) == 1):
	d=np.load(file_name).item()

for ff in range(len(files)):

	face_file = 'results_face_recogn/'+files[ff]
	place_file = 'Results_Episodes/'+files[ff]
	face = np.load(face_file).item()
	place = np.load(place_file).item()['place_prob']
	folder = files[ff].split('.')[0]
	print(files[ff])
	# exit(0)

	l = 0  # stands for the place
	v_len = 48	# length of video sequence
	z=len(place)-v_len-1		# mx numbers of frames

	people = list(sys.argv[1])
	people = [int(x) for x in people]
	print(people)
	# people = [0,0,0,0,0,1]

	# thres = math.pow(0.2,v_len)
	# thres = 200

	thres = 40
	# exit(0)


	no_of_vid = 0
	# num_videos = 29000
	num_videos = z

	# face_file = 'results_face_recogn/s07e03.npy'
	# place_file = 'Results_Episodes/s07e03.npy'



	


	add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

	def find_max(startframe_prob,z,thres):
		index = 0
		for i in range(1,z):
			if(startframe_prob[i][0] > startframe_prob[index][0]):
				index = i
		if(startframe_prob[index][0] < thres):
			return -1
		return index

	def copy_frames(index,v_len):
		os.system('rm tmp3/*')
		for i in range(index+1,index+1+v_len):
				source = add_nulls(i,6)+'.jpg'
				dest = 'tmp3/'+source
				source = 's07e03/'+source
				# print(source,dest)
				shutil.copy(source,dest)

	def prob_change(total_prob,index,v_len):
		for i in range(index-v_len,index+v_len):
			total_prob[i] = 0
		return total_prob


	def max_prob(faces,c):
		n = faces[0]
		m = 0.001 					# changed to 0.001 from 0.01
		# print('n',n)
		for i in range(n):
			m = max(m,faces[1][i][4+c])
		return m


	def lg(x):
		# x = x*1000
		# if(x < 1):
		# 	x = 1
		# return log(x)
		x = x*1.0
		ans = 1+pow(2.73,-10.0*(x-0.5))
		# print('sigmoid',1.0/ans)
		return 1.0/ans


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


		# print(i,total_prob[i])
	print(np.max(total_prob),np.min(total_prob),np.mean(total_prob))
	thres = np.mean(total_prob)
	differ = np.max(total_prob) - thres
	thres = thres + 0.4*differ
	print(thres)

	# exit(0)

	ll = 1	
	k=1
	


	while(k < num_videos):
		# index = find_max(total_prob,z,thres)
		index = k
		k = k + 1
		# if(index == -1 or no_of_vid == 0):
			# break
		# print(k)
		# print(index)
		# index = 835
		# change prob of neighbouring frames
		flag = 1
		# print('hello= ',index,total_prob[index],place[index],thres)
		for i in range(6):
			if(people[i]==1 and total_prob[index][i] < thres):
				flag = 0
		if(flag == 0):
			for i in range(6):
				total_prob[index][i] = 0
			# print('index breaked due to less threshold of one person')
			continue

		# print('index = ',index,total_prob[index])
		# print('total_prob = ',total_prob[index])

		# total_prob = prob_change(total_prob,index,v_len)
		# changing because of taking continuos frames
		# print('hi')
		# print(k)
		k = k+47
		# print(k)
		# rand = random.randint(0, 100000)
		startframe = index
		endframe = index + 48
		curr_sub_video = [folder,startframe,endframe,total_prob[index]]
		all_sub_video.append(curr_sub_video)
		# print(curr_sub_video)
		# copy_frames(index,v_len)
		# command = "ffmpeg -framerate 24 -pattern_type glob -i 'tmp3/*.jpg' Videos/"+'s07_ep3_'+str(index)+"_chandler_video.mp4"
		# print(command)
		# os.system(command)
		
		no_of_vid = no_of_vid+1
	print(no_of_vid)
	total_vid = total_vid+no_of_vid	

d[sys.argv[1]]=all_sub_video
print(total_vid)
np.save(file_name,d)




