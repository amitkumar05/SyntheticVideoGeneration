# ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' video.mp4

import os,sys
import numpy as np 
import shutil
import random,math

files = os.listdir('Results_Episodes')
file_name = 'cafe_speaking.npy'
total_vid = 0
all_sub_video = []

d= {}
if(os.path.exists(file_name) == 1):
	d=np.load(file_name).item()

for ff in range(len(files)):

	# face_file = 'results_face_recogn/s07e03.npy'
	# place_file = 'Results_Episodes/s07e03.npy'
	# speak_file = 'sp_result/sp_s07e03.npy'

	face_file = 'results_face_recogn/'+files[ff]
	place_file = 'Results_Episodes/'+files[ff]
	speak_file = 'sp_result/sp_'+files[ff]


	folder = files[ff].split('.')[0]
	def find_max(startframe_prob,z,result):
		index = result[0]
		# m = -1
		# print(m)
		for i in range(len(result)):
			if(startframe_prob[result[i]] > startframe_prob[index]):
				index = result[i]
				# m=i
		# print('hello',m)
		# if(m==-1):
			# return -1,result
		# result[m]=-1
		return index

	def copy_frames(index,v_len):
		os.system('rm tmp3/*')
		for i in range(index+1,index+1+v_len):
				source = add_nulls(i,6)+'.jpg'
				dest = 'tmp3/'+source
				source = 's07e03/'+source
				# print(source,dest)
				shutil.copy(source,dest)

	def prob_change(index,v_len):
		for i in range(index-v_len,index+v_len):
			startframe_prob[i] = 0
		return startframe_prob

	def lg(x):
			# x = x*1000
			# if(x < 1):
			# 	x = 1
			# return log(x)
			x = x*1.0

			ans = 1+pow(2.73,-10.0*(x-0.5))
			# print('sigmoid',1.0/ans)
			# print('sigmoid = ',x,1.0/ans)
			return 1.0/ans

	def max_prob(faces,c):
		n = faces[0]
		m = 0.000001
		for i in range(n):
			m = max(m,faces[1][i][4+c])
		return m

	face = np.load(face_file).item()
	place = np.load(place_file).item()['place_prob']
	speak = np.load(speak_file).item()

	v_len = 35
	z=len(place)-v_len

	l = 0  # stands for the place
	# people = [1,0,0,0,0,0]
	threshold = 50.0	
	# speak_th = 2.0
	speak_th = np.mean(speak.values())
	# exit(0)
	add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

	startframe_prob = np.zeros(len(place)-v_len)

	for i in range(v_len):
		startframe_prob[0]=startframe_prob[0]+lg(place[i][l])

	for i in range(1,z):
		startframe_prob[i]=startframe_prob[i-1]+lg(place[i+v_len-1][l])-lg(place[i-1][l])
	startframe_prob = startframe_prob*(100.0/35.0)
	print('avg_prob= ',np.max(startframe_prob),np.min(startframe_prob),np.mean(startframe_prob))

	speak_keys = speak.keys()
	speak_values = speak.values()
	per = int(sys.argv[1])
	# print(per)
	# exit(0)

	cnt = 0
	for i in range(len(speak_keys)):
		speak_prob = speak[speak_keys[i]]
		per_prob = np.zeros(6)
		for j in range(6):
			per_prob[j]=max_prob(face[speak_keys[i]],j)
		if(speak_prob > speak_th and np.argmax(per_prob) == per ):
			t = speak_keys[i].split('.')[0]
			index = int(t)
			# print('place_prob= ',index,startframe_prob[index])
			if(startframe_prob[index] > threshold):
				print(t)
				startframe = index
				endframe = index + 35
				curr_sub_video = [folder,startframe,endframe,startframe_prob[index]]
				print(curr_sub_video)
				all_sub_video.append(curr_sub_video)
				total_vid = total_vid+1				
	print(total_vid)
print(total_vid)
d[sys.argv[2]]=all_sub_video
np.save(file_name,d)
# print(d)
print(d.keys())

# exit(0)

