# ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' video.mp4

import os,sys
import numpy as np 
import shutil
import random,math



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
		return 1.0/ans

def max_prob(faces,c):
	n = faces[0]
	m = 0.000001
	for i in range(n):
		m = max(m,faces[1][i][4+c])
	return m

l = 0  # stands for the place
v_len = 35
# z=30000
# z=len(place)-v_len
people = [1,0,0,0,0,0]
# thres = math.pow(0.2,v_len)
threshold = math.pow(0.3,v_len)
speak_th = 2.0

face_file = 'results_face_recogn/s07e03.npy'
place_file = 'Results_Episodes/s07e03.npy'
speak_file = 'sp_result/sp_s07e03.npy'

add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

face = np.load(face_file).item()
place = np.load(place_file).item()['place_prob']
speak = np.load(speak_file).item()

z=len(place)-v_len

startframe_prob = np.ones(len(place)-v_len)

for i in range(v_len):
	startframe_prob[0]=startframe_prob[0]*place[i][l]
# for i in range(1,len(place)-vlen):
for i in range(1,z):
	startframe_prob[i]=(startframe_prob[i-1]*place[i+v_len-1][l])/place[i-1][l]
	# r = add_nulls(i,6)
	# print(r+'.jpg',startframe_prob[i])

# print(startframe_prob[30101:30140])
# print('hi')
# print(place[30101:30140])

# exit(0)


speak_keys = speak.keys()
speak_values = speak.values()
per = 5

print('dsadsa = ',len(speak_keys))
# print(face.keys())
# print(face[])


cnt = 0
result = []
for i in range(len(speak_keys)):
	speak_prob = speak[speak_keys[i]]
	# speak_person_prob = max()
	per_prob = np.zeros(6)
	for j in range(6):
		per_prob[j]=max_prob(face[speak_keys[i]],j)
	# print(per_prob,np.argmax(per_prob))
	if(speak_prob > speak_th and np.argmax(per_prob) == per ):
		# print('hi',speak_prob,speak_th)
		# print('hk')
		t = speak_keys[i].split('.')[0]
		result.append(int(t))
		index = int(t)
		if(startframe_prob[index] > threshold):
			print(t)
		cnt = cnt+1

print(cnt)
print(result)

# exit(0)
# people starts and place finishes



count = 5
num_videos = 29000
for k in range(cnt):
	index = find_max(startframe_prob,z,result)
	# print('iside',index,result)
	# if(index == -1):
		# print('all finished')
		# break
	print('place_prob= ',k,index,startframe_prob[index])
	if(startframe_prob[index] < threshold):
		break
	startframe_prob[index] = -1
	copy_frames(index,v_len)
	number = add_nulls(k,3)
	command = "ffmpeg -framerate 24 -pattern_type glob -i 'tmp3/*.jpg' 1Videos/"+str(number)+'_'+str(index)+"_video.mp4"
	print(command)
	os.system(command)
	# count = count-1
	# if(count == 0):
		# exit(0)






