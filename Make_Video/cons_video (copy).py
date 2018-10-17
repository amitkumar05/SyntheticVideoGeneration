# ffmpeg -framerate 24 -pattern_type glob -i '*.jpg' video.mp4

import os,sys
import numpy as np 
import shutil
import random,math



def find_max(startframe_prob,z,result):
	index = result[0]
	m = -1
	# print(m)
	for i in range(len(result)):
		if(startframe_prob[result[i]] > startframe_prob[index]):
			index = result[i]
			m=i
	if(m==-1):
		return -1,result
	result[m]=-1
	return index,result

def copy_frames(index,v_len):
	os.system('rm tmp3/*')
	for i in range(index+1,index+1+v_len):
			source = add_nulls(i,6)+'.jpg'
			dest = 'tmp3/'+source
			source = 's07e03/'+source
			print(source,dest)
			shutil.copy(source,dest)

def prob_change(index,v_len):
	for i in range(index-v_len,index+v_len):
		startframe_prob[i] = 0
	return startframe_prob

def max_prob(faces,c):
	n = faces[0]
	m = 0.000001
	for i in range(n):
		m = max(m,faces[1][i][4+c])
	return m

l = 0  # stands for the place
v_len = 48
z=30000
people = [1,0,0,0,0,0]
thres = math.pow(0.2,v_len)
speak_th = 2.0

face_file = 'results_face_recogn/s07e03.npy'
place_file = 'Results_Episodes/s07e03.npy'
speak_file = 'sp_result/sp_s07e03.npy'

add_nulls = lambda number, zero_count : "{0:0{1}d}".format(number, zero_count)

face = np.load(face_file).item()
place = np.load(place_file).item()['place_prob']
speak = np.load(speak_file).item()



startframe_prob = np.ones(len(place)-v_len)

for i in range(v_len):
	startframe_prob[0]=startframe_prob[0]*place[i][l]
# for i in range(1,len(place)-vlen):
for i in range(1,z):
	startframe_prob[i]=(startframe_prob[i-1]*place[i+v_len-1][l])/place[i-1][l]
	# r = add_nulls(i,6)
	# print(r+'.jpg',startframe_prob[i])

# print(startframe_prob[4564])

# exit(0)


speak_keys = speak.keys()
speak_values = speak.values()
per = 5

print(speak_keys[0])
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
		print('hi',speak_prob,speak_th)
		# print('hk')
		t = speak_keys[i].split('.')[0]
		result.append(int(t))
		cnt = cnt+1

print(cnt)
print(result)

# people starts and place finishes



# person_prob = np.ones((len(place)-v_len,6))
# # print(person_prob[0])
# # exit(0)
# for i in range(v_len):
# 	img = add_nulls(i+1,6)+'.jpeg'
# 	# print(face[img])
# 	for j in range(6):
# 		n_prob = max_prob(face[img],j)
# 		person_prob[0][j]=(person_prob[0][j]*n_prob)

# 	# print(i,person_prob[0])

# # # for i in range(1,len(face)-vlen):
# # for i in range(1,z):
# # 	person_prob[i]=(person_prob[i-1]*face[i+v_len-1][l])/face[i-1][l]


# for i in xrange(1,z):
# 	c_img = add_nulls(i+v_len,6)+'.jpeg'
# 	p_img = add_nulls(i,6)+'.jpeg'
# 	for j in range(6):
# 		c_prob = max_prob(face[c_img],j)
# 		p_prob = max_prob(face[p_img],j)
# 		# if(c_prob ==0 or p_prob ==0):
# 		# print('hello   = ',c_prob,p_prob)
# 		person_prob[i][j]=(person_prob[i-1][j]*c_prob)/p_prob
# 	# if(i%1000 == 0):
# 		# print(i,person_prob[i])


# # Check for character and build episodes

# # num_videos = 1
# # for i in range(6):
# # 	if(people[i] == 1):
# # 		for k in range(num_videos):
# # 			index = find_max(startframe_prob,z)
# # 			# print(index,startframe_prob[index])

# # 			# change prob of neighbouring frames
# # 			startframe_prob = prob_change(index,v_len)
# # 			rand = random.randint(0, 100000)

# # 			copy_frames(index,v_len)
# # 			command = "ffmpeg -framerate 24 -pattern_type glob -i 'tmp3/*.jpg' Videos/"+str(rand)+"_char_video.mp4"
# # 			print(command)
# # 			os.system(command)

# # 			print(index)







# # exit(0)


count = 5
num_videos = 29000
for k in range(cnt):
	index,result = find_max(startframe_prob,z,result)
	
	if(index == -1):
		print('all finished')
		break
	print('place_prob',index,startframe_prob[index])
	copy_frames(index,v_len)
	command = "ffmpeg -framerate 24 -pattern_type glob -i 'tmp3/*.jpg' Videos/"+str(index)+"_video.mp4"
	print(command)
	os.system(command)
	count = count-1
	if(count == 0):
		exit(0)






