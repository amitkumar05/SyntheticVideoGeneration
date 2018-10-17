import os,sys
import numpy as np 
import shutil
import random,math
from math import log10 as log


for i in range(1,64):
	people = "{0:{fill}6b}".format(i,fill='0')
	# print(list(people))
	count = 0
	for x in people:
		if x == '1':
			count=count+1
	if(count == 6):
		# print(people)
		command = "python cons_video1.py "+people+' > Log/'+people+'.txt'
		print(command)
		os.system(command)
		# exit(0)