# -*- coding: utf-8 -*-
# @Author: amit
# @Date:   2017-02-25 10:18:35
# @Last Modified by:   amit
# @Last Modified time: 2017-03-25 13:37:59

from sklearn import svm
import sys,os
import glob
import numpy as np
import h5py

from sklearn.svm import NuSVC
from sklearn.externals import joblib

print('full fc7 SVM')
print('type = ',sys.argv[1])

# new added

nb_epoch = 500
nb_classes = 3

train_cafe = 'feature/train_cafe.npy'
train_chandler = 'feature/train_chandler.npy'
train_monica = 'feature/train_monica.npy'

validation_cafe = 'feature/validation_cafe.npy'
validation_chandler = 'feature/validation_chandler.npy'
validation_monica = 'feature/validation_monica.npy'

test='feature/test.npy'
test_images='feature/test_images.npy'

top_model_weights_path = 'svm_model.h5'
train_data1 = np.load(open(train_cafe))
train_data2 = np.load(open(train_chandler))
train_data3 = np.load(open(train_monica))

validation_data1 = np.load(open(validation_cafe))
validation_data2 = np.load(open(validation_chandler))
validation_data3 = np.load(open(validation_monica))

test_data = np.load(open(test))
test_data_images = np.load(open(test_images))


nb_train_samples = len(train_data1)+len(train_data2)+len(train_data3)
nb_validation_samples = len(validation_data1)+len(validation_data2)+len(validation_data3)
nb_test_samples=len(test_data)


#concatenating train data
train_data=train_data1
train_data=np.concatenate((train_data,train_data2),axis=0)
train_data=np.concatenate((train_data,train_data3),axis=0)


validation_data=validation_data1
validation_data=np.concatenate((validation_data,validation_data2),axis=0)
validation_data=np.concatenate((validation_data,validation_data3),axis=0)

# to be deleted
# test_data=train_data

# compution labels for training and validation data
train_labels = np.array([0] * (nb_train_samples / 3) + [1] * (nb_train_samples / 3) +  [2] * (nb_train_samples / 3))
# train_labels = np_utils.to_categorical(train_labels, nb_classes)    
print(train_labels)

validation_labels = np.array([0] * (nb_validation_samples / 3) + [1] * (nb_validation_samples / 3) +   [2] * (nb_validation_samples / 3))
# validation_labels = np_utils.to_categorical(validation_labels, nb_classes)    
print(validation_labels)



# new finished


# TRAINING LABELS

# text_file = open(training_labels_file, "r")
# lines = text_file.read().splitlines()
# train_file_labels = [n.split('\t') for n in lines]
# train_labels = [n[1] for n in train_file_labels]
# # print((train_labels))
# print(len(train_labels))
# print('Loaded Training labels')



# svm TRAINING
if(sys.argv[1] == '0'):
	clf = svm.LinearSVC()
elif(sys.argv[1] == '1'):
	# clf = svm.SVC(decision_function_shape='ovo')
	clf = svm.SVC(probability=True)
else:
	clf = NuSVC()


clf.fit(train_data,train_labels)
print('Training done')


print('PREDICTION FOR VALIDATION DATA')
Y = clf.predict(validation_data)



correct = 0  
for i in range(len(Y)):
	# print(i,Y[i])
	if(Y[i] == validation_labels[i] ):
		correct = correct + 1
print("TRAINING SCORE BY SVM = ",100*clf.score(train_data,train_labels))
print('validation_correctly_predicted = ',correct,' OUTOFF = ',len(Y),' ACCURACY = ',(float)(correct*100)/len(Y)) 
print('TRAINING DATA Prediction done')

print('saving model')

if(sys.argv[1] == '0'):
	s = joblib.dump(clf,'model_svm_linear.pkl')
	# s = joblib.dump(clf,'model.pkl')

elif(sys.argv[1] == '1'):
	s = joblib.dump(clf,'model_svm_svc.pkl')
else:
	s = joblib.dump(clf,'model_svm_nu.pkl')


exit(0)
clf2 = joblib.load('model.pkl')
acc = clf2.score(X,train_labels)
print(acc)