# -*- coding: utf-8 -*-
# @Author: amit
# @Date:   2016-11-16 13:16:41
# @Last Modified by:   amit
# @Last Modified time: 2016-11-29 01:36:41

# TRAINING DATA = 	train_data
# VALIDATION DATA =	validation_data
# TESTING DATA =	test_data
# TESTING DATA IMAGE =	test_data_image

import os,sys
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.optimizers import SGD
import shutil

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

top_model_weights_path = 'model.h5'
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
train_labels = np_utils.to_categorical(train_labels, nb_classes)    
validation_labels = np.array([0] * (nb_validation_samples / 3) + [1] * (nb_validation_samples / 3) +   [2] * (nb_validation_samples / 3))
validation_labels = np_utils.to_categorical(validation_labels, nb_classes)    

# fully connected model in keras starts
model = Sequential()
# model.add(Flatten(input_shape=train_data.shape[1:]))
print(train_data.shape[1:])
# exit(0)
model.add(Dense(256,input_shape=(4096,)))
model.add(Dense(256, activation='relu'))

# model.add(Dense(1024,input_shape=(4096,)))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes, activation='sigmoid'))
####

# model.add(Dense(2, activation='softmax'))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# print("done eleven")
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])


# print("10")

model.fit(train_data, train_labels,
          nb_epoch=nb_epoch, batch_size=32,
          validation_data=(validation_data, validation_labels))
print("11")

############################## testing

pdct=model.predict_classes(test_data, batch_size=1, verbose=1)
pdct2=model.predict_proba(test_data, batch_size=1, verbose=1)


# print("hi")

# print(pdct)
# print("hi")
# print(pdct2)
# print(label)
# print("hi")
# print(pdct-test_labels)
# print("hi")
#######################

class_threshold=0.1
other_class_threshold=0.1

# originalpath = "/home/amit/MTPcodes/classify/data/test/"
newpath0 = '../input/aftertest/tested_cafe/'
newpath1 = '../input/aftertest/tested_chandler/'
newpath2 = '../input/aftertest/tested_monica/'
newpath3 = '../input/aftertest/notsure/'
# imgs=os.listdir(path)

# for i in range(0,len(test_data_images)):
#     # print(originalpath+test_images[i])
#     print(str(i)+'.jpg')     
#     print(pdct2[i][0],pdct2[i][1],pdct2[i][2])
#     print(test_data_images[i])     
# exit(0)

for i in range(0,len(test_data_images)):
    # print(originalpath+test_images[i])
    print(str(i)+'.jpg')     
    print(pdct2[i][0],pdct2[i][1],pdct2[i][2])
    print(test_data_images[i])     

    if(pdct2[i][0] >=class_threshold and pdct2[i][1] < pdct2[i][0]/10 and pdct2[i][2] < pdct2[i][0]/10):
        # shutil.copy2(test_data_images[i],newpath0+str(i)+'.jpg')
        shutil.copy2(test_data_images[i],newpath0+test_data_images[i])

    elif(pdct2[i][1] >= class_threshold  and pdct2[i][0] < pdct2[i][1]/10 and pdct2[i][2] < pdct2[i][1]/10):
        shutil.copy2(test_data_images[i],newpath1+str(i)+'.jpg')
    elif(pdct2[i][2] >= class_threshold  and pdct2[i][1] < pdct2[i][2]/10 and pdct2[i][0] < pdct2[i][2]/10):
        shutil.copy2(test_data_images[i],newpath2+str(i)+'.jpg')
    else:
        shutil.copy2(test_data_images[i],newpath3+str(i)+'.jpg')


class_threshold=0.5
other_class_threshold=0.5
newpath0 = '../input/aftertest1/tested_cafe/'
newpath1 = '../input/aftertest1/tested_chandler/'
newpath2 = '../input/aftertest1/tested_monica/'
newpath3 = '../input/aftertest1/notsure/'
# imgs=os.listdir(path)

# for i in range(0,len(test_data_images)):
#     # print(originalpath+test_images[i])
#     print(str(i)+'.jpg')     
#     print(pdct2[i][0],pdct2[i][1],pdct2[i][2])
#     print(test_data_images[i])     
# exit(0)

for i in range(0,len(test_data_images)):
    # print(originalpath+test_images[i])
    print(str(i)+'.jpg')     
    print(pdct2[i][0],pdct2[i][1],pdct2[i][2])
    print(test_data_images[i])     

    if(pdct2[i][0] >=class_threshold and pdct2[i][1] < pdct2[i][0]/10 and pdct2[i][2] < pdct2[i][0]/10):
        # shutil.copy2(test_data_images[i],newpath0+str(i)+'.jpg')
        shutil.copy2(test_data_images[i],newpath0+test_data_images[i])

    elif(pdct2[i][1] >= class_threshold  and pdct2[i][0] < pdct2[i][1]/10 and pdct2[i][2] < pdct2[i][1]/10):
        shutil.copy2(test_data_images[i],newpath1+str(i)+'.jpg')
    elif(pdct2[i][2] >= class_threshold  and pdct2[i][1] < pdct2[i][2]/10 and pdct2[i][0] < pdct2[i][2]/10):
        shutil.copy2(test_data_images[i],newpath2+str(i)+'.jpg')
    else:
        shutil.copy2(test_data_images[i],newpath3+str(i)+'.jpg')

model.save_weights(top_model_weights_path)