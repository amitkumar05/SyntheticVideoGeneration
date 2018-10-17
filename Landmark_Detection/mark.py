#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING THE DLIB PYTHON INTERFACE
#   Dlib comes with a compiled python interface for python 2.7 on MS Windows. If
#   you are using another python version or operating system then you need to
#   compile the dlib python interface before you can use this file.  To do this,
#   run compile_dlib_python_module.bat.  This should work on any operating
#   system so long as you have CMake and boost-python installed.
#   On Ubuntu, this can be done easily by running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install -U scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io
import scipy.misc
from PIL import Image

if len(sys.argv) != 3:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
m = 0
for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
    print("Processing file: {}".format(f))
    img = io.imread(f)
    name=f.split("/")
    filename="Landmarks_"+name[0]+"/"+name[1]+".txt"
    print(filename)
    # exit(0)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(img, 1)
   # print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
           k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        
        shape = predictor(img, d)
        # print(img[0][0])
        le_x=0
        le_y=0
        for i in xrange(36,42):
            le_x=le_x+shape.part(i).x
            le_y=le_y+shape.part(i).y

        re_x=0
        re_y=0
        for i in xrange(42,48):
            re_x=re_x+shape.part(i).x
            re_y=re_y+shape.part(i).y

            # for i in range(m+1):
            #     # x = shape.part(i).x
            #     for k in xrange(-10,10):
            #         for l in xrange(-10,10):    
            #             img[shape.part(i).y+k][shape.part(i).x+l][0]=0
            #             img[shape.part(i).y+k][shape.part(i).x+l][1]=0
            #             img[shape.part(i).y+k][shape.part(i).x+l][2]=0
        le_x=int(le_x/6)
        le_y=int(le_y/6)
        re_x=int(re_x/6)
        re_y=int(re_y/6)

        for k in xrange(-5,5):
            for l in xrange(-5,5):    
                # if le_y+k >=0  and le_x+l>=0 and le_x+l<224 and le_y+k<224:
                    img[le_y+k][le_x+l][0]=255
                    img[le_y+k][le_x+l][1]=0
                    img[le_y+k][le_x+l][2]=0
        for i in range(-1):
            print(i)
            le_y = shape.part(i).y
            le_x = shape.part(i).x
            for k in xrange(-3,3):
                for l in xrange(-3,3):
                    # if le_y+k >=0  and le_x+l >= 0 and le_x+l < 224 and le_y+k < 224:
                        img[le_y+k][le_x+l][0]=255
                        img[le_y+k][le_x+l][1]=0
                        img[le_y+k][le_x+l][2]=0



        for k in xrange(-5,5):
            for r in xrange(-5,5):
                # if le_y+k >=0  and le_x+l>=0 and le_x+l<224 and le_y+k<224:    
                    img[re_y+k][re_x+r][0]=255
                    img[re_y+k][re_x+r][1]=0
                    img[re_y+k][re_x+r][2]=0
            # print(x)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                              shape.part(1)))
            # Draw the face landmarks on the screen.
        # imshow(img)
        # file = open(filename, 'w')
    
        # for i in range(64):
        #     file.write(str(shape.part(i).x)+" "+str(shape.part(i).y)+"\n")
        #     print(shape.part(i).x,shape.part(i).y)
        # # file.write(str(le_x)+"\n")
        # # file.write(str(le_y)+"\n")
        # # file.write(str(re_x)+"\n")
        # # file.write(str(re_y))
        # file.close()  # you can omit in most cases as the destructor will call it        
        im = Image.fromarray(img)
        im.save("output/"+str(m)+".png")
        m=m+1
        # io.imwrite(img,'land.jpg')
        #print(shape)