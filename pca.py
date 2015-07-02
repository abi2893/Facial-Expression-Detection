import cv2
import numpy as np
import sys
import getopt
import os

img_names = []
rootdir = '/Users/abhishekghosh/Documents/Python/Facial_Recognition/att_faces'

for subdir, dirs, files in os.walk(rootdir):
	img_names.append(subdir+"/1.pgm")
img_names.pop(0)


faceList = []
for fileName in img_names:
	faceList.append(cv2.imread(fileName))

covMatrix = None
for im in faceList:
	try:
		covMatrix = np.vstack((covMatrix,np.reshape(im,np.product(im.shape))))
	except:
		covMatrix = np.reshape(im,np.product(im.shape))

mean, eigenvectors = cv2.PCACompute(covMatrix,np.mean(covMatrix,axis=0).reshape(1,-1))

print(mean)
print(np.mean(covMatrix,axis=0))
print(np.mean(covMatrix,axis=0).reshape(1,-1))
#print(eigenvectors)


        