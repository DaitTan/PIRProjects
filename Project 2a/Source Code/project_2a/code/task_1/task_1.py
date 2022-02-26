#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 18:58:37 2021

@author: tanmay
"""

import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import random


def obtainCorners(im, chessboardSize):
    chessboardSize_h = chessboardSize[0]
    chessboardSize_w = chessboardSize[1]
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
    ret, corners  = cv.findChessboardCorners(im_gray, (chessboardSize_h,chessboardSize_w), None)
    if ret == True:
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv.cornerSubPix(im_gray,corners, (3,3), (-1,-1), criteria)
    else:
        corners2 = []
    return corners, corners2, im_gray, ret

def getCorrespondence(path, chessboardSize):
    
    imgpoints = [] # 2d points in image plane.
    
    images_ = glob.glob(path)
    for image in images_:
        im = cv.imread(image)
        _, corners2, _,_ = obtainCorners(im, chessboardSize)
        # im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
        # ret, corners  = cv.findChessboardCorners(im_gray, (chessboardSize_h,chessboardSize_w), None)
        imgpoints.append(corners2)
        
    return imgpoints

def readImagesAndGetPoints(path, chessboardSize):
    
    chessboardSize_h = chessboardSize[0]
    chessboardSize_w = chessboardSize[1]
    
    objp = np.zeros((chessboardSize_w* chessboardSize_h,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize_h,0:chessboardSize_w].T.reshape(-1,2)
    
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images_ = glob.glob(path)
    
    
    
    
    for image in images_:   
        im = cv.imread(image)
        im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        
        ret, corners  = cv.findChessboardCorners(im_gray, (chessboardSize_h,chessboardSize_w), None)
        
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
    return objpoints, imgpoints
  
 
def getOptimalMatrix(objpoints, imgpoints, criteria, imagesize_h= 480, imagesize_w = 640):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (imagesize_w,imagesize_h),None,None)
    # newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(imagesize_w,imagesize_w),0,(imagesize_w,imagesize_h))
    return ret, mtx, dist, rvecs, tvecs    
    
    
def seeOptimalNumeberOfImages(objpoints, imgpoints, criteria):  
    globalErr = []
    
    for sampleCount in range(1,12):
        err = []
        print(sampleCount)
        for count_2 in range(1,50):
        
            indices = [i for i in range(len(objpoints))]
            
            randomSampleIndices = random.sample(indices, sampleCount)
                
            objpoint_sample = [objpoints[i] for i in randomSampleIndices]
            imgpoint_sample = [imgpoints[i] for i in randomSampleIndices]
            ret, mtx, dist, rvecs, tvecs, = getOptimalMatrix(objpoint_sample, imgpoint_sample, criteria)
            
            # h = 480
            # w = 640
            
            
            # ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoint_sample, imgpoint_sample, (w,h),None,None)
            
            # newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
            
            tot_error = 0
            for i in range(len(objpoints)):
                ret_,rvecs_, tvecs_ = cv.solvePnP(objpoints[i], imgpoints[i], mtx, dist)
                imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs_, tvecs_, mtx, dist)
                error = cv.norm(imgpoints[i],imgpoints2, cv.NORM_L2)/len(imgpoints2)
                tot_error += error
                
            err.append(tot_error/len(objpoints))
            
        globalErr.append(np.mean(err))
    return globalErr, np.argmin(globalErr)

def runForXCamera(objPoints, imgPoints, sampleCount = 11):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 1e-5)
    
    # globalErr, minError = seeOptimalNumeberOfImages(objpoints, imgpoints, criteria)
    
    randomSampleIndices = random.sample([0,1,2,3,4,5,6,7,8,9,10],sampleCount)
    
    objpoint_sample = [objPoints[i] for i in randomSampleIndices]
    imgpoint_sample = [imgPoints[i] for i in randomSampleIndices]
    
    w = 640
    h = 480
    
    ret, mtx, dist, rvecs, tvecs = getOptimalMatrix(objpoint_sample, imgpoint_sample, criteria)
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))  
    
    return ret, mtx, dist, rvecs, tvecs, newcameramtx, roi

# os.chdir("/home/tanmay/perInRobo_submission/Project_2")
chessboardSize = [6,9]
path_right = 'images/task_1/right_*.png'
path_left = 'images/task_1/left_*.png'
     


# Define Chessboard Sizes
chessboardSize = [6,9]
chessboardSize_h = chessboardSize[0]
chessboardSize_w = chessboardSize[1]


# Obtain corresponding point on left and right camera
imgPoints_r = getCorrespondence(path_right, chessboardSize)
imgPoints_l = getCorrespondence(path_left, chessboardSize)

# Obtain Object map coordinates
objp = np.zeros((chessboardSize_w* chessboardSize_h,3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize_h,0:chessboardSize_w].T.reshape(-1,2)
objPoints = []
for i in range(len(imgPoints_l)):
    objPoints.append(objp)
    


ret_right, right_intrinsic, right_dist, rvecs_right, tvecs_right, _, _  = runForXCamera(objPoints, imgPoints_r, sampleCount = 11)
ret_left, left_intrinsic, left_dist, rvecs_left, tvecs_left, _, _ = runForXCamera(objPoints, imgPoints_l, sampleCount = 10)

# right_intrinsic = np.round(right_intrinsic)
# # right_dist = np.round(right_dist)
# left_intrinsic = np.round(left_intrinsic)
# # left_dist = np.round(left_dist)

img_right = cv.imread('images/task_1/right_2.png') 
img_left = cv.imread('images/task_1/left_2.png') 


# undistort and plot right images
dst_right = cv.undistort(img_right, right_intrinsic, right_dist, None)

fig, ax = plt.subplots(1,3)

ax[0].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
ax[0].axis('off')

_, corners,_,ret, = obtainCorners(img_right, chessboardSize[::-1])
cv.drawChessboardCorners(img_right, (9,6), corners,ret)
ax[1].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
ax[1].axis('off')

ax[2].imshow(cv.cvtColor(dst_right, cv.COLOR_BGR2RGB))
ax[2].axis('off')
plt.savefig('output/task_1/task_1_right.png', dpi = 700)
plt.close()



# undistort and plot left images
dst_left = cv.undistort(img_left, left_intrinsic, left_dist, None)

fig, ax = plt.subplots(1,3)

ax[0].imshow(cv.cvtColor(img_left, cv.COLOR_BGR2RGB))
ax[0].axis('off')

_, corners,_,ret, = obtainCorners(img_left, chessboardSize[::-1])
cv.drawChessboardCorners(img_left, (9,6), corners,ret)
ax[1].imshow(cv.cvtColor(img_left, cv.COLOR_BGR2RGB))
ax[1].axis('off')

ax[2].imshow(cv.cvtColor(dst_left, cv.COLOR_BGR2RGB))
ax[2].axis('off')
plt.savefig('output/task_1/task_1_left.png', dpi = 700)
plt.close()


img_right = cv.imread('images/task_1/right_2.png') 
img_left = cv.imread('images/task_1/left_2.png') 


fig, ax = plt.subplots(2,3)

ax[0,0].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
ax[0,0].axis('off')

_, corners,_,ret, = obtainCorners(img_right, chessboardSize[::-1])
cv.drawChessboardCorners(img_right, (9,6), corners,ret)
ax[0,1].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
ax[0,1].axis('off')

ax[0,2].imshow(cv.cvtColor(dst_right, cv.COLOR_BGR2RGB))
ax[0,2].axis('off')


# undistort and plot left images
dst_left = cv.undistort(img_left, left_intrinsic, left_dist, None)


ax[1,0].imshow(cv.cvtColor(img_left, cv.COLOR_BGR2RGB))
ax[1,0].axis('off')

_, corners,_,ret, = obtainCorners(img_left, chessboardSize[::-1])
cv.drawChessboardCorners(img_left, (9,6), corners,ret)
ax[1,1].imshow(cv.cvtColor(img_left, cv.COLOR_BGR2RGB))
ax[1,1].axis('off')

ax[1,2].imshow(cv.cvtColor(dst_left, cv.COLOR_BGR2RGB))
ax[1,2].axis('off')
plt.savefig('output/task_1/task_1_lefttop_rightdown.png', dpi = 700)
plt.close()
    


cv_file = cv.FileStorage('parameters/left_camera_intrinsics.xml', cv.FILE_STORAGE_WRITE)
cv_file.write("left_intrinsic", left_intrinsic)
cv_file.write("left_dist", left_dist)
cv_file.release()

cv_file = cv.FileStorage('parameters/right_camera_intrinsics.xml', cv.FILE_STORAGE_WRITE)
cv_file.write("right_intrinsic", right_intrinsic)
cv_file.write("right_dist", right_dist)
cv_file.release()
# np.savetxt('project_2a/parameters/right_camera_intrinsics.txt', right_intrinsic)
# np.savetxt('project_2a/parameters/left_camera_intrinsics.txt', left_intrinsic)
# np.savetxt('project_2a/parameters/right_camera_dist.txt', right_dist)
# np.savetxt('project_2a/parameters/left_camera_dist.txt', left_dist)

