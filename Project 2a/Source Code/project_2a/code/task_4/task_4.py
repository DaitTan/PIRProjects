#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:17:33 2021

@author: tanmay
"""

import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import random

def visualiseTriangulation(R_o, T_o,R,T, homoge_points, plotPoints = True):      
    f = 1
    tan_x = 1
    tan_y = 1
     
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    R_prime = R_o
    t_prime = T_o
     
    cam_center_local = np.asarray([
        [0, 0, 0],      [tan_x, tan_y, 1],
        [tan_x, -tan_y, 1],     [0, 0, 0],      [tan_x, -tan_y, 1],
        [-tan_x, -tan_y, 1],    [0, 0, 0],      [-tan_x, -tan_y, 1],
        [-tan_x, tan_y, 1],     [0, 0, 0],      [-tan_x, tan_y, 1],
        [tan_x, tan_y, 1],      [0, 0, 0]
        ]).T
     
    cam_center_local *= f
    cam_center = np.matmul(R_prime, cam_center_local) + t_prime
     
    
    ax.plot(cam_center[0, :], cam_center[1, :], cam_center[2, :],
                    color='k', linewidth=2)
    
    R_prime = R
    t_prime = T
     
    cam_center_local *= f
    cam_center = np.matmul(R_prime, cam_center_local) + t_prime
     
    
    ax.plot(cam_center[0, :], cam_center[1, :], cam_center[2, :],
                    color='k', linewidth=2)
    
    if plotPoints == True:
        x_unhomo = homoge_points[0]/homoge_points[3]
        y_unhomo = homoge_points[1]/homoge_points[3]
        z_unhomo = homoge_points[2]/homoge_points[3]
        ax.scatter(x_unhomo, y_unhomo, z_unhomo)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
def rectifyImage(im, cameraIntrinsic, cameraDist, R, P):
    imSize = im.shape[0:2]
    newcameramtx, roi=cv.getOptimalNewCameraMatrix(cameraIntrinsic,cameraDist,imSize[::-1],0.7,imSize[::-1])
    mapx,mapy = cv.initUndistortRectifyMap(cameraIntrinsic,cameraDist,R,P,imSize[::-1],cv.CV_32FC1)
    undst_img = cv.remap(im,mapx,mapy,cv.INTER_LINEAR, cv.BORDER_CONSTANT)
    x,y,w,h = roi
    # rectifiedImage = undst_img[y:y+h, x:x+w]
    return undst_img

def obtainUndistortedImagePoints(imgPoints, cameraIntrinsic, cameraDist, R = None, P = None):
    img_points_reshaped = imgPoints[0]
    img_points_reshaped = img_points_reshaped[:,0,:].T
    undistortedPoints = cv.undistortPoints(img_points_reshaped, cameraIntrinsic, cameraDist, None, R, P)
    undistortedPoints = undistortedPoints[:,0,:]
    return undistortedPoints.T



def getCorrespondence(path):

    im = cv.imread(path)
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    return im, im_gray
    

def getOptimalMatrix(objpoints, imgpoints, criteria, imagesize_h= 480, imagesize_w = 640):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (imagesize_w,imagesize_h),None,None)
    # newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(imagesize_w,imagesize_w),0,(imagesize_w,imagesize_h))
    return ret, mtx, dist, rvecs, tvecs    

def getDisparityMaps(path_right, path_left, imNo):
    cv_file = cv.FileStorage('parameters/left_camera_intrinsics.xml', cv.FILE_STORAGE_READ)
    left_intrinsic = cv_file.getNode("left_intrinsic").mat()
    left_dist = cv_file.getNode("left_dist").mat()
    cv_file.release()
    
    
    cv_file = cv.FileStorage('parameters/right_camera_intrinsics.xml', cv.FILE_STORAGE_READ)
    right_intrinsic = cv_file.getNode("right_intrinsic").mat()
    right_dist = cv_file.getNode("right_dist").mat()
    cv_file.release
    
    
    cv_file = cv.FileStorage('parameters/stereo_calibration.xml', cv.FILE_STORAGE_READ)
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    cv_file.release()
    
    cv_file = cv.FileStorage('parameters/stereo_rectification.xml', cv.FILE_STORAGE_READ)
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()
    cv_file.release()
    
    
    # Obtain corresponding point on left and right camera
    im_r, im_gray_r = getCorrespondence(path_right)
    im_l, im_gray_l = getCorrespondence(path_left)
    
    # Get Image Size
    imSize = im_gray_r.shape
    imSize_h = imSize[0]
    imSize_w = imSize[1]
    
    # Define criteria for terminating optimization
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-7)
    
    
    dst_right = cv.undistort(im_r, right_intrinsic, right_dist, None)
    dst_left = cv.undistort(im_l, left_intrinsic, left_dist, None)
    
    rect_r_img = rectifyImage(im_gray_r, right_intrinsic, right_dist, R1, P1)
    rect_l_img = rectifyImage(im_gray_l, left_intrinsic, left_dist, R2, P2)
    
    
    
    # x = min(rect_l_img.shape[0],rect_r_img.shape[0])
    # y = min(rect_l_img.shape[1],rect_r_img.shape[1])
    # rect_l_img = rect_l_img[0:x, 0:y,:]
    # rect_r_img = rect_r_img[0:x, 0:y,:]
    
    # for line in range(0, int(rect_l_img.shape[0] / 20)):
    #     rect_l_img[line * 20, :] = (0, 255)
    #     rect_r_img[line * 20, :] = (0, 255)
        
    # cv.imwrite('project_2a/output/task_4/left_right_comparison.png', np.hstack([rect_l_img, rect_r_img]))
    
    # win_size = 5
    # min_disp = -3
    # max_disp = 61 #min_disp * 9
    # num_disp = max_disp - min_disp # Needs to be divisible by 16#Create Block matching object. 
    # stereo = cv.StereoSGBM_create(minDisparity= min_disp,
    #                                 numDisparities = num_disp,
    #                                 blockSize = 5,
    #                                 uniquenessRatio = 10,
    #                                 speckleWindowSize = 150,
    #                                 speckleRange = 2,
    #                                 disp12MaxDiff = 1,
    #                                 P1 = 8*3*win_size**2,#8*3*win_size**2,
    #                                 P2 =32*3*win_size**2,
    #                                 mode = cv.STEREO_SGBM_MODE_SGBM_3WAY) #32*3*win_size**2)#Compute disparity map
    
    
    win_size = 9
    min_disp = 0
    
    max_disp = 64
    num_disp = max_disp - min_disp # Needs to be divisible by 16#Create Block matching object. 
    stereo = cv.StereoSGBM_create(minDisparity= min_disp,
                                    numDisparities = num_disp,
                                    blockSize = win_size,
                                    preFilterCap = 63,
                                    uniquenessRatio = 15,
                                    speckleWindowSize = 10,
                                    speckleRange = 1,
                                    disp12MaxDiff = 20,
                                    P1 = 8*3*win_size**2,#8*3*win_size**2,
                                    P2 =32*3*win_size**2,
                                    mode = cv.STEREO_SGBM_MODE_SGBM_3WAY)
    
    left_matcher = stereo
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)     
    
    l = 70000
    s = 1.2
    
    disparity_filter = cv.ximgproc.createDisparityWLSFilter(left_matcher)
    disparity_filter.setLambda(l)
    disparity_filter.setSigmaColor(s)
    
    d_l = left_matcher.compute(rect_l_img, rect_r_img)
    d_r = right_matcher.compute(rect_r_img, rect_l_img)
    d_l = np.int16(d_l)
    d_r = np.int16(d_r)
    
    d_filter = disparity_filter.filter(d_l, rect_l_img, None, d_r)
    
    print ("\nComputing the disparity  map...")
    
    disparity_map = stereo.compute(rect_l_img, rect_r_img)
    disparity_map = cv.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    
    
    d_filter = cv.normalize(d_filter, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    plt.imshow(d_filter, cmap="gray")
    

    a = np.hstack([im_gray_r, disparity_map, d_filter])
    print(a.shape)
    cv.imwrite('output/task_4/task_4_im_'+imNo+'_disparity.png', a)
            

# input Paths
# os.chdir("/home/tanmay/perInRobo_submission/Project_2/project_2a")

for i in range(5,12):
    i = np.str(i)
    print("Image "+ i)
    path_right = 'images/task_3_and_4/right_'+i+'.png'
    path_left = 'images/task_3_and_4/left_'+i+'.png'
    getDisparityMaps(path_right, path_left, i)