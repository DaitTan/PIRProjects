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
    return ax
    
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
    im = cv.imread(path)
    _, corners2,im_gray,_ = obtainCorners(im, chessboardSize)
    imgpoints.append(corners2)
    return im, im_gray, imgpoints

def runExperiment(path_right, path_left, imageView_1, imageView_2, saveParam = False):
    imagesConfig = imageView_1 + '_' + imageView_2
    
    print("Reading " + imageView_1+" parameters")
    cv_file = cv.FileStorage('parameters/' + imageView_1 + '_camera_intrinsics.xml', cv.FILE_STORAGE_READ)
    right_intrinsic  = cv_file.getNode(imageView_1 + "_intrinsic").mat()
    right_dist  = cv_file.getNode(imageView_1 + "_dist").mat()
    cv_file.release
    
    print("Reading " + imageView_2+" parameters")
    cv_file = cv.FileStorage('parameters/' + imageView_2 + '_camera_intrinsics.xml', cv.FILE_STORAGE_READ)
    left_intrinsic  = cv_file.getNode(imageView_2 + "_intrinsic").mat()
    left_dist   = cv_file.getNode(imageView_2 + "_dist").mat()
    cv_file.release()
    

    
    # Define Chessboard Sizes
    chessboardSize = [6,9]
    chessboardSize_h = chessboardSize[0]
    chessboardSize_w = chessboardSize[1]
    
    
    # Obtain Object map coordinates
    objp = np.zeros((chessboardSize_w* chessboardSize_h,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize_h,0:chessboardSize_w].T.reshape(-1,2)
    objpoints = []
    objpoints.append(objp)
    
    # Obtain corresponding point on left and right camera
    im_r, im_gray_r, imgPoints_r = getCorrespondence(path_right, chessboardSize)
    im_l, im_gray_l, imgPoints_l = getCorrespondence(path_left, chessboardSize)
    
    # Get Image Size
    imSize = im_gray_r.shape
    imSize_h = imSize[0]
    imSize_w = imSize[1]
    
    # Define criteria for terminating optimization
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10000, 1e-7)
    
    # Stereo Calibrate
    # R = Rotation Matrix
    # T = Translation Matrix
    # E = Essential Matrix
    # F = Fundamental Matrix
    ret, _,_,_,_, R, T, E, F = cv.stereoCalibrate(objpoints,imgPoints_r, imgPoints_l, right_intrinsic, right_dist,left_intrinsic, left_dist,  imSize[::-1], criteria, flags = cv.CALIB_FIX_INTRINSIC)
    
    
    if saveParam == True:
        cv_file = cv.FileStorage('parameters/stereo_calibration.xml', cv.FILE_STORAGE_WRITE)
        print("Saving Params")
        cv_file.write("R", R)
        cv_file.write("T", T)
        cv_file.write("E", E)
        cv_file.write("F", F)
        cv_file.release()
    
    R1, R2, P1, P2, Q, roi_r, roi_l = cv.stereoRectify(right_intrinsic, right_dist, left_intrinsic, left_dist, imSize[::-1], R, T)
    
    
    dst_right = cv.undistort(im_r, right_intrinsic, right_dist, None)
    dst_left = cv.undistort(im_l, left_intrinsic, left_dist, None)
    
    rect_r_img = rectifyImage(im_r, right_intrinsic, right_dist, R1, P1)
    rect_l_img = rectifyImage(im_l, left_intrinsic, left_dist, R2, P2)
    
    
    
    
    # Plot images
    fig, ax = plt.subplots(3,2)
    
    im_gray = cv.cvtColor(im_l, cv.COLOR_BGR2GRAY)
    ret_1, corners_1  = cv.findChessboardCorners(im_gray, (chessboardSize_w, chessboardSize_h), None)
    cv.drawChessboardCorners(im_l, (9,6), corners_1,ret_1)
    ax[0,0].imshow(cv.cvtColor(im_l, cv.COLOR_BGR2RGB))
    ax[0,0].axis('off')
    
    
    im_gray = cv.cvtColor(dst_left, cv.COLOR_BGR2GRAY)
    ret_2, corners_2  = cv.findChessboardCorners(im_gray, (chessboardSize_w,chessboardSize_h), None)
    cv.drawChessboardCorners(dst_left, (9,6), corners_2,ret_2)
    ax[1,0].imshow(cv.cvtColor(dst_left, cv.COLOR_BGR2RGB))
    ax[1,0].axis('off')
    
    # im_gray = cv.cvtColor(rect_l_img, cv.COLOR_BGR2GRAY)
    # ret_3, corners_3  = cv.findChessboardCorners(rect_l_img, (chessboardSize_w,chessboardSize_h), None)
    # cv.drawChessboardCorners(rect_l_img, (9,6), corners_3,ret_3)
    ax[2,0].imshow(cv.cvtColor(rect_l_img, cv.COLOR_BGR2RGB))
    ax[2,0].axis('off')
    
    im_gray = cv.cvtColor(im_r, cv.COLOR_BGR2GRAY)
    ret_4, corners_4  = cv.findChessboardCorners(im_gray, (chessboardSize_w, chessboardSize_h), None)
    img_right = cv.drawChessboardCorners(im_r, (9,6), corners_4,ret_4)
    ax[0,1].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
    ax[0,1].axis('off')
    
    im_gray = cv.cvtColor(dst_right, cv.COLOR_BGR2GRAY)
    ret_5, corners_5  = cv.findChessboardCorners(im_gray, (chessboardSize_w, chessboardSize_h), None)
    dst_right = cv.drawChessboardCorners(dst_right, (9,6), corners_5,ret_5)
    ax[1,1].imshow(cv.cvtColor(dst_right, cv.COLOR_BGR2RGB))
    ax[1,1].axis('off')
    
    # im_gray = cv.cvtColor(rect_r_img, cv.COLOR_BGR2GRAY)
    # ret_6, corners_6  = cv.findChessboardCorners(im_gray, (chessboardSize_w, chessboardSize_h), None)
    # rect_r_img = cv.drawChessboardCorners(rect_r_img, (9,6), corners_6,ret_6)
    ax[2,1].imshow(cv.cvtColor(rect_r_img, cv.COLOR_BGR2RGB))
    ax[2,1].axis('off')
    plt.savefig('output/task_2/' + imagesConfig + '_comparison.png', dpi = 700, bbox_inches='tight')
    
    
    if saveParam == True:
        print("Saving Params")
        cv_file = cv.FileStorage('parameters/stereo_rectification.xml', cv.FILE_STORAGE_WRITE)
        cv_file.write("R1", R1)
        cv_file.write("R2", R2)
        cv_file.write("P1", P1)
        cv_file.write("P2", P2)
        cv_file.write("Q", Q)
        cv_file.release()
    
    
    x = min(rect_l_img.shape[0],rect_r_img.shape[0])
    y = min(rect_l_img.shape[1],rect_r_img.shape[1])
    rect_l_img = rect_l_img[0:x, 0:y,:]
    rect_r_img = rect_r_img[0:x, 0:y,:]
    
    for line in range(0, int(rect_l_img.shape[0] / 20)):
        rect_l_img[line * 20, :] = (0, 0, 255)
        rect_r_img[line * 20, :] = (0, 0, 255)
    cv.imwrite('output/task_2/task_2_' + imagesConfig + '_rectification_comparison.png', np.hstack([rect_l_img, rect_r_img]))
    
    
    undst_right = obtainUndistortedImagePoints(imgPoints_r, right_intrinsic, right_dist)
    undst_left = obtainUndistortedImagePoints(imgPoints_l, left_intrinsic, left_dist)
    
    
    # Assuming origin of world frame for camera 
    projection_visual_1 = np.append(np.eye(3), np.array([[0,0,0]]).T,1)
    # Roatation and Translation with of camera 2 with respect to camera2
    projection_visual_2 = np.append(R, T,1)
    # Triangulate Points
    homoge_points = cv.triangulatePoints(projection_visual_1, projection_visual_2, undst_right, undst_left)
    
    R_o = np.eye(3)
    T_o = np.array([[0,0,0]]).T
    visualiseTriangulation(R_o, T_o, R, T, homoge_points, plotPoints = True)
    
    plt.savefig('output/task_2/task_2_'+imagesConfig+'_pre_rect_withPoints.png', dpi = 700,bbox_inches='tight')
    
    visualiseTriangulation(R_o, T_o, R, T, homoge_points, plotPoints = False)
    plt.savefig('output/task_2/task_2_'+imagesConfig+'_pre_rect_withoutPoints.png', dpi = 700,bbox_inches='tight')
    
    # Assuming origin of world frame for camera 
    
    rect_r_img = obtainUndistortedImagePoints(imgPoints_r, right_intrinsic, right_dist)
    rect_l_img = obtainUndistortedImagePoints(imgPoints_l, left_intrinsic, left_dist)
    
    
    homoge_points_r = cv.triangulatePoints(projection_visual_1, projection_visual_2, rect_r_img, rect_l_img)
    R_o = np.eye(3)
    visualiseTriangulation(R1, T_o, R2, T, homoge_points_r, plotPoints = True)
    plt.savefig('output/task_2/task_2_'+imagesConfig+'_post_rect_withPoints.png', dpi = 700, bbox_inches='tight')
    visualiseTriangulation(R1, T_o, R2, T, homoge_points_r, plotPoints = False)
    plt.savefig('output/task_2/task_2_'+imagesConfig+'_post_rect_withoutPoints.png', dpi = 700, bbox_inches='tight')
    

# input Paths


# Left _ right
path_right = 'images/task_2/right_0.png'
path_left = 'images/task_2/left_0.png'
runExperiment(path_right, path_left, 'right', 'left', saveParam = True)

# Left _ left
path_right = 'images/task_2/left_0.png'
path_left = 'images/task_2/left_1.png'
runExperiment(path_right, path_left, 'left', 'left', saveParam = False)

# right _ right
path_right = 'images/task_2/right_0.png'
path_left = 'images/task_2/right_1.png'
runExperiment(path_right, path_left, 'right', 'right', saveParam = False)


