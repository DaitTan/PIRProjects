#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 17:27:38 2021

@author: daittan
"""

import cv2 as cv
import glob
import numpy as np
import sys
import matplotlib.pyplot as plt


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

def visualiseTriangulation_2(R_o, T_o, homoge_points, plotPoints = True):      
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
    
    
    if plotPoints == True:
        x_unhomo = homoge_points[:,0]
        y_unhomo = homoge_points[:,1]
        z_unhomo = homoge_points[:,2]
        ax.scatter(x_unhomo, y_unhomo, z_unhomo)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax
    

def undistortImage(img, mtx, distCoeffs):
    height,  width = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx,distCoeffs,(width,height),0,(width,height))

    mapx, mapy = cv.initUndistortRectifyMap(left_intrinsic, distCoeffs_left, None, newcameramtx, (width,height), 5)
    dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    x,y,w,h = roi
    return dst[y:y+height, x:x+width]

def drawCorners(im):
    
    retval, corners_2D = cv.findChessboardCorners(im,(6,9))
    
    corner_img = im.copy()
    corner_img = cv.drawChessboardCorners(corner_img,(6,9),corners_2D,retval)
    return corner_img, corners_2D

def obtainUndistortedImagePoints(imgPoints, cameraIntrinsic, cameraDist, R = None, P = None):
    img_points_reshaped = imgPoints[0]
    img_points_reshaped = img_points_reshaped[:,0,:].T
    undistortedPoints = cv.undistortPoints(img_points_reshaped, cameraIntrinsic, cameraDist, None, R, P)
    undistortedPoints = undistortedPoints[:,0,:]
    return undistortedPoints.T


def plotArrayOfFigures(figs, nrows=1, ncols=1):

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    if(nrows > 1 or ncols > 1):
        for ind,title in enumerate(figs):
            axeslist.ravel()[ind].imshow(figs[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout() # optional
    else:
        for ind,title in enumerate(figs):
            axeslist.imshow(figs[title], cmap=plt.gray())
            axeslist.set_title(title)
            axeslist.set_axis_off()


left_intrinsic = np.array([[423.27381306, 0, 341.34626532],
                           [0, 421.27401756, 269.28542111],
                           [0, 0, 1]])

right_intrinsic = np.array([[420.91160482, 0, 352.16135589],
                            [0, 418.72245958, 264.50726699],
                            [0, 0, 1]])

distCoeffs_left = np.array([-0.43394157423038077, 0.26707717557547866,
                             -0.00031144347020293427, 0.0005638938101488364,
                             -0.10970452266148858])
distCoeffs_right = np.array([-0.4145817681176909, 0.19961273246897668,
                             -0.00014832091141656534, -0.0013686760437966467,
                             -0.05113584625015141])

leftImages = [cv.imread(image) for image in sorted(glob.glob("../../images/task_5/left_*.png"))]
rightImages = [cv.imread(image) for image in sorted(glob.glob("../../images/task_5/right_*.png"))]


rightImages_undistorted = [undistortImage(rightImages[i], right_intrinsic, distCoeffs_right) for i in range(len(rightImages))]
leftImages_undistorted = [undistortImage(leftImages[i], left_intrinsic, distCoeffs_left) for i in range(len(leftImages))]


retval, corners_2D = cv.findChessboardCorners(leftImages[0],(6,9))

corner_img = leftImages[0].copy()
corner_img_2 = cv.drawChessboardCorners(corner_img,(6,9),corners_2D,retval)

leftImage_undist_corners = []
left_undist_corners = []
for i in range(len(leftImages_undistorted)):
    a,b = drawCorners(leftImages_undistorted[i])
    leftImage_undist_corners.append(a)
    left_undist_corners.append(b)

rightImage_undist_corners = []
right_undist_corners = []
for i in range(len(rightImages_undistorted)):
    a,b = drawCorners(rightImages_undistorted[i])
    rightImage_undist_corners.append(a)
    right_undist_corners.append(b)

figDict = {'Left Image 0': leftImages[0],
           'Left Image 1': leftImages[1],
           'Right Image 0': rightImages[0],
           'Right Image 1': rightImages[1],
           'Left Image Undistorted 0': leftImage_undist_corners[0],
           'Left Image Undistorted 1': leftImage_undist_corners[1],
           'Right Image Undistorted 0': rightImage_undist_corners[0],
           'Right Image Undistorted 1': rightImage_undist_corners[1],}


plotArrayOfFigures(figDict,2,4)
plt.savefig('../../output/task_5/task_5_initialImages.png', dpi = 800,bbox_inches='tight')
  

dstPoints = []

for i in range(9):
    for j in range(6):
        dstPoints.append([250 + i*8, 400 - j*8])
        
dstPoints = np.array(dstPoints)
dstPoints = dstPoints.reshape((54,1,2))
dstPoints.shape

np.set_printoptions(suppress=True)
H, mask = cv.findHomography(left_undist_corners[0], dstPoints)
print(np.round(H,2))
leftImage_0_warped = cv.warpPerspective(leftImages_undistorted[0].copy(), H, (leftImages_undistorted[0].shape[:2][1],leftImages_undistorted[0].shape[:2][0]))

H, mask = cv.findHomography(left_undist_corners[1], dstPoints)
print(np.round(H,2))
leftImage_1_warped = cv.warpPerspective(leftImages_undistorted[1].copy(), H, (leftImages_undistorted[1].shape[:2][1],leftImages_undistorted[1].shape[:2][0]))

H, mask = cv.findHomography(right_undist_corners[0], dstPoints)
print(np.round(H,2))
rightImage_0_warped = cv.warpPerspective(rightImages_undistorted[0].copy(), H, (rightImages_undistorted[0].shape[:2][1],rightImages_undistorted[0].shape[:2][0]))

H, mask = cv.findHomography(right_undist_corners[1], dstPoints)
print(np.round(H,2))
rightImage_1_warped = cv.warpPerspective(rightImages_undistorted[1].copy(), H, (rightImages_undistorted[1].shape[:2][1],rightImages_undistorted[1].shape[:2][0]))



figDict = {'Warped Perspective Left 0':leftImage_0_warped,
           'Warped Perspective Left 1':leftImage_1_warped,
           'Warped Perspective Right 0':rightImage_0_warped,
           'Warped Perspective Right 1':rightImage_1_warped,
           }
plotArrayOfFigures(figDict,2,2)

plt.savefig('../../output/task_5/task_5_warpedPerspective.png', dpi = 800,bbox_inches='tight')
    

objp = np.zeros((6* 9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = objp+[0,0,4]


retval, corners_2D = cv.findChessboardCorners(leftImage_0_warped,(9,6))
if retval:
    corner_img = leftImage_0_warped.copy()

    
    retval, rot_vec, trans_vec = cv.solvePnP(objp, corners_2D[:,0,:], left_intrinsic, distCoeffs_left)
    
    R = np.eye(3,3)
    R = (cv.Rodrigues(rot_vec, R)[0]).T
    t = -(np.matmul(R, trans_vec))
    t
    visualiseTriangulation_2(R, t, objp, plotPoints = True)
    plt.savefig('../../output/task_5/task_5_triangulation.png', dpi = 700,bbox_inches='tight')
    