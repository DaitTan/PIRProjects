#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:21:24 2021

@author: daittan
"""

import cv2
import glob
import numpy as np
import sys

import matplotlib.pyplot as plt


def visualiseTriangulation(R,T, homoge_points, plotPoints = True):      
    f = 1
    tan_x = 1
    tan_y = 1
     
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    for i in range(len(R)):
        R_prime = R[i]
        t_prime = T[i]
         
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
                        color='k', linewidth=1)
    
    if plotPoints == True:
        x_unhomo = homoge_points[:,0]
        y_unhomo = homoge_points[:,1]
        z_unhomo = homoge_points[:,2]
        ax.scatter(x_unhomo, y_unhomo, z_unhomo)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    return ax

def plot_figures(figures, nrows=1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10,10))
    if(nrows > 1 or ncols > 1):
        for ind,title in enumerate(figures):
            axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
            axeslist.ravel()[ind].set_title(title)
            axeslist.ravel()[ind].set_axis_off()
        plt.tight_layout() # optional
    else:
        for ind,title in enumerate(figures):
            axeslist.imshow(figures[title], cmap=plt.gray())
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



left_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_6/left_*.png"))]
right_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_6/right_*.png"))]


from cv2 import aruco

dic = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)



for i, img in enumerate(left_):
    markers = aruco.detectMarkers(img,dic)   # detect the marker on a single image

    temp = aruco.drawDetectedMarkers(img.copy(), markers[0])
    cv2.imwrite('../../output/task_6/left_%d_aruco_marker.png' % i, temp)

for i, img in enumerate(right_):
    markers = aruco.detectMarkers(img,dic)   # detect the marker on a single image

    temp = aruco.drawDetectedMarkers(img.copy(), markers[0])
    cv2.imwrite('../../output/task_6/right_%d_aruco_marker.png' % i, temp)



objPoints = np.array([(0,0,0),(1,0,0),(1,1,0),(0,1,0)],dtype=np.float64)
objPoints.shape

verbose = True
R_fin = []
T_fin = []
# verbose = False
for i, image in enumerate(left_):

    markers = aruco.detectMarkers(image,dic) 
    
    # rotation vector and translation vector of the camera relative to the objectPoints
    retval, rot_vec, trans_vec = cv2.solvePnP(objPoints, markers[0][0][0], left_intrinsic, distCoeffs_left)
    
    if(verbose):
        print("\nleft_%i.png" % i)
        print("R:\n", cv2.Rodrigues(rot_vec)[0].T)
        print("t:\n", trans_vec*-3)
    
    R = np.eye(3,3)
    R = (cv2.Rodrigues(rot_vec, R)[0]).T
    t = -(np.matmul(R, trans_vec*4))
    R_fin.append(R)
    T_fin.append(t)
      # rotate and translate those camera points
    
visualiseTriangulation(R_fin,T_fin, objPoints, plotPoints = True)
plt.savefig('../../output/task_6/task_6_cameraPose_left.png', dpi = 800, bbox_inches='tight')