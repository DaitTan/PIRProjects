#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 21:17:01 2021

@author: tanmay
"""


import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

  
def computEucledianDistance(kp1,kp2):
    return np.sqrt(np.sum(np.square(np.array(kp1.pt) - np.array(kp2.pt))))
    # return np.sqrt(np.sum(np.square(np.array(kp1.pt) - np.array(kp2.pt))))
    
def findLocalMaxima(keypoints, radius):
    kpGlobal = []
    distGlobal = []
    for kp1 in keypoints:
        kp = []
        distl = []
        for kp2,count in zip(keypoints, range(len(keypoints))):
            dist = computEucledianDistance(kp1,kp2)
            if dist<=radius:
                kp.append(count)
                distl.append(dist)
        distGlobal.append(distl)
        kpGlobal.append(kp)
        
    kpIndex = []
    responseGlobal = []
    for i in range(len(kpGlobal)):
        response = []
        maxKpIndex = i
        maxKp = keypoints[maxKpIndex]
        maxKpResponse = maxKp.response
        for j in kpGlobal[i]:
            dummyKp = keypoints[j]
            response.append(dummyKp.response)
            if dummyKp.response >= maxKpResponse:
                maxKpIndex = j
                maxKp = keypoints[maxKpIndex]
                maxKpResponse = maxKp.response
        responseGlobal.append(response)    
        kpIndex.append(maxKpIndex)
    return np.unique(np.array(kpIndex)), kpGlobal, distGlobal, responseGlobal
        
# def obtainCorners(im, chessboardSize):
#     chessboardSize_h = chessboardSize[0]
#     chessboardSize_w = chessboardSize[1]
    
    
#     ret, corners  = cv.findChessboardCorners(im_gray, (chessboardSize_h,chessboardSize_w), None)
#     if ret == True:
#         criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         corners2 = cv.cornerSubPix(im_gray,corners, (3,3), (-1,-1), criteria)
#     else:
#         corners2 = []
#     return corners, corners2, im_gray, ret

def getCorrespondence(path):

    im = cv.imread(path)
    im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    return im, im_gray
    

def getOptimalMatrix(objpoints, imgpoints, criteria, imagesize_h= 480, imagesize_w = 640):
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (imagesize_w,imagesize_h),None,None)
    # newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(imagesize_w,imagesize_w),0,(imagesize_w,imagesize_h))
    return ret, mtx, dist, rvecs, tvecs    
    
def getDataForLineMatch(matches, kp_query, kp_train):
    # query = 
    # train = 
    
    queryIndices = []
    queryPts = []
    
    trainIndices = []
    trainPts = []
    for mat in matches:
        queryIndices.append(mat.queryIdx)
        kp_q = kp_query[mat.queryIdx]
        
        trainIndices.append(mat.trainIdx)
        kp_t = kp_train[mat.trainIdx]
        
        
        point_q = np.array(kp_q.pt)
        queryPts.append(point_q)
        
        point_t = np.array(kp_t.pt)
        trainPts.append(point_t)
        
            
    return queryIndices, queryPts, trainIndices, trainPts

def checkEpipolarConstraints(matches, qPts, tPts, F, epsi = 1e-3):
    qPts = np.array(qPts)
    tPts = np.array(tPts)
    
    modifiedMatches = []
    epsiArr = []
    for ptCount in range(qPts.shape[0]):
        a = np.append(np.reshape(qPts[ptCount,:],[1,2]),1)
        b = np.reshape(np.append(tPts[ptCount,:],1), [3,1])
        valEpsi =  np.matmul(a,np.matmul(F,b))
        epsiArr.append(abs(valEpsi))
        if abs(valEpsi) < epsi:
            modifiedMatches.append(matches[ptCount])
            
    return modifiedMatches, epsiArr
        
        
        
        
# def getPointOfFeatures(finalMatches, kp_r, kp_l):  
def getPointOfFeatures(finalMatches, kp_r, kp_l, rightCamInt, rightCamDist, leftCamInt, leftCamDist):

    dst_right_features_index = [i.trainIdx for i in finalMatches]
    dst_left_features_index = [i.queryIdx for i in finalMatches]

    imagePoint_r = []
    for idx in dst_right_features_index:
        kp = kp_r[idx]
        imagePoint_r.append(kp.pt)
        
    imagePoint_l = []
    for idx in dst_left_features_index:
        kp = kp_l[idx]
        imagePoint_l.append(kp.pt) 
    
    right_undistortedPoints = cv.undistortPoints(np.array(imagePoint_r).reshape(len(imagePoint_r),1,2), rightCamInt, rightCamDist, None)
    left_undistortedPoints = cv.undistortPoints(np.array(imagePoint_l).reshape(len(imagePoint_l),1,2), leftCamInt, leftCamDist, None)
    
        
    return np.array(right_undistortedPoints[:,0,:]).T, np.array(left_undistortedPoints[:,0,:]).T
    # return np.array(imagePoint_r).T,np.array(imagePoint_l).T 
    

def runForImagePair(path_right, path_left, imNo):
        
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
    
    
    im_r, im_gray_r = getCorrespondence(path_right)
    im_l, im_gray_l = getCorrespondence(path_left)
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    
    # undistort
    dst_right = cv.undistort(im_r, right_intrinsic, right_dist, None)
    dst_left = cv.undistort(im_l, left_intrinsic, left_dist, None)
    
    
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp_r = orb.detect(dst_right,None)
    im_right_woSup = cv.drawKeypoints(dst_right,kp_r, outImage = None, color=(255,0,0), flags=0)
    
    kp_indices_r,_,_,_ = findLocalMaxima(kp_r, 7)
    kp_r = [kp_r[i] for i in kp_indices_r]
    kp_r_new, des_r = orb.compute(dst_right, kp_r)
    # draw only keypoints location,not size and orientation
    
    im_right_wSup = cv.drawKeypoints(dst_right,kp_r_new,outImage = None, color=(255,0,0), flags=0)
    
    cv.imwrite('output/task_3/task_3_im_' + imNo +'_LocalMaxSupression_right.png', np.hstack([im_right_woSup, im_right_wSup]))
    
    
    
    kp_l = orb.detect(dst_left,None)
    im_left_woSup = cv.drawKeypoints(dst_left,kp_l, outImage = None, color=(255,0,0), flags=0)
    
    kp_indices_l,_,_,_ = findLocalMaxima(kp_l, 7)
    kp_l = [kp_l[i] for i in kp_indices_l]
    kp_l_new, des_l = orb.compute(dst_left, kp_l)
    # im_left = cv.drawKeypoints(dst_left,kp_l,outImage = None, color=(255,0,0), flags=0)
    im_left_wSup = cv.drawKeypoints(dst_left,kp_l_new, outImage = None, color=(255,0,0), flags=0)
    
    cv.imwrite('output/task_3/task_3_im_'+imNo+'_LocalMaxSupression_left.png', np.hstack([im_left_woSup, im_left_wSup]))
    
    
    
    # query descriptor = left camera images
    # train_descriptor = right cmaera images
    matcher = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck = True) 
    matches = matcher.match(des_l,des_r) 
    
    q_indices, qPts, tIndices, tPts = getDataForLineMatch(matches, kp_l_new, kp_r_new)
    
    # _, epsiArr = checkEpipolarConstraints(matches, qPts, tPts, E, epsi = 0)
    # finalMatches, epsiArr = checkEpipolarConstraints(matches, qPts, tPts, E, epsi = np.quantile(epsiArr,0.5))
    finalMatches, epsiArr = checkEpipolarConstraints(matches, qPts, tPts, F, epsi = 0.6)
    
    
    preEpipolarCon = cv.drawMatches(dst_left, kp_l_new, dst_right, kp_r_new, matches[:len(matches)],None, flags = 2) 
    preEpipolarCon = cv.resize(preEpipolarCon, (1000,650))
    
    postEpipolarCon = cv.drawMatches(dst_left, kp_l_new, dst_right, kp_r_new, finalMatches[:len(finalMatches)],None, flags = 2) 
    postEpipolarCon = cv.resize(postEpipolarCon, (1000,650)) 
    # plt.imshow(cv.cvtColor(np.hstack([preEpipolarCon, postEpipolarCon]), cv.COLOR_BGR2RGB))
    cv.imwrite('output/task_3/task_3_' + imNo + '_matches_left.png', np.hstack([preEpipolarCon, postEpipolarCon]))
    
    
    # Assuming origin of world frame for camera 
    projection_visual_1 = np.append(np.eye(3), np.array([[0,0,0]]).T,1)
    # Roatation and Translation with of camera 2 with respect to camera2
    projection_visual_2 = np.append(R, T, 1)
    
    
    dst_right_features, dst_left_features =  getPointOfFeatures(finalMatches, kp_r_new, kp_l_new, right_intrinsic, right_dist, left_intrinsic, left_dist)
    # Triangulate Points
    homoge_points = cv.triangulatePoints(projection_visual_1,projection_visual_2, dst_right_features, dst_left_features)
    
    # Unhomonizing the homogeneous points
    x_unhomo = homoge_points[0]/homoge_points[3]
    y_unhomo = homoge_points[1]/homoge_points[3]
    z_unhomo = homoge_points[2]/homoge_points[3]
    # x_unhomo = homoge_points[0]/1.
    # y_unhomo = homoge_points[1]/1.
    # z_unhomo = homoge_points[2]/1.
    
    # Plot images
    fig = plt.figure()
      
    plt.scatter(x_unhomo, z_unhomo)
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    # plt.xlim([4,10])
    plt.ylim([4,10])
    plt.savefig('output/task_3/2d_im' + imNo + '.png', dpi = 900, bbox_inches='tight')
       
    fig = plt.figure() 
    ax = fig.add_subplot(1, 1, 1, projection='3d')
      
    ax.scatter(x_unhomo, y_unhomo, z_unhomo)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    # ax.set_xlim([4,10])
    # ax.set_ylim([-100,10])
    ax.set_zlim([4,10])
    plt.savefig('output/task_3/3d_im' + imNo + '.png', dpi = 900,  bbox_inches='tight')
    
    # import plotly.express as px
    # fig = px.scatter_3d(x = x_unhomo, y = y_unhomo, z = z_unhomo)
    # fig.show()



# os.chdir("/home/tanmay/perInRobo_submission/Project_2")
# i = 0, 8, 9
i = 0
i = np.str(i)
print("Image "+ i)
path_right = 'images/task_3_and_4/right_'+i+'.png'
path_left = 'images/task_3_and_4/left_'+i+'.png'
runForImagePair(path_right, path_left, i)

i = 3
i = np.str(i)
print("Image "+ i)
path_right = 'images/task_3_and_4/right_'+i+'.png'
path_left = 'images/task_3_and_4/left_'+i+'.png'
runForImagePair(path_right, path_left, i)

i = 8
i = np.str(i)
print("Image "+ i)
path_right = 'images/task_3_and_4/right_'+i+'.png'
path_left = 'images/task_3_and_4/left_'+i+'.png'
runForImagePair(path_right, path_left, i)

i = 9
i = np.str(i)
print("Image "+ i)
path_right = 'images/task_3_and_4/right_'+i+'.png'
path_left = 'images/task_3_and_4/left_'+i+'.png'
runForImagePair(path_right, path_left, i)


# for i in range(11):
    # i = np.str(i)
    # print("Image "+ i)
    # path_right = 'project_2a/images/task_3_and_4/right_'+i+'.png'
    # path_left = 'project_2a/images/task_3_and_4/left_'+i+'.png'
    # runForImagePair(path_right, path_left, i)

