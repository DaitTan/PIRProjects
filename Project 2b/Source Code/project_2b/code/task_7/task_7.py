#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 21:37:21 2021

@author: daittan
"""


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

  

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

# def getCorrespondence(path):
#     im_gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#     return im, im_gray
    

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
    

        
left_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_7/left_*.png"))]
right_ = [cv2.imread(image) for image in sorted(glob.glob("../../images/task_7/right_*.png"))]

im_gray_l = [cv.cvtColor(left_[i], cv.COLOR_BGR2GRAY) for i in range(len(left_))]
im_gray_l = [cv.cvtColor(right_[i], cv.COLOR_BGR2GRAY) for i in range(len(right_))]


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


# im_r, im_gray_r = getCorrespondence(path_right)
# im_l, im_gray_l = getCorrespondence(path_left)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# undistort
dst_im_1 = cv.undistort(left_[0], left_intrinsic, distCoeffs_left, None)
dst_im_2 = cv.undistort(left_[1], left_intrinsic, distCoeffs_left, None)


orb = cv.ORB_create(nfeatures = 1000)
# find the keypoints with ORB
kp_1 = orb.detect(dst_im_1,None)
im_1_woSup = cv.drawKeypoints(dst_im_1,kp_1, outImage = None, color=(255,0,0), flags=0)

kp_indices_1,_,_,_ = findLocalMaxima(kp_1, 0)
kp_1 = [kp_1[i] for i in kp_indices_1]
kp_1_new, des_1 = orb.compute(dst_im_1, kp_1)
# draw only keypoints location,not size and orientation

im_1_wSup = cv.drawKeypoints(dst_im_1,kp_1_new,outImage = None, color=(255,0,0), flags=0)

cv.imwrite('../../output/task_7/task_7_im_' + str(1) +'_LocalMaxSupression_right.png', np.hstack([im_1_woSup, im_1_wSup]))



kp_2 = orb.detect(dst_im_2,None)
im_2_woSup = cv.drawKeypoints(dst_im_2,kp_2, outImage = None, color=(255,0,0), flags=0)

kp_indices_2,_,_,_ = findLocalMaxima(kp_2, 0)
kp_2 = [kp_2[i] for i in kp_indices_2]
kp_2_new, des_2 = orb.compute(dst_im_2, kp_2)
# draw only keypoints location,not size and orientation

im_2_wSup = cv.drawKeypoints(dst_im_2,kp_2_new,outImage = None, color=(255,0,0), flags=0)

cv.imwrite('../../output/task_7/task_7_im_' + str(2) +'_LocalMaxSupression_right.png', np.hstack([im_2_woSup, im_2_wSup]))



# query descriptor = left camera images
# train_descriptor = right cmaera images
matcher = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck = True) 
matches = matcher.match(des_1,des_2) 
inliers_im = cv.drawMatches(dst_im_1, kp_1_new, dst_im_2, kp_2_new, matches, None, flags = 2)
plt.imshow(inliers_im)
plt.axis('off')
plt.savefig('../../output/task_7/task_7_matches_before.png', dpi = 700,bbox_inches='tight')



q_indices, qPts, tIndices, tPts = getDataForLineMatch(matches, kp_1_new, kp_2_new)

# _, epsiArr = checkEpipolarConstraints(matches, qPts, tPts, E, epsi = 0)
# finalMatches, epsiArr = checkEpipolarConstraints(matches, qPts, tPts, E, epsi = np.quantile(epsiArr,0.5))
E, mask = cv.findEssentialMat(np.array(qPts), np.array(tPts), left_intrinsic)
E = np.mat(E)

inliers = []
for ind,mask_ in enumerate(mask):
    if mask_[0]:
        inliers.append(matches[ind])


qPts_ = []
tPts_ = []

for ind,match in enumerate(inliers):
    qPts_.append(kp_1_new[match.queryIdx].pt)
    tPts_.append(kp_2_new[match.trainIdx].pt)
    
qPts_ = np.array(qPts_)
tPts_ = np.array(tPts_)

inliers_im = cv.drawMatches(dst_im_1, kp_1_new, dst_im_2, kp_2_new, inliers, None, flags = 2)
plt.imshow(inliers_im)
plt.axis('off')
plt.savefig('../../output/task_7/task_7_matches_after.png', dpi = 700,bbox_inches='tight')


_, R, T, mask_2 = cv.recoverPose(E, qPts_, tPts_, left_intrinsic)

_, R, T, mask_2, traingulatedPoints = cv.recoverPose(E, qPts_, tPts_, left_intrinsic, 10, R, T, mask_2)

R0 = np.eye(3)
    # Roatation and Translation with of camera 2 with respect to camera2
T0 = np.zeros((3,1))

visualiseTriangulation(R0,T0, R, T*5, traingulatedPoints, plotPoints = True)
plt.savefig('../../output/task_7/task_7_cameraPose.png', dpi = 800, bbox_inches='tight')
