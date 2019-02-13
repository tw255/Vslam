import numpy as np  
import cv2
    
from skimage.measure import ransac
#from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib

import math

#import pyglet

#from pyglet.gl import *

cap = cv2.VideoCapture('drivebridge.mp4')

w,h = 1080, 620
resizewindow = (w, h)
prev = None
xx,yy,zz = [],[],[]
cu,cv,cw = [],[],[]
## principle point
cx = w//2
cy = h//2
##camera matrix, fundamentalmatrix?
#f = 690*.7
#690 is value found through "np.median(festaverage)"
#f is the focal length
f = 690
###k is fundamentalmatrix?
k = np.array([[f, 0,cx],
              [0 ,f,cy],
              [0 , 0, 1]])
kinverse = np.linalg.inv(k)

festaverage = []
#a is pts from first and seocnd images
#one is kp1 with a third column filled with 1 to conform 3x3
#then dot prod. of (kinv and one.transformed).transformed
def normalize(a):
    one = np.concatenate([a, np.ones((a.shape[0], 1))], axis=1)
    return np.dot(kinverse, one.T).T[:,0:2]
#should do the opposite of normalize
def denormalize(b):
    bat = np.dot(k, np.array([b[0], b[1], 1.0]))
    return int(round(bat[0])), int(round(bat[1]))    
def triangulatePoints(p1, p2, x1, x2):
    return cv2.triangulatePoints(p1, p2, x1.T, x2.T).T
    
###x'^(T)Fx = 0
#x is point of first image
#x' is point of second image
#F - fundamental matrix
dst = []
dst = np.array(dst)

fc = 1
bat = None
matplotlib.interactive(True)

cpos = []
mx,my,mz = [],[],[]

frames = []
idxs = []
while(cap.isOpened()):
    
    #if fc == 0:
    ret, frame = cap.read()
    #frames.append(frame)
    frame0 = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    orb = cv2.ORB_create()
    pts = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 7)

    kps = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in pts]
    kps, des = orb.compute(frame, kps)

    #for pt in pts:
     #   x,y = pt.ravel()
      #  cv2.circle(frame,(x,y), 3, (0,255,0), -1)
    
    #if fc == 1:
    ret, frame = cap.read()
    bat = []
    idx1,idx2 = [],[]
    frame1 = frame
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    if prev is not None:
        matches = bf.knnMatch(des, prev['des'], k =2)
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                #if 15 < m.distance < 45:
                #if m.distance < 40:
                
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                kp0 = (kps[m.queryIdx].pt)
                kp1 = prev['kps'][m.trainIdx].pt
                bat.append((kp0,kp1))
                
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
        
    #frames[-1] is frame0 and frames[-2] is frame1
    #filter using ransac
    if len(bat) > 0:
        bat = np.array(bat)
        #for b1,b2 in bat:
            
                
###queryidx is kp0 and trainidx is kp1
###kps is kp0 and prev is kp1

##normalize
##bat here is kp0 and kp1
##split up bat back into kp0 and kp1
##defined in the equation as x'^(T) and x (first and second pts)

        bat[:, 0, :] = normalize(bat[:,0,:])
        bat[:, 1, :] = normalize(bat[:,1,:])

        model, inliers = ransac((bat[:, 0], bat[:, 1]),
                                 #FundamentalMatrixTransform,
                                 EssentialMatrixTransform,
                                 min_samples = 8,
                                 residual_threshold=0.001, max_trials=100)
        bat = bat[inliers]
        idx1 = idx1[inliers]
        idx2 = idx2[inliers]
        #for i, idx in enumerate(idx2):
         #   print(idx)
        
        W = np.mat([[0,-1,0],
                    [1, 0,0],
                    [0, 0,1]],dtype=float)
        u,d,vt = np.linalg.svd(model.params)
        assert np.linalg.det(u) > 0
        if np.linalg.det(vt) < 0:
            vt *= -1.0
        #R without W.T is main one chosen since output has diagnol
        R = np.dot(np.dot(u,W), vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(u,W.T), vt)
        #R1 = np.dot(np.dop(u,W), vt))
        #R2 = np.dot(np.dot(u,W.T), vt)
###pulls value from the third column of u 
        t = u[:, 2]
        b = t
        #create the 3x4 Rt matrix
        Rt = np.concatenate([R, t.reshape(3,1)], axis=1)

        idenRt = np.eye(4)
        
        fr0 = np.dot(Rt, idenRt)

        pts4d = triangulatePoints(idenRt[0:3,:], fr0,
                                  bat[:,0], bat[:,1])
        
                
        

        good4d1 = np.abs(pts4d[:,3]) > 0.005

        
        '''for i,p in enumerate(pts4d):
            if not good4d1[i]:
                continue
            #frames.append(frame0)
            idxs.append(idx1[i])
            #frames.append(frame1)
            idxs.append(idx2[i])
    
            #print(idxs[i])'''
        pts4d = pts4d[good4d1]


        #good4d2 = pts4d[:,2] > 0
        #pts4d = pts4d[good4d2]

        #convert to euclidean coord
        pts3d = []
        #for ptsd in pts4d:
            #both ways result in the same answer            
            #dst = cv2.conv ertPointsFromHomogeneous(ptsd.T)
         #   ptsd = ptsd[:-1]/ptsd[-1]
          #  pts3d.append(ptsd)
        pts4d /= pts4d[:, 3:]
        pts3d = pts4d[:,0:3]
        pts3d = np.array(pts3d)
        #print(pts3d,'pts3d')

        mx.append(b[0])
        my.append(b[1])
        mz.append(b[2])

        #xx,yy,zz = [],[],[]
        #cu,cv,cw = [],[],[]
        if fc%2 == 0:

            #Camera position
            cpos1 = np.dot(-(R.T), t)
            #cpos1[:,2] = cpos1[:,2] + 5
            cpos1 = np.array(cpos1)

            for cx,cy,cz in cpos1:
                cx = cx + sum(mx)
                cu.append(cx)
                cy = cy + sum(my)
                cv.append(cy)
                cz = cz + sum(mz)
                cw.append(cz)

            for x,y,z in pts3d:
                x = x + sum(mx)
                xx.append(x)
                y = y + sum(my)
                yy.append(y)
                z = z + sum(mz)
                zz.append(z)  

            
            ax = plt.axes(projection='3d')
            #ax.view_init(azim=0, elev=90)
            ax.set_axis_off()
            ax.set_xlim(-50,50)
            ax.set_ylim(-50,50)
            ax.set_zlim(-50,50)
            plt.autoscale(False)

            ax.scatter3D(cu, cv,cw, s = 50 ,cmap='Greens',depthshade=False)
            ax.scatter3D(xx,yy,zz,s = 1 ,cmap='Greens',depthshade=False)
            plt.pause(0.05)
        

                
        #s,v,d = np.linalg.svd(model.params)
        #fest = np.sqrt(2)/((v[0]+v[1]/2))
        #festaverage.append(fest)
        #print(fest, np.median(festaverage))
  
    
#denormalize to bring back points to reg pos
#if k is F then F is dot with p1 and
    for p1, p2 in bat:
        x1, y1 = denormalize(p1)
        x2, y2 = denormalize(p2)
        
        cv2.circle(frame,(x1,y1), 3, (0,255,0), -2)
        cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 1, 8)
######update old points that are present in new frames and delete the old ones
######think use idx on bat to 
    
        '''xy1 = (x1,y1)
        xy2 = (x2,y2)
        kl.append((xy1,xy2))
    kl = np.array(kl)
    if len(kl) > 0:
        dne = False
        for k1, k2 in kl:
            dist1 = math.sqrt(abs(math.pow((k1[0] - w//2), 2) +
                                  math.pow((k1[1] - h//2), 2)))
            if dist1 < 40 and dne == False:
                dist2 = math.sqrt(abs(math.pow((k2[0] - k1[0]), 2) +
                                      math.pow((k2[1] - k1[1]), 2)))
                dne = True'''
                            
                


    #print(len(bat))
    #print(fc)
    #if fc%2 == 1:
    #bat0 = bat
        #print(frame, fc)
    fc += 1
    prev = {"kps": kps, "des" : des}
    
    if len(bat) > 0:
        cv2.imshow('frame', cv2.resize(frame, resizewindow))

      
    #ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

cap.release()
cv2.destroyAllWindows()
