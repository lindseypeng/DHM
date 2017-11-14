#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 15:37:06 2017

@author: alinsi
"""



################################################
import tifffile 
import timeit
import numpy as np
import math
import cv2
from skimage.filters import threshold_sauvola
from skimage.morphology import binary_opening
from skimage.util import invert
import imutils
import pandas as pd

#import sys
#if len(sys.argv) !=2: 
#placeread=str(sys.argv[2])

start=timeit.default_timer()
###################################################################

#
#import sys
#
#if len(sys.argv) <2:
#    print"at least 2 arguments needed"
#    sys.exit(1)
#else:
#    folderread=sys.argv[0]
#    foldersave=sys.argv[1]

folderread="/home/alinsi/Desktop/Pos0"
foldersave="/home/alinsi/Desktop/Pos0"
#    for x in sys.argv:
#        print "Argument:{}".format(x)


######################################################################
averagedframe=10##this is only for normallization--whats appropriate for normalization?

capturedframei=0##you cant choose 3 or 4 as it will think its 3 channel of colors
capturedframef=503

placeread=folderread+"/img_000000{}_Default_000.tif"
save_path=foldersave+"/MIPelapsed{}.tif"
placesave=foldersave+"/reconstruct{}.tif"


rangenum=range(capturedframei,capturedframef+1)





minN = 1024

window_size = 101##for thresholding window
##enter some parameters for recontruction##
lambda0 = 0.000488
delx=5.32/1024
dely=6.66/1280
i=complex(0,1)
pi=math.pi

maxd=52#looping of distance in mm from object to CCD , maximum is 22cm, 220mm,minmum is 60mm6=
mind=35
steps=1#5
###0=yes, 1=No normalizaiton step###
normalization=1
##########initialize empty arrays and indexing####################

##index the number of reconstructing distances
imageslices=int((maxd-mind)/steps)#when steps are not in intergers you must convert the index to int
slicerange = np.arange(0, imageslices, 1)    
##initializaitng empty arrays
threeD=np.empty((imageslices,minN,minN))##this is the stack of reconstructed images from a single frame
##captured frames?? or imageslices
# this is reference point for finding dp, its a copy of threeD

interval=10

intermediate=int(averagedframe/interval)
n=1


minprojstack=np.empty((interval,minN,minN))
#minprojstack=np.empty((len(rangenum),minN,minN))
threeDPositions=pd.DataFrame()


##############calculate Transfer Function only once #############
   ###smart adaptives, this need to be manually entered or using computer vision to detect##########

start_time=timeit.default_timer()
dp = np.arange(mind, maxd, steps)

####################################################################


#distanced looping step

xmax=minN*delx/2
ymax=minN*dely/2

nx = np.arange (-minN/2,minN/2,1)
ny = np.arange (-minN/2,minN/2,1)

X=nx*delx
Y=ny*dely

[XX,YY]=np.meshgrid(X,Y)



#########transfer function only needs to be calculated once for everything ##################
GG2=np.zeros((imageslices,minN,minN),dtype=np.complex64)


for d in dp:
    ind=int((d-mind)/steps)
    
    start_time=timeit.default_timer()
    den = np.sqrt(d**2+XX**2+YY**2)
    num = np.exp(-i*2*pi/lambda0*den)
    g=i/lambda0*num/den#g is the impulse response of propagation
    GG2[ind,:,:]=np.fft.fft2(g)

#######################normalize image###########################3


if normalization==0:
    
    stackss=np.float32(np.zeros((minN,minN)))
##remember to use the number of frames for all the captured images for good average affect
##this may differ from the number of frames you actually want to reconstruct
    for f in rangenum:
        f2="{:03d}".format(f)
        read_path=placeread.format((f2))

        
        h = (tifffile.imread(read_path))
        h1 = np.array(h).astype(np.float32)
        (Nx,Ny)= h1.shape[:2]
        minN = min(Nx,Ny)
        h1 = h1[:minN,:minN]
        
        stackss +=h1
    
    averagestack=stackss/(averagedframe)
    
    
else:
    averagestack=1
###############################Reconstruction################################################################
####################every 20 frames at a time######################################################################

for f in rangenum:

    f2="{:03d}".format(f)
    ##index for frames in threed positions##
    frameindex=(f-rangenum[0])%interval##################INTERVAL10
    read_path=placeread.format((f2))
    save_path2=placesave.format((f2))
    q = (tifffile.imread(read_path))
        
    q1 = np.array(q).astype(np.float32)

    q2 = q1[:minN,:minN]
    q3 = q2/averagestack
    
    hh = np.array(q3).astype(np.float32)
    
    
    ###if needed pre treat this stack for gpu and save into specific location
    
    H=np.fft.fft2(hh)
    ###recontruction~~~~~~~~####
    for d in dp:
        ind=int((d-mind)/steps)
        Rec_image=np.fft.fftshift(np.fft.ifft2(GG2[ind]*H))
        amp_rec_image = np.abs(Rec_image)
        threeD[ind,:,:]=amp_rec_image.astype(np.float32)
        
    tifffile.imsave(save_path2,threeD.astype(np.float32))  ##stop this  

#    
#    ########maxproj##########################
    maxproj=np.ndarray.min(threeD,axis=0)

  

    ##threshold to find 
    maxproj*=255.0/maxproj.max()

#

    start_time=timeit.default_timer()
    thresh_sauvola = threshold_sauvola(maxproj, window_size=window_size, k=0.8, r=180)
    
    binary_sauvola = maxproj > thresh_sauvola
    
    opening=binary_opening(invert(binary_sauvola),np.ones((1,1),np.uint8))
    opening=opening.astype(np.uint8)
    

    minprojstack[frameindex,:,:]=opening

    maxproj=maxproj.astype(np.uint8)
    ##save binarized MIP###################

    ###
    color=cv2.cvtColor(maxproj,cv2.COLOR_GRAY2RGB)
#    
    cnts=cv2.findContours(opening.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[0] if imutils.is_cv2() else cnts[1]

    start_time=timeit.default_timer()
    #cXs=[]#a list of center x position of particles
    #cYs=[]#a list of center y position of particles
   # metricarray=[]
   
 #i need to create a 3d array for recording positions   
#    threeDPosition=np.empty((len(rangenum),len(cnts),3))
    threeDPosition=np.empty((interval,len(cnts),3))
#    #index for particles identified
    index=range(len(cnts))
    ###loop over the contours
    for k,c in enumerate (cnts):
         M = cv2.moments(c)
         (x,y,w,h) = cv2.boundingRect(c)
         if w > 1 & h > 1:
             cX = int(M["m10"] / M["m00"])
             cY = int(M["m01"] / M["m00"])
         else:
             (cX,cY,w,h) = cv2.boundingRect(c)
         metricarray=[]  
         
         for d2 in dp:##becareful d was used before, so use d2, ind was also used, use ind2
             ind2=int((d2-mind)/steps)
             
             particle = threeD[ind2,y:y+h,x:x+w]
#                        
             Metric=np.linalg.norm(particle)#CALCULATE AREA INTENSITY OF THE PARTICULAR PARTICLES IN EACH RECONSTRUCTION STEP
             #print ("The norm iof particle {} at distance{}mm is {} ".format(i,d,Metric))
             metricarray=np.append(metricarray,[Metric])# A LIST OF INTENSITY OF THE OBJECT OF INTERST AT EACH RECONSTRUCTION STEP
 

         minimumvalue=min(metricarray)
         minimumindex=np.argmin(metricarray)
         minimumdp=dp[minimumindex]
         #print ('the minimum value of {} is {} at index {} and distance {}'.format(i,minimumvalue,minimumindex,minimumdp))
         
         threeDPosition[frameindex,k,:]=cX,cY,minimumdp
         
         threeDpositions=pd.DataFrame( {'frame':f,'particle': k, 'cX': cX, 'cY': cY, 'mindp': minimumdp},index=[str(f)])
         threeDPositions=threeDPositions.append(threeDpositions,ignore_index=True)

    if ((f+1)%interval==0):
        threeDPositions.to_csv('/home/alinsi/Desktop/Pos0/threeDPositions{}.csv'.format(n))
        f3="{:03d}".format(n)
        savepath=save_path.format((f3))
        tifffile.imsave(savepath,minprojstack.astype(np.float32))
        minprojstack=np.empty((interval,minN,minN))#reset minprojstack
        threeDPositions=pd.DataFrame()#reset 3d posiitions frame
        threeDPosition=np.empty((interval,len(cnts),3))#resetmin
        n=n+1
    else:
        pass
