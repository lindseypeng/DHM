################################################
import tifffile 
import timeit
import numpy as np
import math
import cv2
from skimage.filters import threshold_sauvola
from skimage.filters import threshold_otsu
from skimage.morphology import binary_opening
from scipy.ndimage import gaussian_filter
from skimage.util import invert
import imutils
import pandas as pd
from timeit import default_timer as timer
import os 
#import sys
#if len(sys.argv) !=2: 
#placeread=str(sys.argv[2])


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

folderread="/home/alinsi/Desktop/wpp_2/Pos0"
foldersave="/home/alinsi/Desktop/wpp_2/Pos0"
#    for x in sys.argv:
#        print "Argument:{}".format(x)
###################################################################
#ROIs=pd.read_csv('/home/alinsi/Desktop/10x_4/Pos0/ROIs.csv')
######################################################################
averagedframei=0##this is only for normallization--whats appropriate for normalization?
averagedframef=2005
capturedframei=0#you cant choose 3 or 4 as it will think its 3 channel of colorshh
capturedframef=2005
placeread=folderread+"/img_00000{}_Default_000.tif"#"/img_000000{}_Default_000.tif"
save_path=foldersave+"/MIPelapsed{}.tif"
placesave=foldersave+"/binary{}.tif"


rangenum=range(capturedframei,capturedframef+1)





minN = 1024##change


##enter some parameters for recontruction##
lambda0 = 0.000532
delx=5.32/1024
dely=6.66/1280
i=complex(0,1)
pi=math.pi

#maxd=150#loop82ing of distance in mm from object to CCD , maximum is 22cm, 220mm,minmum is 60mm6=
#mind=30
#steps=5#5
maxd=320#loop82ing of distance in mm from object to CCD , maximum is 22cm, 220mm,minmum is 60mm6=
mind=200
steps=5#5
###0=yes, 1=No normalizaiton step###
normalization=1
##########initialize empty arrays and indexing####################

##index the number of reconstructing distances
imageslices=int((maxd-mind)/steps)#when steps are not in intergers you must convert the index to int
slicerange = np.arange(0, imageslices, 1)    
##initializaitng empty arrays
threeD=np.empty((imageslices,minN,minN))##this is the stack of reconstructed images from a single frame
##captured frames? or imageslices
# this is reference point for finding dp, its a copy of threeD

interval=5
intermediate=int(len(range(averagedframei,averagedframef+1))/interval)
n=0##CHANGE THIS DOUBLE CHECK BEFORE WRITTING SCRIPT


minprojstack=np.empty((interval,minN,minN))
#minprojstack=np.empty((len(rangenum),minN,minN))
threeDPositions=pd.DataFrame()


##############calculate Transfer Function only once #############
   ###smart adaptives, this need to be manually entered or using computer vision to detect##########

start=timer()
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
    ind=int(round((d-mind)/steps))
    den = np.sqrt(d**2+XX**2+YY**2)
    num = np.exp(-i*2*pi/lambda0*den)
    g=i/lambda0*num/den#g is the impulse response of propagation
    GG2[ind,:,:]=np.fft.fft2(g)



#end1=timer()
#print("time for transfer is : ",(end1-start))
#######################normalize image###########################3
start2=timer()

if normalization==0:
    
    stackss=np.float32(np.zeros((1024,1280)))
##remember to use the number of frames for all the captured images for good average affect
##this may differ from the number of frames you actually want to reconstruct
    for f in range(averagedframei,averagedframef+1):
        
        f2="{:04d}".format(f)
#        f2=f
        if os.path.exists(placeread.format((f2)))==False:
#            print "pass"            
            continue##if file dosnt exit 
        else:
            read_path=placeread.format((f2))
            h = (tifffile.imread(read_path))
            h1 = np.array(h).astype(np.float32)
    #        h1=gaussian_filter(h1,sigma=5.0)
#            (Nx,Ny)= h1.shape[:2]
#            minN = min(Nx,Ny)
#            h1 = h1[:,1280-minN:]
#            
            stackss +=h1
    
    averagestack=stackss/(len(range(averagedframei,averagedframef+1)))
    
else:
    averagestack=1
    #averagestack=tifffile.imread('/home/alinsi/Desktop/biofilm/Pos0/reconstruct007.tif')
###############################Reconstruction################################################################
####################every 20 frames at a time######################################################################
#end2=timer()
#print("time for normalization is : ",(end2-start2))    
#start3=timer()

porosities=pd.DataFrame()
number=1
#background=tifffile.imread('/home/alinsi/Desktop/biofilm/Pos0/reconstruct007.tif')
for f in rangenum:
#    f2=f
    f2="{:04d}".format(f)
    if os.path.exists(placeread.format((f2)))==False:
        print "file dosnt exist {}".format(f2)        
        continue##if file dosnt exit 
    else:    
        frameindex=(f-rangenum[0])%interval##################INTERVAL10
        read_path=placeread.format((f2))
        save_path2=placesave.format((f2))
        q = (tifffile.imread(read_path))            
        q1 = np.array(q).astype(np.float32)
    #    q1=gaussian_filter(q1,sigma=5.0)  
        q2= q1/averagestack  
        q3 = q2[:minN,:minN]  
        hh = np.array(q3).astype(np.float32)
        H=np.fft.fft2(hh)
        ###recontruction~~~~~~~~####
        for d in dp:
            ind=int(round((d-mind)/steps))
            
            Rec_image=np.fft.fftshift(np.fft.ifft2(GG2[ind]*H))
            amp_rec_image = np.abs(Rec_image)
            threeD[ind,:,:]=amp_rec_image.astype(np.float32)
            
        threeD=threeD/threeD.max()*255.0
