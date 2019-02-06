
import tifffile 
import numpy as np
import math
import cv2
from skimage.filters import threshold_sauvola
from skimage.morphology import binary_opening
from skimage.util import invert
import imutils
import pandas as pd
import os 
import argparse

######################################################################################## 
# construct the argument parser and parse the arguments
########################################################################################
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,help="path to input image")
ap.add_argument("-o", "--output", required=True,help="path to output image")
ap.add_argument("-ci", "--capturedframei", required=True,help="starting frame number for reconstruction")
ap.add_argument("-cf", "--capturedframef", required=True,help="end frame number fore reconstruction")
args = vars(ap.parse_args())
########################################################################################
## load the varaibles from argument parser
########################################################################################
folderread = args["input"]
foldersave = args["output"]
capturedframei=int(args["capturedframei"])#you cant choose 3 or 4 as it will1hink its 3 channel of colorshh
capturedframef=int(args["capturedframef"])
rangenum=range(capturedframei,capturedframef+1)
########################################################################################
##file name to read and file name to save##
########################################################################################
placeread=folderread+"/img_00000{}_Default_000.tif"#"/img_00000{}_Default_000.tif"#"/img_000000{}_Default_000.tif"
save_path=foldersave+"/MIPelapsed{}.tif"
placesave=foldersave+"/maxproj{}.tif"
placesave2=foldersave+"/binary{}.tif"
########################################################################################
##parameters that need to be changed manually in the script to match your experiment##
########################################################################################
averagedframei=0##normalization start frame
averagedframef=200##normalization end frame
minN = 512##minimum dimension of the image height or width
lambda0 = 0.000650##wavelength in mm
delx=5.32/1024##camera x 
dely=6.66/1280## camera y 
i=complex(0,1)
pi=math.pi
maxd=300#maximum distance
mind=10##minimum distance 
steps=5#step size 
########################################################################################
##########initialize empty arrays and indexing##########################################
########################################################################################
imageslices=int((maxd-mind)/steps)#total number of imageslices for reconstruction stack 
interval=1##numbers of holograms processed before saving their coordinates
intermediate=int(len(range(averagedframei,averagedframef+1))/interval)
n=0##strating index for saving coordinates file
minprojstack=np.empty((interval,minN,minN))##initiate minimum projection stack
threeDPositions=pd.DataFrame()
threeD=np.empty((imageslices,minN,minN))
########################################################################################
##############calculate Transfer Function for all the distancces in the threeD stack####
########################################################################################
dp = np.arange(mind, maxd, steps)
xmax=minN*delx/2
ymax=minN*dely/2
nx =np.arange (-minN/2,minN/2,1)
ny =np.arange (-minN/2,minN/2,1)
X=nx*delx
Y=ny*dely
[XX,YY]=np.meshgrid(X,Y)
GG2=np.zeros((imageslices,minN,minN),dtype=np.complex64)
for d in dp:
    ind=int(round((d-mind)/steps))
    den = np.sqrt(d**2+XX**2+YY**2)
    num = np.exp(-i*2*pi/lambda0*den)
    g=i/lambda0*num/den#g is the impulse response of propagation
    GG2[ind,:,:]=np.fft.fft2(g)
##############################################################
#######################normalize image########################
##############################################################
normalization=1
if normalization==0:
    stackss=np.float32(np.zeros((1024,1280)))
    for f in range(averagedframei,averagedframef+1):
        f2="{:04d}".format(f)
        if os.path.exists(placeread.format((f2)))==False:           
            continue##if file dosnt exit 
        else:
            read_path=placeread.format((f2))
            h = (tifffile.imread(read_path))
            h1 = np.array(h).astype(np.float32)
            stackss +=h1  
    averagestack=stackss/(len(range(averagedframei,averagedframef+1)))
else:
    averagestack=1
#####################################################################################################
###############################Reconstruction########################################################
#####################################################################################################
porosities=pd.DataFrame()
number=1
for f in rangenum:
    f2="{:04d}".format(f)
    if os.path.exists(placeread.format((f2)))==False:      
        continue##if file dosnt exit 
    else:    
        frameindex=(f-rangenum[0])%interval##################INTERVAL10
        read_path=placeread.format((f2))
        save_path2=placesave.format((f2))
        save_path3=placesave2.format((f2))
        q = (tifffile.imread(read_path))            
        q1 = np.array(q).astype(np.float32)
        q2= q1/averagestack  
        q3 = q2[:minN,:minN]  
        hh = np.array(q3).astype(np.float32)
        H=np.fft.fft2(hh)
        cv2.imshow('croppedimage','q3')
        cv2.waitKey(0)
        ###recontruction~~~~~~~~####
        for d in dp:
            ind=int(round((d-mind)/steps))
            Rec_image=np.fft.fftshift(np.fft.ifft2(GG2[ind]*H))
            amp_rec_image = np.abs(Rec_image)
            threeD[ind,:,:]=amp_rec_image.astype(np.float32)      
        threeD=threeD/threeD.max()*255.0
    ########maxproj##########################
        maxproj=np.ndarray.min(threeD,axis=0)
        maxproj*=255.0/maxproj.max()
        thresh_sauvola = threshold_sauvola((maxproj), window_size=25, k=0.5, r=128)
        binary_sauvola = maxproj > thresh_sauvola    
        opening=binary_opening((binary_sauvola),np.ones((1,1),np.uint8))
        opening=invert(binary_sauvola)
        opening=opening.astype(np.uint8)
            ########CALCULATE POROSITY###################################################################################
        porosity=np.float(np.count_nonzero(opening)/np.size(opening)*100.0)
        
        a=np.float(np.count_nonzero(opening))
        b=np.float(np.size(opening))
        porosity=a/b*100.0 
        maxproj=maxproj.astype(np.uint8)   
        cnts=cv2.findContours((opening.copy()),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts=cnts[0] if imutils.is_cv2() else cnts[1]   
        cXs=[]#a list of center x position of particles
        cYs=[]#a list of center y position of particles
        metricarray=[]
        threeDPosition=np.empty((len(rangenum),len(cnts),3))
        threeDPosition=np.empty((interval,len(cnts),3))
        index=range(len(cnts))
        ###loop over the contours
        for k,c in enumerate (cnts):
             M = cv2.moments(c)
             (x,y,w,h2) = cv2.boundingRect(c)
             (x3,y3), radius= cv2.minEnclosingCircle(c)
    
             if (M["m00"] != 0):  
                 cX = int(M["m10"] / M["m00"])
                 cY = int(M["m01"] / M["m00"])   
                 metricarray=[]  
                 for d2 in dp:##becareful d was used before, so use d2, ind was also used, use ind2
                     ind2=int(round((d2-mind)/steps))
                     particle = threeD[ind2,y:y+h2,x:x+w]
                     Metric=np.linalg.norm(particle)#CALCULATE AREA INTENSITY OF THE PARTICULAR PARTICLES IN EACH RECONSTRUCTION STEP
                     metricarray=np.append(metricarray,[Metric])# A LIST OF INTENSITY OF THE OBJECT OF INTERST AT EACH RECONSTRUCTION STEP
                 minimumvalue=min(metricarray)
                 minimumindex=np.argmin(metricarray)
                 minimumdp=dp[minimumindex]
                 threeDPosition[frameindex,k,:]=cX,cY,minimumdp          
                 threeD2=maxproj[cY-100:cY+100,cX-100:cX+100]
                 if threeD2.size==0:
                     break              
                 threeDposition=pd.DataFrame( {'frame':f,'particle': k, 'cX': cX, 'cY': cY, 'mindp': minimumdp,'radius':radius,'porosity':porosity,'number':number},index=[str(f)])
                 threeDPositions=threeDPositions.append(threeDposition,ignore_index=True)
                 number=number+1
         
             else:
                 pass      
    if ((f+1)%interval==0):
        stackss=np.float32(np.zeros((1024,1280)))
        for u in range(f+1,f+11):
            uu="{:04d}".format(u)
            if os.path.exists(placeread.format((uu)))==False:           
                continue##if file dosnt exit 
            else:
                read_path=placeread.format((uu))
#                print(read_path)
                h = (tifffile.imread(read_path))
                h1 = np.array(h).astype(np.float32)
                stackss +=h1
        averagestack=stackss/(len(range(f+1,f+11)))
        tifffile.imsave(save_path2,maxproj.astype(np.float32))
        tifffile.imsave(save_path3,opening.astype(np.float32))
    else:
        pass

window_size=13
kk=0.05
rr=150
imageinfo=pd.DataFrame( {'mind':mind,'maxd': maxd, 'step_size': steps, 'normalized(0yes1no)':normalization , 'laser(nm)':lambda0,'rr':rr,'kk':kk,'window':window_size},index=[0])
imageinfo.to_csv(args["output"]+'/imageinfoall.csv')
#tifffile.imsave(save_path.format(),minprsojstack.astype(np.float32))
