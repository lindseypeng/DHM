from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import math
#import holopy as hp
import imutils
#import tifffile 


import matplotlib.pyplot as plt

from skimage.filters import (threshold_sauvola)
from skimage.morphology import binary_opening
#from skimage.morphology import binary_closing
from skimage.util import invert

#load image
image_path="/home/alinsi/Desktop/inlinehologram/may11/further/pnc7010x.bmp"

h = cv2.imread(image_path,0)


h1 = np.array(h).astype(float)

(Nx,Ny)= h1.shape[:2]
minN = min(Nx,Ny)
h1 = h1[:minN,:minN]


(Nx2,Ny2) = h1.shape[:2]

lambda0 = 0.000488
delx=5.32/1024
dely=6.66/1280
i=complex(0,1)
pi=math.pi
maxd=90#looping of distance in mm from object to CCD , maximum is 22cm, 220mm,minmum is 60mm6
mind=30
steps=5#distanced looping step

xmax=Nx2*delx/2
ymax=Ny2*dely/2

nx = np.arange (-Nx2/2,Nx2/2,1)
ny = np.arange (-Ny2/2,Ny2/2,1)

X=nx*delx
Y=ny*dely

[XX,YY]=np.meshgrid(X,Y)


Gy = 10
k=2*pi/lambda0

imageslices=(maxd-mind)/steps
minproj=np.zeros([minN,minN,imageslices])

metricarray=[]
dp = np.arange(mind, maxd, steps)
H=np.fft.fft2(h1)

indd=np.arange(1,minN,1)


threeD=np.empty((imageslices,minN,minN))
for d in dp:
    ind=(d-mind)/steps
    num = np.exp(-i*2*pi/lambda0*np.sqrt((d**2+XX**2+YY**2)))
    den = np.sqrt(d**2+XX**2+YY**2)
    g=i/lambda0*num/den#g is the impulse response of propagation
    G=np.fft.fft2(g)
    Rec_image=np.fft.fftshift(np.fft.ifft2(G*H))
    amp_rec_image = np.abs(Rec_image)
    threeD[ind,:,:]=amp_rec_image
 
    if ind < 1:
        maxproj = amp_rec_image
        
##finding Minimum projection###
    
    for p in indd:
       for j in indd:
            if maxproj[p][j] > amp_rec_image[p][j]:
                maxproj[p][j]=amp_rec_image[p][j]

#hp.show(threeD)

#tifffile.imsave("/home/alinsi/Desktop/detection/stacks")
######################FINDING INFOCUSE XY PLANE#################################
window_size = 25
#maxproj = np.array(maxproj*255,dtype=np.uint8)
maxproj2=maxproj*255.0/maxproj.max()

#binary_global = maxproj > threshold_otsu(maxproj)
#thresh_niblack = threshold_niblack(maxproj, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(maxproj2, window_size=window_size, k=0.5, r=128)

#binary_niblack = maxproj > thresh_niblack
binary_sauvola = maxproj2 > thresh_sauvola

#opening=binary_closing(invert(binary_sauvola),np.ones((5,5),np.uint8))

#opening2=binary_opening(opening,np.ones((3,3),np.uint8))
opening=binary_opening(invert(binary_sauvola),np.ones((3,3),np.uint8))
opening=opening.astype(np.uint8)
maxproj2=maxproj2.astype(np.uint8)
color=cv2.cvtColor(maxproj2,cv2.COLOR_GRAY2RGB)


cnts=cv2.findContours(opening.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[0] if imutils.is_cv2() else cnts[1]

cXs=[]
cYs=[]
speeds=[]
#loop over the contours
for i,c in enumerate (cnts):
     M = cv2.moments(c)
     cX = int(M["m10"] / M["m00"])
     cY = int(M["m01"] / M["m00"])
     (x,y,w,h) = cv2.boundingRect(c)
     cXs.append(cX)
     cYs.append(cY)
     cv2.drawContours(color, [c], -1 , (0, 255, 0), 1)
     #cv2.circle(color, (cX, cY), 1, (255, 255, 255), -1)
     cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,255),2)
     cv2.putText(color, "# {}".format(i), (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
     #######calculate intensity changes######
     localmaxproj = maxproj[y:y+h+h,x:x+w+w]
     peak=localmaxproj.max()
     valley=localmaxproj.min()
     speed=(peak-valley)/(h+w)
     speeds.append(speed)
     
     
     for d in dp:
        ind=(d-mind)/steps
        particle = threeD[ind,y:y+h,x:x+w]
        Metric=np.linalg.norm(particle)
        metricarray=np.append(metricarray,[Metric])
        
cv2.imshow("labeled",color)




metricarray2=metricarray.reshape(len(cnts),len(dp))

#the minimum of rows is axis=1, col is axis=0

minimumvalue=metricarray2.min(axis=1)
#this gives the minimum value of each xy object

minimumdp=np.argmin(metricarray2,axis=1)
ranges = np.arange(0, len(minimumdp), 1)

plt.plot(ranges,speeds)
plt.show()

