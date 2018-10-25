import numpy as np
import cv2
import math
#import holopy as hp
#import imutils
import tifffile 

import timeit
import matplotlib.pyplot as plt

#from skimage.filters import threshold_sauvola
from skimage.morphology import binary_opening
from skimage.util import invert



start_time=timeit.default_timer()
#load image
image_path="C:/Users/Yip Group/Desktop/june261/secondconcentration/cellulose/Pos0/normalized18.tif"


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
mind=40
steps=2#distanced looping step

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
    
