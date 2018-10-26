import numpy as np
import cv2
import math
import tifffile 
import timeit
import matplotlib.pyplot as plt
from skimage.morphology import binary_opening
from skimage.util import invert

#load image
image_path="C:/Users/Yip Group/Desktop/june261/secondconcentration/cellulose/Pos0/normalized18.tif"
###################################################Set Up########################################################
##Input hologram##
h1 = cv2.imread(image_path,0)
##Extract row and column pixel sizes##
(Nx2,Ny2) = h1.shape[:2]
##Input Variable##
lambda0 = 0.000488  ##wavelength of laser
delx=5.32/1024      ##row pitch size    
dely=6.66/1280      ##column pitch size     
i=complex(0,1)      ##complex number
pi=math.pi          
maxd=90             ##max reconstruction distance in mm
mind=80             ##min reconstruction distance in mm
steps=1             ##step sizes of reconstruction in mm
imageslices=(maxd-mind)/steps ##number of reconstruction images
minproj=np.zeros([minN,minN,imageslices]) ##create placeholder for an stack of reconstruction images 
metricarray=[]
dp = np.arange(mind, maxd, steps)         #list containing reconstruction steps to be iterated
indd=np.arange(1,minN,1)    ##index to step through
##Set up Spatial coordinates of impulse function that matches with hologram aperture size##
xmax=Nx2*delx/2     
ymax=Ny2*dely/2
nx = np.arange (-Nx2/2,Nx2/2,1)
ny = np.arange (-Ny2/2,Ny2/2,1)
X=nx*delx
Y=ny*dely
[XX,YY]=np.meshgrid(X,Y)
##Set up Impulse Function##
k=2*pi/lambda0 ##wave vector
###################################################Normalization########################################################
averagedframei=0        ##start frame for stacks used for normalization
averagedframef=20       ##end frame for stacks used for normalization
stackss=np.float32(np.zeros((1024,1280)))
for f in range(averagedframei,averagedframef+1): ##adding all the image together 
    f2="{:04d}".format(f)
    if os.path.exists(placeread.format((f2)))==False:
        continue            ##continue if file dosnt exit 
        else:
            read_path=placeread.format((f2))
            h = (tifffile.imread(read_path))
            h1 = np.array(h).astype(np.float32)
            stackss +=h1    
averagestack=stackss/(len(range(averagedframei,averagedframef+1)))##dividing sum of images by the number of images
h1=h1/averagestack #Normalization step used to remove uneven background and DC term
###################################################Reconstruction##########################################################
## Convolution Steps
H=np.fft.fft2(h1) ##fast 2d fourier trnasform of hologram
for d in dp: 
    num = np.exp(-i*2*pi/lambda0*np.sqrt((d**2+XX**2+YY**2))) ##numerator of impulse function
    den = np.sqrt(d**2+XX**2+YY**2) ##denominator of impulse function
    g=i/lambda0*num/den# impulse response of propagation
    G=np.fft.fft2(g) ##fast 2d fourier trnasform of impusle function
    Rec_image=np.fft.fftshift(np.fft.ifft2(G*H)) ##obtaining reconstructed field from inverse fourier transform product 
    amp_rec_image = np.abs(Rec_image) ##the real component is the intensity 

    
