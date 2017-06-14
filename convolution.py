
  
import numpy as np
import cv2
import math





##from skimage.morphology import binary_closing


#load image series from a directory using for loop
width=1280
height=1024
numbers=20#number of time seried images 
(Nx,Ny) = (width,height)
minN = min(Nx,Ny)
stack=np.float32(np.zeros((height,width)))

rangenum=range(numbers)##0,1,2,3,4,..
##replacing the numbers in  the right foramt

placeholder="/home/alinsi/Desktop/june14beadsonglass7cm/Pos0/img_0000000{}_Default_000.tif"
for f in rangenum:##looping through every single image##
    
   
    f2="{:02d}".format(f)
    image_path=placeholder.format((f2))
    h = np.float32(cv2.imread(image_path,0))

    stack +=h 

averagestack=stack/numbers
newrawstack=np.float32(np.zeros((numbers,height,width)))
minprojstack = np.empty((numbers,minN,minN))

for f in rangenum:##looping through every single image##

    f2="{:02d}".format(f)
    image_path=placeholder.format((f2))
    h = np.float32(cv2.imread(image_path,0))
    newraw=h/averagestack
    newrawstack[f,:,:] += newraw

    h1 = np.array(newraw).astype(float)

    (Nx,Ny)= h1.shape[:2]
#h=cv2.imread('/home/alinsi/Desktop/june14beadsonglass7cm/Pos0/normalized.jpg',0)       


    #h1 = np.array(h).astype(float)
    h1 = h1[:minN,:minN]
    
    
    (Nx2,Ny2) = h1.shape[:2]
    
    lambda0 = 0.000488
    delx=5.32/1024
    dely=6.66/1280
    i=complex(0,1)
    pi=math.pi
    maxd=70#looping of distance in mm from object to CCD , maximum is 22cm, 220mm,minmum is 60mm6
    mind=40
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
    
    
    threeD=np.empty((imageslices,minN,minN))##initializing for stack
    
    
    #threeDF=np.empty((imageslices,minN,minN))
    for d in dp:
        ind=(d-mind)/steps
        num = np.exp(-i*2*pi/lambda0*np.sqrt((d**2+XX**2+YY**2)))
        den = np.sqrt(d**2+XX**2+YY**2)
        g=i/lambda0*num/den#g is the impulse response of propagation
        G=np.fft.fft2(g)
        intermediate=np.real(G*H)
        Rec_image=np.fft.fftshift(np.fft.ifft2(G*H))
        amp_rec_image = np.abs(Rec_image)
        threeD[ind,:,:]=amp_rec_image.astype(np.float32)
        #threeDF[ind,:,:]=intermediate.astype(np.float32)
    #        
        if ind < 1:
            maxproj = amp_rec_image
    #            
    #    ##finding Minimum projection###
    #        
        for p in indd:
           for j in indd:
                if maxproj[p][j] > amp_rec_image[p][j]:
                    maxproj[p][j]=amp_rec_image[p][j]
    #    
    #    
    #hp.show(threeD.transpose())
#    
#    #tifffile.imsave("/home/alinsi/Desktop/detection/stacks")
#    ######################FINDING INFOCUSE XY PLANE#################################

   # maxproj2 = np.array(maxproj*255,dtype=np.uint8)
   # maxproj2=maxproj2*255.0/maxproj2.max()
#    
    minprojstack[f,:,:]=maxproj

#tifffile.imsave("/home/alinsi/Desktop/normalizedstacksminproj",minprojstack)
