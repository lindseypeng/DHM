import numpy as np
import cv2
import math
#import holopy as hp
import imutils
#import tifffile 


import matplotlib.pyplot as plt

from skimage.filters import (threshold_sauvola)
from skimage.morphology import binary_opening
from skimage.util import invert

#load image
image_path="/home/alinsi/Desktop/inlinehologram/may11/further/pnc7010x.bmp"
#image_path2="/home/owner/hologram/samples/dec22_3.tif"
#image_path3="/home/owner/hologram/samples/dec22_2.tif"

h = cv2.imread(image_path,0)
#plt.imshow(h,cmap='Greys_r')
#k = cv2.imread(image_path2,0)
#l = cv2.imread (image_path3,0)

h1 = np.array(h).astype(float)

#plt.imshow(k,cmap='Greys_r')
#plt.imshow(h1,cmap='Greys_r')
#
# #load background
# back = cv2.imread('/home/owner/hologram/samples/background2.tif',0)
# background = back.astype(float)
# # #subtract background from image
# h1=h1-background

(Nx,Ny)= h1.shape[:2]
minN = min(Nx,Ny)
h1 = h1[:minN,:minN]
#hp.save('/home/alinsi/Desktop/lasersquare',h1)

(Nx2,Ny2) = h1.shape[:2]

#plt.imsave("/home/alinsi/Desktop/inlinehologram/processed/may1microbeads5original",h1,cmap='Greys_r')
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

# X2=nx/(Nx2*delx)
# Y2=ny/(Ny2*dely)
#
# [XX2,YY2]=np.meshgrid(X2,Y2)
#gy maginification factor for the reconstruction
Gy = 10
k=2*pi/lambda0



#mag0 = np.arange[0,10,5]
#mag = 15
imageslices=(maxd-mind)/steps
threeD=np.zeros([minN,minN,imageslices])
imaginary=np.zeros([minN,minN,imageslices])
original=np.zeros([minN,minN,imageslices])

#maxproj=np.ones([minN,minN])*8270177
#maxproj=np.array(maxproj,dtype=np.uint8)
minproj=np.zeros([minN,minN,imageslices])

metricarray=[]
dp = np.arange(mind, maxd, steps)
H=np.fft.fft2(h1)

indd=np.arange(1,minN,1)

for d in dp:
    ind=(d-mind)/steps
    


    #d0= dp*mag
    #f= (1/dp + 1/d0)**(-1)
    #L = np.exp(i*pi/lambda0/f*(XX**2,YY**2)) #reference wave
    #P = #factor , correction of reconstructed wave field caused by lens(phase aberrations)

    num = np.exp(-i*2*pi/lambda0*np.sqrt((d**2+XX**2+YY**2)))
    den = np.sqrt(d**2+XX**2+YY**2)
    g=i/lambda0*num/den#g is the impulse response of propagation
    #############################################
    # SPHEREICAL Waves##
    # zi = -Gy * d;
    # zc = 1 / (1 / d + 1 / zi)
    # sphere = np.exp(i * k/2/zc*(XX**2+YY**2))

    #################################################

    G=np.fft.fft2(g)

    Rec_image=np.fft.fftshift(np.fft.ifft2(G*H))
   
  
    amp_rec_image = np.abs(Rec_image)
    
    if ind < 1:
        maxproj = amp_rec_image
        
##finding Minimum projection###
    
    for p in indd:
       for j in indd:
            if maxproj[p][j] > amp_rec_image[p][j]:
                maxproj[p][j]=amp_rec_image[p][j]
                #minproj[:,:,ind]=maxproj

                
    #ampmax=maxproj.max()
    #ampmin=maxproj.min()
    #print ampmax
    #print ampmin
    
    
   #minproj = [[maxproj[p][j]+amp_rec_image[p][j] for j in range(len(maxproj[0]))] for p in range(len(maxproj))]
    
    
    #contrast=np.arctan(np.imag(Rec_image)/np.real(Rec_image))`` ##this is phase contrast

    #height=lambda0/4/pi*contrast

    

   
    
    #
    
    #threeD[:,:,ind]=np.abs(Rec_image)
    #threeD=threeD.astype(np.float32)


    #imaginary[:, :, ind] = np.imag(Rec_image)
    #imaginary= imaginary.astype(np.float32)

    #
    #original[:, :, ind] = np.angle(Rec_image)
    #original= original.astype(np.float32)




     #to close windows plt.close('all')
      #plt.savefig("/home/owner/hologram/samples/1/convolution/" + str(d) + ".tif")

# #
# tifffile.imsave("/home/owner/hologram/samples/1/convolution/dec15micro2/50220angle.tif",original.transpose())
# tifffile.imsave("/home/owner/hologram/samples/1/convolution/dec15micro2/50220imaginary.tif",imaginary.transpose())
# tifffile.imsave("/home/owner/hologram/samples/1/convolution/dec15micro2/50220amp.tif",threeD.transpose())
# # #
#hp.show(original)
#hp.show(imaginary)
#threeD=threeD.astype(np.float32)



######################FINDING INFOCUSE XY PLANE#################################
window_size = 25
#maxproj = np.array(maxproj*255,dtype=np.uint8)
maxproj*=255.0/maxproj.max()

#binary_global = maxproj > threshold_otsu(maxproj)
#thresh_niblack = threshold_niblack(maxproj, window_size=window_size, k=0.8)
thresh_sauvola = threshold_sauvola(maxproj, window_size=window_size, k=0.5, r=128)

#binary_niblack = maxproj > thresh_niblack
binary_sauvola = maxproj > thresh_sauvola

opening=binary_opening(invert(binary_sauvola),np.ones((3,3),np.uint8))
opening=opening.astype(np.uint8)
maxproj=maxproj.astype(np.uint8)
color=cv2.cvtColor(maxproj,cv2.COLOR_GRAY2RGB)


cnts=cv2.findContours(opening.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=cnts[0] if imutils.is_cv2() else cnts[1]

#cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('Image',600,600)   
#cv2.namedWindow('particle cropped', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('particle cropped',600,600)


#loop over the contours
for i,c in enumerate (cnts):
    M = cv2.moments(c)
    (x,y,w,h) = cv2.boundingRect(c)
    particle = maxproj[y:y+h,x:x+w]
    
    for d in dp:
        ind=(d-mind)/steps
        Metric=np.linalg.norm(particle)
        metricarray=np.append(metricarray,[Metric])
        
    plt.figure()
    start=i*len(dp)
    end=(i+1)*len(dp)
    plt.plot(dp,metricarray[start:end])
    plt.title('Intensity versus distance for particle # %s'%(i+1))
    #plt.ylabel('entropy')
    #plt.xlabel('distance')
    plt.show() 
    
##initiate a 3d images for cropped image
    

  
   

#plt.close()   
    #cX = int(M["m10"] / M["m00"])
    #cY = int(M["m01"] / M["m00"])
    #cv2.drawContours(color, [c], -1, (0, 0, 255), 1)#draws contours found using cnts on picture
    #cv2.ellipse(color,cv2.fitEllipse(c),(0,255,0),2)
    #cv2.circle(color, (cX, cY), 3, (128, 0, 128), -1)
    #cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),1)
    #cv2.putText(color, "center", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
 
    #cv2.imshow("Image", color)
    #cv2.waitKey(0)

    #cv2.imshow('particle cropped', particle)

 
#cv2.destroyAllWindows()


###FINDING Z PLANE FOR EACH XY OBJECTS#
  

    
    #Metric=np.linalg.norm(np.real(G*H),ord=1)#plane wave is assumed to be a constant amplitude#metric using fourier modulus
    #Metric=np.linalg.norm(Rec_image)#metric using spatial frequencies modulus
    # Metric=np.log((1+np.abs(G*H)))#focuse metric using the log, self entropy, Patrik Langehanenberg*2007
    #metricarray=np.append(metricarray,[Metric])
    #metricarray=np.append(metricarray,[Metric])
    


	# draw the contour and center of the shape on the image

	# show the image



#hp.show(threeD)

#plt.figure()
#plt.plot(dp,metricarray)

#plt.ylabel('entropy')
#plt.xlabel('distance')
#plt.show()


#tifffile.imsave("/home/alinsi/Desktop/minimumprojection",threeD.transpose())



#plt.figure()
#plt.imshow(maxproj,cmap=plt.cm.gray)
#plt.title('Original')
#plt.axis('off')
#
#
#
#plt.figure()
#plt.imshow(opening,cmap=plt.cm.gray)
#plt.title('Sauvola Threshold with opening and inverted')
#plt.axis('off')
#
#
#plt.show()
