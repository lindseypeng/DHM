
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import cv2

drawing = False # true if mouse is pressed

# mouse callback function
def draw(event,x,y,flags,param):
    global refPt,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
  
        refPt = [(x, y)]
#    elif event == cv2.EVENT_MOUSEMOVE:
#        if drawing == True:
#            cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),2)


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        refPt.append((x, y))
        cv2.rectangle(img,refPt[0],(x,y),(0,255,0),2)

import tifffile
img = '/home/alinsi/Desktop/lensless/Untitled_4/Pos0/thresholdparse/0k0.2r10win9.tif'
img = tifffile.imread((img))
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw)
clone=img.copy()
##of you dont have 0xFF then k wouldnt be regonize ord(m) since ord gives you the 8bit code

while(1):#while true
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('r'):
        img = clone.copy()
    elif k == 27:##esc key sto stop
        break

cv2.destroyAllWindows()



## In[20]:
threeDPositions=pd.DataFrame()
numberoffiles=1
#
j=1
for i in range(numberoffiles):
    #n="{:03d}".format(i)
    filename='/home/alinsi/Desktop/lensless/Untitled_4/Pos0/threeDPositions{}.csv'.format(j)
    position =pd.read_csv(filename)
    threeDPositions=threeDPositions.append(position,ignore_index=True)
    j=j+1
#
#
#
#

#
##################you can sub select a region of interest in x and y############
#check first the list of cX and cY to see if things are making sense
#xmin=threeDPositions['cX'].min()
#xmax=threeDPositions['cX'].max()
#
#ymin=threeDPositions['cY'].min()
#ymax=threeDPositions['cX'].max()
#
#print('the minimum of x and y are {},{} and maximum of x and y are {},{}'.format(xmin,ymin,xmax,ymax))
##sort x first, take out range outside x
delx=threeDPositions[threeDPositions['cX'].between(refPt[0][0],refPt[1][0],inclusive=True)]

##sort y then, take a smaller sample set of the first sorted
dely=delx[delx['cY'].between(refPt[0][1],refPt[1][1],inclusive=True)]
## only image these frames!!##
dely=dely.reset_index(drop=False)
#xmin=dely['cX'].min()
#xmax=dely['cX'].max()
#
#ymin=dely['cY'].min()
#ymax=dely['cX'].max()
#
#print('the minimum of x and y are {},{} and maximum of x and y are {},{}'.format(xmin,ymin,xmax,ymax))

startingframes=(np.array(dely.frame)).min()
endingframes=(np.array(dely.frame)).max()
rangenum=range(int(startingframes),int(endingframes))
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
ax.set_xlim3d([0.0, 1920*0.0052])
ax.set_ylim3d([0.0, 1920*0.0052])
ax.set_zlim3d([np.min(dely.mindp), np.max(dely.mindp)])
#
plt.ion()

for f in (rangenum):
    
    f2=f+1
    cXs=np.array(dely.loc[dely.frame==f2,'cX'])
    cYs=np.array(dely.loc[dely.frame==f2,'cY'])
    minimumdp=np.array(dely.loc[dely.frame==f2,'mindp'])
    ax.scatter(cXs*0.0052,cYs*0.0052,minimumdp*0.04*0.0052, c=np.random.rand(3,1), marker='_')
       
    plt.pause(1)
    fig.clear()
    ax = fig.add_subplot(111, projection ='3d')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    plt.ion()
    print('this is frame {}'.format(f))
