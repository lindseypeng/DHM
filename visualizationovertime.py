
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


j=1



for i in range(numberoffiles):
    #n="{:03d}".format(i)
##must be atching type of threshold from img earlier
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



##################CALCULATE SPARSITY FOR IMAGING########################################
##sparsity is subjective always to the total number of particles in the region selected
sparsities=[]
threeDPositions2=pd.DataFrame()

grids=3
##########calculate sparsity for each particle##########SHOULD MAKE IT INTO DEFINITION SO YOU CAN PUT IT IN THE PLOTTING LATER##
###divide the array of x range and y range in and z range to a 4by4by4 grids
xmin=np.min(dely.cX)
xmax=np.max(dely.cX)
ymin=np.min(dely.cY)
ymax=np.max(dely.cY)
zmin=np.min(dely.mindp)
zmax=np.max(dely.mindp)
xunit=((xmax-xmin)/grids)
yunit=((ymax-ymin)/grids)
zunit=((zmax-zmin)/grids)

xstar=xmin


#sparsitygrids=pd.DataFrame()
for ii in range(grids):

    xend=xstar+ii*(xunit)
    ystar=ymin
    
    
    for kk in range(grids):

        yend=ystar+kk*(yunit)
        zstar=zmin
        
        for ll in range(grids):
           
            zend=zstar+ll*(zunit)
            
            xcount=dely[dely['cX'].between(xstar,xend,inclusive=True)]
            ycount=xcount[xcount['cY'].between(ystar,yend,inclusive=True)]
            zcount=(ycount[ycount['mindp'].between(zstar,zend,inclusive=True)])
            zcount['sparsity']=sparsity
            finalcount=len(zcount)
            sparsity=finalcount/len(dely)
            sparsities.append(sparsity)
            threeDPositions2=threeDPositions2.append(zcount)
            zstar=zend
        ystar=yend
    xstar=xend

            ##add the spasity for all these items ##
            ###assign sparsities to particular rane so you need the range of x,y,and z to be sored##
            #sparsitygrid=pd.DataFrame({'sparsity':[sparsity],'xi':[xstar],'xf':[xend],'yi':[ystar],'yf':[yend],'zi':[zstar],'zf':[zend]})      
            ##or assign sparsities to each particles and hence give colors based on that##
            ##the colormap will map the color based on the third variable:sparsities#

rangeofspar=np.unique(np.array(sparsities))
##assign sparsities to regions in matplot lib##
##assign color to sparsities###
import matplotlib.colors as colors
import matplotlib.cm as cm

norm = colors.Normalize(vmin=min(np.array(sparsities)), vmax=max(np.array(sparsities)))
f2rgb = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('RdYlGn'))

def f2hex(f2rgb, f):
    rgb = f2rgb.to_rgba(f)[:3]#change colormap f2rgb to rgba color, where a is opacity
    return '#%02x%02x%02x' % tuple([255*fc for fc in rgb])##scaling values from 0 to 1 back to 0 to 255

##create a list, right column is sparsity, left column is color in hex values
sparcolors=pd.DataFrame()
for spar in rangeofspar:
    color=f2hex(f2rgb,spar)
    print('spar and color is {},{}'.format(spar,color))
    threeDPositions2['color']=np.where(threeDPositions2['sparsity']==spar,color,0)
#    sparcolor=pd.DataFrame({'sparsity':[spar],'color':[color]})
#    sparcolors.append(sparcolor)
#    
#    
#    
    
    
#print f2hex(f2rgb, -13.5)  #   gives #a60126  ie, dark red
#print f2hex(f2rgb, 85.5)   #   gives #016937  ie, dark green with so
################################################
startingframes=(np.array(threeDPositions2.frame)).min()
endingframes=(np.array(threeDPositions2.frame)).max()
rangenum=range(int(startingframes),int(endingframes))
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
ax.set_xlim3d([np.min(threeDPositions2.cX)*0.0052, np.max(threeDPositions2.cX)*0.0052])
ax.set_ylim3d([np.min(threeDPositions2.cY)*0.0052, np.max(threeDPositions2.cY)*0.0052])
ax.set_zlim3d([np.min(threeDPositions2.mindp), np.max(threeDPositions2.mindp)])
#plt.pcolor()
#
plt.ion()


f=0

while f<9: 
    for f in (rangenum):
        
        f2=f+1
        cXs=np.array(threeDPositions2.loc[threeDPositions2.frame==f2,'cX'])
        cYs=np.array(threeDPositions2.loc[threeDPositions2.frame==f2,'cY'])
        minimumdp=np.array(threeDPositions2.loc[threeDPositions2.frame==f2,'mindp'])
        color=np.array(threeDPositions2.loc[threeDPositions2.frame==f2,'color'])
        colormapped=np.vstack((cXs,color)).T
        ax.scatter(cXs*0.0052,cYs*0.0052,minimumdp*0.04*0.0052, c=colormapped, marker='o')
           
        plt.pause(0.02)
        fig.clear()
        
        
        ax = fig.add_subplot(111, projection ='3d')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        plt.title('this is frame {}'.format(f2))
        plt.ion()
        #print('this is frame {}'.format(f))
    if f ==max(rangenum):
        f=min(rangenum)##LOOP FOREVER LOL
