numberoffiles=9
j=0
threeDPositions=pd.DataFrame()
for i in range(numberoffiles):

    filename='/home/alinsi/Desktop/LP_Aug/permisiumAaron_Wheel_3/Pos0/threeDPositions{}.csv'.format(j)

    if os.stat(filename).st_size == 0:###pass files that are black but this skips the time frame, this should be fixed by mapping the invididual metatxt to the time
        pass
    else:
        position =pd.read_csv(filename)
        threeDPositions=threeDPositions.append(position,ignore_index=True)
        j=j+1
        
threeDPositions2=threeDPositions
uniqueframes=np.unique(np.array(threeDPositions2.frame))
repeating=8
##try viewing first frame only 
staticframe=np.repeat(min(uniqueframes),20)
totalframe=np.concatenate((staticframe,uniqueframes),axis=0)
##finish##
repeatframes=np.repeat(uniqueframes,repeating)
totallength=len(repeatframes)
#

maxangle=150
minangle=0
interval=(maxangle-minangle)/totallength
angles=np.arange(minangle,maxangle,interval)
elevations=np.arange(0,90,interval)
elev = 89.9
azim = 270.1
dist = 11.0
azims=[]
elevs=[]
dists=[]

cut = range(int(np.ceil(totallength/6)))
for i in (cut*1):
    elev=elev-1
    elevs.append(elev)
    azim=azim+2.5
    azims.append(azim)
    dist=dist-0.1
    dists.append(dist)
for i in (cut):
    elev=elev-0.5
    elevs.append(elev)
    azim=azim+2.5
    azims.append(azim)
    dist=dist-0.1
    dists.append(dist)
for i in (cut):
    elev=elev-1.0
    elevs.append(elev)
    azim=azim+2
    azims.append(azim)
    dist=dist
    dists.append(dist)

elev=elev-0.8
elevs.append(elev)
azim=azim-8
azims.append(azim)
dist=dist-1.2
dists.append(dist)
for i in (cut):
    elev=elev-1.5
    elevs.append(elev)
    azim=azim-5.5
    azims.append(azim)
    dist=dist+0.15
    dists.append(dist)
for i in (cut):
    elev=elev-1.1
    elevs.append(elev)
    azim=azim-0.5
    azims.append(azim)
    dist=dist+0.05
    dists.append(dist)
for i in (range(20)):
    elev=elev-0.8
    elevs.append(elev)
    azim=azim-0.5
    azims.append(azim)
    dist=dist
    dists.append(dist)
    
#staticazims=np.repeat(azim,6)
#staticaelevs=np.repeat(elev,6)
#staticdists=np.repeat(dist,6)

azims=np.array(azims)
elevs=np.array(elevs)
dists=np.array(dists)

#
#azims=np.concatenate((azims,staticazims),axis=0)
#elevs=np.concatenate((elevs,staticaelevs),axis=0)
#dists=np.concatenate((dists,staticdists),axis=0)


def animate(i):
    f2=repeatframes[i]
    xx=np.array(threeDPositions2.loc[threeDPositions2.frame==f2,'cX'])*0.0052
    yy=np.array(threeDPositions2.loc[threeDPositions2.frame==f2,'cY'])*0.0052
    zz=np.array(threeDPositions2.loc[threeDPositions2.frame==f2,'mindp'])*0.04*0.0052
    graph._offsets3d=(xx,yy,zz)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    ax.view_init(elev=elevs[i], azim=azims[i])
    ax.dist=dists[i]
    title.set_text('Paramecium, Frame={}'.format(f2))

xx=np.array(threeDPositions2.loc[threeDPositions2.frame==10,'cX'])*0.0052
yy=np.array(threeDPositions2.loc[threeDPositions2.frame==10,'cY'])*0.0052
zz=np.array(threeDPositions2.loc[threeDPositions2.frame==10,'mindp'])*0.04*0.0052
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
title=ax.set_title('3d test')
graph=ax.scatter(xx,yy,zz,s=5,color='red')
ax.set_xlim3d([np.min(xx), np.max(xx)])
ax.set_ylim3d([np.min(yy), np.max(yy)])
ax.set_zlim3d([0.02,0.08])
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
ax.view_init(elev=89.9, azim=270.1)
ax.dist=11.0



anim = animation.FuncAnimation(fig, animate,frames=totallength, interval=300, blit=False)
