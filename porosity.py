## -*- coding: utf-8 -*-
#"""
#Created on Thu Apr 19 14:21:30 2018
#
#@author: alinsi
#"""
#
import pandas as pd
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import os 


numberoffiles=181
read='/home/alinsi/Desktop/base_1/Pos0/threeDPositions{}.csv'
fff='frame'
timestamp='/home/alinsi/Desktop/base_1/Pos0/metadata/timestamp.csv'
class porosity:
##reading the threeDPositions files and: aggregate into one master files
    def __init__(self,read,numberoffiles,timestamp,fff):
        self.read=read
        self.numberoffiles=numberoffiles
        self.timestamp=timestamp
        self.fff=fff
    
    
    def aggregate_files(self):
        self.threeDPositions=pd.DataFrame()
        for i in range(numberoffiles):
            self.filename=(self.read).format(i+1)
            if os.path.exists(self.filename)==False:#if files dosnt exist, pass, or if file size is smaller, use another command
                pass
            else:
                position =pd.read_csv(self.filename)
                self.threeDPositions=self.threeDPositions.append(position,ignore_index=True)
        return self.threeDPositions    
    
    def match_time(self):   
        self.threeDPositions=self.aggregate_files(self.read,self.numberoffiles)
        self.times=pd.read_csv(self.timestamp)
        self.times['cumulative_time']=self.times['timelapses'].cumsum()
        self.times2=self.times[[self.fff,'timelapses','cumulative_time']]##change actual frame<->frame depends on if images were taken over several instances
        self.matched=pd.merge(self.times2,self.threeDPositions,on=self.fff)
        self.matched=self.matched.drop_duplicates(subset=self.fff)
        self.matched=matched[['cX','cY','mindp','particle','porosity','radius',self.fff,'cumulative_time','timelapses']]
        self.matched.sort_values(by=[self.fff])##,merging time with original threeDPositions
        return self.matched
    
    def extract_porosities(self):  ##drop any duplicated lines with repeating xy coordinates, porosity should be same
        self.matched=self.match_time()
        self.x=np.array(self.matched[self.fff])##x axis in frame
        self.x2=np.array(self.matched['cumulative_time'])
        self.timelapses=np.array(self.matched['timelapses'])##x axis in real time, time is in ms,convert to minutes
        self.porosity=np.array(self.matched['porosity'])##y axis in porosity
        return self.x,self.x2,self.porosity,self.timelapses
    
    def plot_frame_porosity(self):
        self.x,self.x2,self.porosity,self.times=self.extract_porosities()
        plt.figure()
        plt.scatter(self.x,self.porosity,label='static')
        plt.legend()
        plt.xlabel('frames')
        plt.ylabel('1-porosities (%)')

    def avg_time_porosity2(self):
        self.averporosities=[]
        self.avertimes=[]
        self.cumsum=0
        self.indexbreak=[]
        self.cumtime=0
        
        c=0
        self.x,self.x2,self.porosity,self.times=self.extract_porosities()
        for i in range(len(self.x2)-1):
            change=self.x2[i+1]-self.x2[i]
            c=c+1
            if change<3500000:
                self.cumsum=self.cumsum+self.porosity[i+1]
                self.cumtime=(self.cumtime+self.x2[i+1])
            else:
                self.average=self.cumsum/c
                self.avertime=self.cumtime/c/1000/60.000/60.000##change time from ms to hours
                self.avertimes.append(self.avertime)
                self.averporosities.append(self.average)
                self.breaks=int(i+1)
                self.indexbreak.append(self.breaks)
                c=0
                self.cumsum=0
                self.cumtime=0
                self.cumsum=self.cumsum+self.porosity[i+1]
                self.cumtime=self.cumtime+self.x2[i+1]
        datas=[]
        y=0
        for i in self.indexbreak:
            data=np.ndarray.tolist(self.porosity[y:i])
            y=i
            datas.append(data)
        return datas,self.avertimes,self.averporosities,self.indexbreak
            
