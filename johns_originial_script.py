#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 21:57:05 2017

@author: timme1mj
"""


from __future__ import division
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter as gfilt
import matplotlib.pyplot as plt
import datetime as dt



TD = pd.read_csv('1950-2016_actual_tornadoes.csv',error_bad_lines=False)

TD = TD[TD['mag']>=1]
TD = TD[TD['yr']>=1953]

TD = TD[TD['slat']!=0]
TD = TD[TD['slon']!=0]
TD = TD[TD['slat']>=20]
TD = TD[TD['slat']<=50]
TD = TD[TD['slon']>=-130]
TD = TD[TD['slon']<=-65]
            

N = [1979,1980,1981,1982,1984,1986,1988,1990,1991,1994,1995,1996,1997,2001,2002,2003,2004,2006,2007,2009,2012,2013,2014]
TornData = TD[TD.yr.isin(N)]    
year = np.array(TornData['yr'])
month = np.array(TornData['mo'])
day = np.array(TornData['dy'])

Julianday3 = []
for n in range(0,len(year)-1):
    Date = dt.datetime(year[n],month[n],day[n])
    Julianday3.append(Date.strftime('%j'))
Julianday3 = [int(i) for i in Julianday3]
Histo = np.histogram(Julianday3, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_SmoothTN = Gauss_Smooth0[366:732]/24


N = [1979,1980,1981,1982,1984,1986,1988,1990,1991,1994,1995,1996,1997,2001,2002,2003,2004,2006,2007,2009,2012,2013,2014]
TornData = TD[TD.Year.isin(N)]    
year = np.array(TornData['Year'])
month = np.array(TornData['Month'])
day = np.array(TornData['Day'])

Julianday3 = []
for n in range(0,len(year)-1):
    Date = dt.datetime(year[n],month[n],day[n])
    Julianday3.append(Date.strftime('%j'))
Julianday3 = [int(i) for i in Julianday3]
Histo = np.histogram(Julianday3, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_SmoothTN = Gauss_Smooth0[366:732]/24


N = [1979,1980,1981,1982,1984,1986,1988,1990,1991,1994,1995,1996,1997,2001,2002,2003,2004,2006,2007,2009,2012,2013,2014]
TornData = TD[TD.Year.isin(N)]    
year = np.array(TornData['Year'])
month = np.array(TornData['Month'])
day = np.array(TornData['Day'])

Julianday3 = []
for n in range(0,len(year)-1):
    Date = dt.datetime(year[n],month[n],day[n])
    Julianday3.append(Date.strftime('%j'))
Julianday3 = [int(i) for i in Julianday3]
Histo = np.histogram(Julianday3, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_SmoothTN = Gauss_Smooth0[366:732]/24





fig = plt.figure(figsize=(8,8))

#ax1 = fig.add_axes([0.0, 0.75, 0.34, 0.2325]) 


#for i in range(0,5):
#    plt.plot(range(0,366),Gauss_SmoothT[i,:]/np.sum(Gauss_SmoothT[i,:]),'r-',linewidth=0.8,alpha=0.35)
    
#for i in range(0,7):    
#   plt.plot(range(0,366),Gauss_SmoothT2[i,:]/np.sum(Gauss_SmoothT2[i,:]),'b-',linewidth=0.8,alpha=0.35)


#p1, = plt.plot(range(0,366),Gauss_SmoothTELN/np.sum(Gauss_SmoothTELN),'r-',linewidth=2.0)
#p2, = plt.plot(range(0,366),Gauss_SmoothTLN/np.sum(Gauss_SmoothTLN),'b-',linewidth=2.0)
p3, = plt.plot(range(0,366),Gauss_SmoothTN/np.sum(Gauss_SmoothTN),'k-',linewidth=2.0)    


plt.ylabel('Fraction of Tornado Reports ((E)F1+)')
plt.xlabel('Day of Year')

plt.xlim(0,366)
plt.ylim(0,0.015)
tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
plt.xticks(tick_locs, tick_lbls)
plt.title('b) Annual Cycle of US Tornado Reports')


#legend = plt.legend([p1,p2,p3],
#                    [u"El Ni\xf1o",
##                    u"La Ni\xf1a",
 #                   "Neutral"],
 #                   loc="upper right",
 #                   fancybox=True, fontsize=12)

plt.show()




