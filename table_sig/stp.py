#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:02:44 2018

Maria J. Molina
PhD Student
Central Michigan University

"""


###############################################################################
###############################################################################
###############################################################################


from __future__ import division

import numpy as np
import pickle
import mm_pkg as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime as dt
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter as gfilt
from itertools import product
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile


###############################################################################
###############################################################################
###############################################################################


datas = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)

data = datas.stp.resample(time="D").mean()

data.to_netcdf('stpmean_daily')


###############################################################################
###############################################################################
###############################################################################


from __future__ import division

import numpy as np
import pickle
import mm_pkg as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime as dt
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter as gfilt
from itertools import product
from mpl_toolkits.basemap import Basemap
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile


###############################################################################
###############################################################################
###############################################################################


datas = xr.open_dataset('stpmean_daily', decode_cf=True)

datas.stp.values 


with open('yr_neut_values_days', 'rb') as f:
    num_neut_yrs = pickle.load(f)
    
with open('yr_nino_values_days', 'rb') as f:
    num_nina_yrs = pickle.load(f)
    
with open('yr_nina_values_days', 'rb') as f:
    num_nino_yrs = pickle.load(f)
    

###############################################################################
###############################################################################
###############################################################################


data_alls = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_ensoannualpaper', decode_cf=True)

yr_choices = pd.date_range('1979-01-01','2016-12-01',freq='AS').year.values

sumalls_tors = data_alls.grid.groupby('time.dayofyear').sum('time')

data_allS = np.divide(sumalls_tors, len(yr_choices))

stpmask_alls = data_allS.sum('dayofyear')


###############################################################################
###############################################################################
###############################################################################


latlon = xr.open_dataset('/storage/timme1mj/NARR/jclimate/latlon', decode_cf=False)

llcrnrlon = -120
llcrnrlat = 15
urcrnrlon = -60
urcrnrlat = 50

m = Basemap(projection='lcc', lat_0 = 39, lon_0 = -96, lat_1 = 40,
            llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat,
            urcrnrlat = urcrnrlat, urcrnrlon = urcrnrlon,
            resolution='l')

x1, y1 = m(latlon.lons.values, latlon.lats.values)


def get_us_border_polygon():

    sf = shapefile.Reader("tl_2017_us_state")
    shapes = sf.shapes()

    fields = sf.fields
    records = sf.records()
    state_polygons = {}
    
    for i, record in enumerate(records):
        
        state = record[5]
        points = shapes[i].points
        poly = Polygon(points)
        state_polygons[state] = poly

    return state_polygons

state_polygons = get_us_border_polygon()   

def in_us(lat, lon):
    p = Point(lon, lat)
    for state, poly in state_polygons.iteritems():
        if poly.contains(p):
            return state
    return None


###############################################################################
###############################################################################
###############################################################################



for i, j in product(xrange(len(datas.stp[0,:,0])),xrange(len(datas.stp[0,0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    if not m.is_land(x,y):
        
        datas.stp[:,i,j] = None
    
    xpt, ypt = m(x,y,inverse=True)
    
    if not in_us(ypt, xpt):

        datas.stp[:,i,j] = None
        
    if np.all(np.isfinite(datas.stp[:,i,j])) and ypt > 50.0:
        
        datas.stp[:,i,j] = None
            
    if stpmask_alls[i,j] <= 2.5:
        
        datas.stp[:,i,j] = None        
  
    
data_line = datas.stp.mean(dim=['lat','lon'], skipna=True)

data_line.to_netcdf('stp_forsigcomp')


###############################################################################
###############################################################################
###############################################################################



from __future__ import division

import numpy as np
import pickle
import mm_pkg as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime as dt
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter as gfilt
from itertools import product


#p = 1


data_line = xr.open_dataarray('stp_forsigcomp', decode_cf=True)


with open('yr_neut_values_days', 'rb') as f:
    num_neut_yrs = pickle.load(f)
    
with open('yr_nino_values_days', 'rb') as f:
    num_nina_yrs = pickle.load(f)
    
with open('yr_nina_values_days', 'rb') as f:
    num_nino_yrs = pickle.load(f)
    

yr_choices = pd.date_range('1979-01-01','2016-12-01',freq='AS').year.values


day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)


day_list_1 = np.cumsum(day_list.astype('int64'))-day_list.astype('int64')
day_list_2 = np.cumsum(day_list.astype('int64'))   


total_days_three = np.array([0,np.nansum(day_list.astype('int64')),(np.nansum(day_list.astype('int64'))*2)])

    
summed_neut = np.zeros([1000, np.sum(day_list.astype('int64'))])
summed_nina = np.zeros([1000, np.sum(day_list.astype('int64'))])
summed_nino = np.zeros([1000, np.sum(day_list.astype('int64'))])


for _ in xrange(1000):
    
    sum_neut = np.zeros([np.sum(day_list.astype('int64'))])
    sum_nina = np.zeros([np.sum(day_list.astype('int64'))])
    sum_nino = np.zeros([np.sum(day_list.astype('int64'))])
    
    
    for x in xrange(12):
    
        rand_neut = np.array([yr_choices[np.random.choice(len(yr_choices))] for i in xrange(num_neut_yrs[x])])
        rand_nina = np.array([yr_choices[np.random.choice(len(yr_choices))] for i in xrange(num_nina_yrs[x])])
        rand_nino = np.array([yr_choices[np.random.choice(len(yr_choices))] for i in xrange(num_nino_yrs[x])])


        for e, i in enumerate(rand_neut):
        
            if e == 0:
            
                file_1 = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                
            elif e == 1:
            
                file_2 = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                file_neut = np.stack([file_1[:day_list[x].astype('int64')],file_2[:day_list[x].astype('int64')]])

            else:
            
                file_more = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                file_more = np.expand_dims(file_more, axis=0)
                file_neut = np.vstack([file_neut, file_more[:,:day_list[x].astype('int64')]])

        
        for e, i in enumerate(rand_nina):
        
            if e == 0:
            
                file_1 = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                
            elif e == 1:
            
                file_2 = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                file_nina = np.stack([file_1[:day_list[x].astype('int64')],file_2[:day_list[x].astype('int64')]])

            else:
            
                file_more = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                file_more = np.expand_dims(file_more, axis=0)
                file_nina = np.vstack([file_nina, file_more[:,:day_list[x].astype('int64')]])
            
            
        for e, i in enumerate(rand_nino):
        
            if e == 0:
                
                file_1 = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                
            elif e == 1:
            
                file_2 = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                file_nino = np.stack([file_1[:day_list[x].astype('int64')],file_2[:day_list[x].astype('int64')]])

            else:
            
                file_more = data_line[(data_line['time.month']==x+1) & (data_line['time.year']==i)]
                file_more = np.expand_dims(file_more, axis=0)
                file_nino = np.vstack([file_nino, file_more[:,:day_list[x].astype('int64')]])


        sum_neut[day_list_1[x]:day_list_2[x]] = np.divide(np.nansum(file_neut,axis=0),len(rand_neut))
        sum_nina[day_list_1[x]:day_list_2[x]] = np.divide(np.nansum(file_nina,axis=0),len(rand_nina))
        sum_nino[day_list_1[x]:day_list_2[x]] = np.divide(np.nansum(file_nino,axis=0),len(rand_nino))


    Gauss_SmoothTN = np.concatenate([sum_neut,sum_neut,sum_neut])
    Gauss_SmoothTLN = np.concatenate([sum_nina,sum_nina,sum_nina])
    Gauss_SmoothTELN = np.concatenate([sum_nino,sum_nino,sum_nino])

    Gauss_SmoothTN = gfilt(Gauss_SmoothTN*1.0,sigma=15.0)
    Gauss_SmoothTLN = gfilt(Gauss_SmoothTLN*1.0,sigma=15.0)
    Gauss_SmoothTELN = gfilt(Gauss_SmoothTELN*1.0,sigma=15.0)

    Gauss_SmoothTN1 = Gauss_SmoothTN[total_days_three[1]:total_days_three[2]]
    Gauss_SmoothTLN1 = Gauss_SmoothTLN[total_days_three[1]:total_days_three[2]]
    Gauss_SmoothTELN1 = Gauss_SmoothTELN[total_days_three[1]:total_days_three[2]]        
        
    summed_neut[_,:] = Gauss_SmoothTN1
    summed_nina[_,:] = Gauss_SmoothTLN1
    summed_nino[_,:] = Gauss_SmoothTELN1
    
    print str(_)+' completed...'
    
    

np.save('stp_sig_neut_'+str(p), summed_neut)
np.save('stp_sig_nina_'+str(p), summed_nina)
np.save('stp_sig_nino_'+str(p), summed_nino)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



import numpy as np


neut_linesig_1 = np.load('stp_sig_neut_1.npy')
neut_linesig_2 = np.load('stp_sig_neut_2.npy')
neut_linesig_3 = np.load('stp_sig_neut_3.npy')
neut_linesig_4 = np.load('stp_sig_neut_4.npy')
neut_linesig_5 = np.load('stp_sig_neut_5.npy')
neut_linesig_6 = np.load('stp_sig_neut_6.npy')
neut_linesig_7 = np.load('stp_sig_neut_7.npy')
neut_linesig_8 = np.load('stp_sig_neut_8.npy')
neut_linesig_9 = np.load('stp_sig_neut_9.npy')
neut_linesig_10 = np.load('stp_sig_neut_10.npy')

neut_linesig = np.vstack([neut_linesig_1,neut_linesig_2,neut_linesig_3,
                          neut_linesig_4,neut_linesig_5,neut_linesig_6,
                          neut_linesig_7,neut_linesig_8,neut_linesig_9,
                          neut_linesig_10])


nino_linesig_1 = np.load('stp_sig_nino_1.npy')
nino_linesig_2 = np.load('stp_sig_nino_2.npy')
nino_linesig_3 = np.load('stp_sig_nino_3.npy')
nino_linesig_4 = np.load('stp_sig_nino_4.npy')
nino_linesig_5 = np.load('stp_sig_nino_5.npy')
nino_linesig_6 = np.load('stp_sig_nino_6.npy')
nino_linesig_7 = np.load('stp_sig_nino_7.npy')
nino_linesig_8 = np.load('stp_sig_nino_8.npy')
nino_linesig_9 = np.load('stp_sig_nino_9.npy')
nino_linesig_10 = np.load('stp_sig_nino_10.npy')

nino_linesig = np.vstack([nino_linesig_1,nino_linesig_2,nino_linesig_3,
                          nino_linesig_4,nino_linesig_5,nino_linesig_6,
                          nino_linesig_7,nino_linesig_8,nino_linesig_9,
                          nino_linesig_10])
    
    
nina_linesig_1 = np.load('stp_sig_nina_1.npy')
nina_linesig_2 = np.load('stp_sig_nina_2.npy')
nina_linesig_3 = np.load('stp_sig_nina_3.npy')
nina_linesig_4 = np.load('stp_sig_nina_4.npy')
nina_linesig_5 = np.load('stp_sig_nina_5.npy')
nina_linesig_6 = np.load('stp_sig_nina_6.npy')
nina_linesig_7 = np.load('stp_sig_nina_7.npy')
nina_linesig_8 = np.load('stp_sig_nina_8.npy')
nina_linesig_9 = np.load('stp_sig_nina_9.npy')
nina_linesig_10 = np.load('stp_sig_nina_10.npy')

nina_linesig = np.vstack([nina_linesig_1,nina_linesig_2,nina_linesig_3,
                          nina_linesig_4,nina_linesig_5,nina_linesig_6,
                          nina_linesig_7,nina_linesig_8,nina_linesig_9,
                          nina_linesig_10])

    
###############################################################################
###############################################################################
###############################################################################
    
 
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]
    

###############################################################################
###############################################################################
###############################################################################
    

neut_diff = np.zeros([10000,6])
nino_diff = np.zeros([10000,6])
nina_diff = np.zeros([10000,6])


for _ in xrange(10000):
    
    neut_linesig[_,:] = np.divide(neut_linesig[_,:],np.sum(neut_linesig[_,:]))
    nino_linesig[_,:] = np.divide(nino_linesig[_,:],np.sum(nino_linesig[_,:]))
    nina_linesig[_,:] = np.divide(nina_linesig[_,:],np.sum(nina_linesig[_,:]))
    
    temp_neut = np.cumsum(neut_linesig[_,:])
    temp_nino = np.cumsum(nino_linesig[_,:])
    temp_nina = np.cumsum(nina_linesig[_,:])
    
    neut_diff[_,0] = np.where(temp_neut==(find_nearest(temp_neut,0.1)))[0]
    nino_diff[_,0] = np.where(temp_nino==(find_nearest(temp_nino,0.1)))[0]
    nina_diff[_,0] = np.where(temp_nina==(find_nearest(temp_nina,0.1)))[0]
    
    neut_diff[_,1] = np.where(temp_neut==(find_nearest(temp_neut,0.25)))[0]
    nino_diff[_,1] = np.where(temp_nino==(find_nearest(temp_nino,0.25)))[0]
    nina_diff[_,1] = np.where(temp_nina==(find_nearest(temp_nina,0.25)))[0]

    neut_diff[_,2] = np.where(temp_neut==(find_nearest(temp_neut,0.5)))[0]
    nino_diff[_,2] = np.where(temp_nino==(find_nearest(temp_nino,0.5)))[0]
    nina_diff[_,2] = np.where(temp_nina==(find_nearest(temp_nina,0.5)))[0]

    neut_diff[_,3] = np.where(temp_neut==(find_nearest(temp_neut,np.nanmax(neut_linesig[_,:]))))[0]
    nino_diff[_,3] = np.where(temp_nino==(find_nearest(temp_nino,np.nanmax(nino_linesig[_,:]))))[0]
    nina_diff[_,3] = np.where(temp_nina==(find_nearest(temp_nina,np.nanmax(nina_linesig[_,:]))))[0]

    neut_diff[_,4] = np.where(temp_neut==(find_nearest(temp_neut,0.75)))[0]
    nino_diff[_,4] = np.where(temp_nino==(find_nearest(temp_nino,0.75)))[0]
    nina_diff[_,4] = np.where(temp_nina==(find_nearest(temp_nina,0.75)))[0]

    neut_diff[_,5] = np.where(temp_neut==(find_nearest(temp_neut,0.9)))[0]
    nino_diff[_,5] = np.where(temp_nino==(find_nearest(temp_nino,0.9)))[0]
    nina_diff[_,5] = np.where(temp_nina==(find_nearest(temp_nina,0.9)))[0]


###############################################################################
###############################################################################
###############################################################################
    

nina_neut = nina_diff - neut_diff
nino_neut = nino_diff - neut_diff


nina_upper95 = np.nanpercentile(nina_neut, 97.5, axis=0)
nina_lower95 = np.nanpercentile(nina_neut, 2.5, axis=0)
nina_upper90 = np.nanpercentile(nina_neut, 95., axis=0)
nina_lower90 = np.nanpercentile(nina_neut, 5., axis=0)

nino_upper95 = np.nanpercentile(nino_neut, 97.5, axis=0)
nino_lower95 = np.nanpercentile(nino_neut, 2.5, axis=0)
nino_upper90 = np.nanpercentile(nino_neut, 95., axis=0)
nino_lower90 = np.nanpercentile(nino_neut, 5., axis=0)


print 'nino'
print '97.5: '+str(nino_upper95)
print '2.5: '+str(nino_lower95)
print '95: '+str(nino_upper90)
print '5: '+str(nino_lower90)
print '  '
print 'nina'
print '97.5: '+str(nina_upper95)
print '2.5: '+str(nina_lower95)
print '95: '+str(nina_upper90)
print '5: '+str(nina_lower90)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



