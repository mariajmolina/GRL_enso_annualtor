#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:56:37 2017

Maria J. Molina
Ph.D. Student
Central Michigan University

"""

###############################################################################
###############################################################################
###############################################################################


from __future__ import division

import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta
from mpl_toolkits.basemap import Basemap


###############################################################################
###############################################################################
###############################################################################


Tors = pd.read_csv(r'1950-2016_actual_tornadoes.csv',error_bad_lines=False,
                   parse_dates=[['mo','dy','yr','time']])


Tors = Tors.assign(UTC_time = Tors['mo_dy_yr_time'] + timedelta(hours=6))       
Tors = Tors.assign(UTC_yr = Tors['UTC_time'].dt.year) 
Tors = Tors.assign(UTC_dy = Tors['UTC_time'].dt.day) 
Tors = Tors.assign(UTC_mo = Tors['UTC_time'].dt.month) 
Tors = Tors.assign(UTC_hr = Tors['UTC_time'].dt.hour) 

Tors = Tors[Tors['UTC_yr']>=1953]
Tors = Tors[Tors['UTC_yr']<=2016]

Tors = Tors[Tors['mag']>=1]

Tors = Tors[Tors['slat']!=0]
Tors = Tors[Tors['slon']!=0]
Tors = Tors[Tors['slat']>=20]
Tors = Tors[Tors['slat']<=50]
Tors = Tors[Tors['slon']>=-130]
Tors = Tors[Tors['slon']<=-65]


###############################################################################
###############################################################################
###############################################################################


Tors = Tors.set_index(['UTC_time']) 
groups = Tors.groupby(pd.TimeGrouper("D")) 

xr_time=[]
for group in groups:
    xr_time.append(pd.to_datetime((group[0].to_pydatetime())))


###############################################################################
###############################################################################
###############################################################################
    

llcrnrlon = -120
llcrnrlat = 15
urcrnrlon = -60
urcrnrlat = 50

m = Basemap(projection='lcc', lat_0 = 39, lon_0 = -96, lat_1 = 40,
            llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat,
            urcrnrlat = urcrnrlat, urcrnrlon = urcrnrlon,
            resolution='l')
       
grid_res = 80  
        
xMax,yMax = m(urcrnrlon,urcrnrlat) 
xMin,yMin = m(llcrnrlon,llcrnrlat) 

x_range = (xMax-xMin) / 1000 
y_range = (yMax-yMin) / 1000 

numXGrids = round(x_range / grid_res + .5,0) 
numYGrids = round(y_range / grid_res + .5,0)
        
xi = np.linspace(xMin,xMax,int(numXGrids))
yi = np.linspace(yMin,yMax,int(numYGrids))
       

###############################################################################
###############################################################################
###############################################################################


aggregate_grid = np.zeros((len(groups),yi.shape[0]-1,xi.shape[0]-1)) 
        
for i, group in enumerate(groups): 
    
    yr, lats, lons = group[1]['UTC_yr'].values, group[1]['slat'].values, group[1]['slon'].values
    x_proj,y_proj = m(lons,lats) 
    grid, _, _ = np.histogram2d(y_proj,x_proj, bins=[yi,xi]) 
    aggregate_grid[i,:,:] = grid 
       

###############################################################################
###############################################################################
###############################################################################
    

data = xr.Dataset({'grid':(['time','y','x'],aggregate_grid)}, \
                   coords={'time':xr_time}) 

data_latlon = xr.Dataset({'lats':(['y'],yi),
                          'lons':(['x'],xi)})

intro_data = np.zeros(aggregate_grid[:7,:,:].shape)
intro_time = pd.date_range('1953-01-01','1953-01-07',freq='D')
intro_dat = xr.Dataset({'grid':(['time','y','x'],intro_data)}, \
                        coords={'time':intro_time}) 

exit_data = np.zeros(aggregate_grid[:2,:,:].shape)
exit_time = pd.date_range('2016-12-30','2016-12-31',freq='D')
exit_dat = xr.Dataset({'grid':(['time','y','x'],exit_data)}, \
                       coords={'time':exit_time}) 

concat_data = xr.concat([intro_dat,data], dim=('time'))
concat_data = xr.concat([concat_data,exit_dat], dim=('time'))

filename = 'tors_5316_ef1'
concat_data.to_netcdf(path='/storage/timme1mj/maria_pysplit/'+filename)
#data_latlon.to_netcdf(path='/storage/timme1mj/maria_pysplit/tor_grid_latlon')


###############################################################################
###############################################################################
###############################################################################
