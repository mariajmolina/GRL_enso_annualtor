
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 20:43:24 2017

Maria J. Molina
Ph.D. Student 
Central Michigan University

"""

###############################################################################
###############################################################################
###############################################################################


from __future__ import division

import numpy as np
import mm_pkg as pk
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime as dt
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter as gfilt
from itertools import product


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


US_tors_ef1 = xr.open_dataset('tors_5316_ef1_obs', decode_cf=True)
US_tors_ef2 = xr.open_dataset('tors_5316_ef2_obs', decode_cf=True)
SE_tors_ef1 = xr.open_dataset('tors_5316_ef1_SE_obs', decode_cf=True)
SE_tors_ef2 = xr.open_dataset('tors_5316_ef2_SE_obs', decode_cf=True)

US_tors_ef1 = US_tors_ef1.grid.where(US_tors_ef1.grid==0, 1)
US_tors_ef2 = US_tors_ef2.grid.where(US_tors_ef2.grid==0, 1)
SE_tors_ef1 = SE_tors_ef1.grid.where(SE_tors_ef1.grid==0, 1)
SE_tors_ef2 = SE_tors_ef2.grid.where(SE_tors_ef2.grid==0, 1)

US_nuyr_ef1 = len(US_tors_ef1.groupby('time.year').sum('time').year.values)
US_nuyr_ef2 = len(US_tors_ef2.groupby('time.year').sum('time').year.values)
SE_nuyr_ef1 = len(SE_tors_ef1.groupby('time.year').sum('time').year.values)
SE_nuyr_ef2 = len(SE_tors_ef2.groupby('time.year').sum('time').year.values)

US_sumtors_ef1 = US_tors_ef1.sum(['x','y'])
US_sumtors_ef2 = US_tors_ef2.sum(['x','y'])
SE_sumtors_ef1 = SE_tors_ef1.sum(['x','y'])
SE_sumtors_ef2 = SE_tors_ef2.sum(['x','y'])

years_count = US_tors_ef1.groupby('time.year').sum('time').year.values


sliced_ef1_US = {}
sliced_ef2_US = {}
sliced_ef1_SE = {}
sliced_ef2_SE = {}

for u, i in enumerate(years_count):

    temp_US_ef1 = US_sumtors_ef1.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))
    temp_US_ef2 = US_sumtors_ef2.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))
    temp_SE_ef1 = SE_sumtors_ef1.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))
    temp_SE_ef2 = SE_sumtors_ef2.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))

    three_sum_ef1_US = xr.concat([temp_US_ef1,temp_US_ef1,temp_US_ef1], dim='time')
    three_sum_ef2_US = xr.concat([temp_US_ef2,temp_US_ef2,temp_US_ef2], dim='time')
    three_sum_ef1_SE = xr.concat([temp_SE_ef1,temp_SE_ef1,temp_SE_ef1], dim='time')
    three_sum_ef2_SE = xr.concat([temp_SE_ef2,temp_SE_ef2,temp_SE_ef2], dim='time') 

    Gauss_Smooth_ef1_US = gfilt(three_sum_ef1_US*1.0, sigma=15.0)
    Gauss_Smooth_ef2_US = gfilt(three_sum_ef2_US*1.0, sigma=15.0)
    Gauss_Smooth_ef1_SE = gfilt(three_sum_ef1_SE*1.0, sigma=15.0)
    Gauss_Smooth_ef2_SE = gfilt(three_sum_ef2_SE*1.0, sigma=15.0)

    sliced_ef1_US[u] = Gauss_Smooth_ef1_US[len(temp_US_ef1):len(temp_US_ef1)*2]
    sliced_ef2_US[u] = Gauss_Smooth_ef2_US[len(temp_US_ef2):len(temp_US_ef2)*2]
    sliced_ef1_SE[u] = Gauss_Smooth_ef1_SE[len(temp_SE_ef1):len(temp_SE_ef1)*2]
    sliced_ef2_SE[u] = Gauss_Smooth_ef2_SE[len(temp_SE_ef2):len(temp_SE_ef2)*2]


for i in xrange(len(sliced_ef1_US)):   
    
    if i == 0:
        
        files_1 = sliced_ef1_US[i]
        files_2 = sliced_ef2_US[i]
        files_3 = sliced_ef1_SE[i]
        files_4 = sliced_ef2_SE[i]
    
    if i == 1:
    
        files1 = np.dstack([files_1, sliced_ef1_US[i]])    
        files2 = np.dstack([files_2, sliced_ef2_US[i]]) 
        files3 = np.dstack([files_3, sliced_ef1_SE[i]]) 
        files4 = np.dstack([files_4, sliced_ef2_SE[i]])
        
    if i > 1:
        
        files1 = np.dstack([files1, sliced_ef1_US[i][:365]])    
        files2 = np.dstack([files2, sliced_ef2_US[i][:365]]) 
        files3 = np.dstack([files3, sliced_ef1_SE[i][:365]]) 
        files4 = np.dstack([files4, sliced_ef2_SE[i][:365]])        
        
        
file1 = np.squeeze(files1)
file2 = np.squeeze(files2)
file3 = np.squeeze(files3)
file4 = np.squeeze(files4)


###############################################################################
###############################################################################
###############################################################################


US_sumtors_ef1_forG = US_sumtors_ef1.groupby('time.dayofyear').mean()
US_sumtors_ef2_forG = US_sumtors_ef2.groupby('time.dayofyear').mean()
SE_sumtors_ef1_forG = SE_sumtors_ef1.groupby('time.dayofyear').mean()
SE_sumtors_ef2_forG = SE_sumtors_ef2.groupby('time.dayofyear').mean()

three_sum_ef1_US = xr.concat([US_sumtors_ef1_forG,US_sumtors_ef1_forG,US_sumtors_ef1_forG], dim='dayofyear')
three_sum_ef2_US = xr.concat([US_sumtors_ef2_forG,US_sumtors_ef2_forG,US_sumtors_ef2_forG], dim='dayofyear')
three_sum_ef1_SE = xr.concat([SE_sumtors_ef1_forG,SE_sumtors_ef1_forG,SE_sumtors_ef1_forG], dim='dayofyear')
three_sum_ef2_SE = xr.concat([SE_sumtors_ef2_forG,SE_sumtors_ef2_forG,SE_sumtors_ef2_forG], dim='dayofyear') 

Gauss_Smooth_ef1_US = gfilt(three_sum_ef1_US*1.0, sigma=15.0)
Gauss_Smooth_ef2_US = gfilt(three_sum_ef2_US*1.0, sigma=15.0)
Gauss_Smooth_ef1_SE = gfilt(three_sum_ef1_SE*1.0, sigma=15.0)
Gauss_Smooth_ef2_SE = gfilt(three_sum_ef2_SE*1.0, sigma=15.0)

Gauss_SmoothAN1 = Gauss_Smooth_ef1_US[len(temp_US_ef1):len(temp_US_ef1)*2]
Gauss_SmoothAN2 = Gauss_Smooth_ef2_US[len(temp_US_ef2):len(temp_US_ef2)*2]
Gauss_SmoothAN3 = Gauss_Smooth_ef1_SE[len(temp_SE_ef1):len(temp_SE_ef1)*2]
Gauss_SmoothAN4 = Gauss_Smooth_ef2_SE[len(temp_SE_ef2):len(temp_SE_ef2)*2]


###############################################################################
###############################################################################
###############################################################################


hi_thresh1 = np.zeros(len(file1[:,0]))
hi_thresh2 = np.zeros(len(file1[:,0]))
hi_thresh3 = np.zeros(len(file1[:,0]))
hi_thresh4 = np.zeros(len(file1[:,0]))

lo_thresh1 = np.zeros(len(file1[:,0]))
lo_thresh2 = np.zeros(len(file1[:,0]))
lo_thresh3 = np.zeros(len(file1[:,0]))
lo_thresh4 = np.zeros(len(file1[:,0]))

for i in xrange(len(file1[:,0])):
    
    hi_thresh1[i] = np.divide(np.nanpercentile(file1[i,:], 75.), np.sum(Gauss_SmoothAN1))
    hi_thresh2[i] = np.divide(np.nanpercentile(file2[i,:], 75.), np.sum(Gauss_SmoothAN2))
    hi_thresh3[i] = np.divide(np.nanpercentile(file3[i,:], 75.), np.sum(Gauss_SmoothAN3))
    hi_thresh4[i] = np.divide(np.nanpercentile(file4[i,:], 75.), np.sum(Gauss_SmoothAN4))

    lo_thresh1[i] = np.divide(np.nanpercentile(file1[i,:], 25.), np.sum(Gauss_SmoothAN1))
    lo_thresh2[i] = np.divide(np.nanpercentile(file2[i,:], 25.), np.sum(Gauss_SmoothAN2))
    lo_thresh3[i] = np.divide(np.nanpercentile(file3[i,:], 25.), np.sum(Gauss_SmoothAN3))
    lo_thresh4[i] = np.divide(np.nanpercentile(file4[i,:], 25.), np.sum(Gauss_SmoothAN4))
    

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)

neut_comp = {}
nino_comp = {}
nina_comp = {}


###############################################################################
###############################################################################
###############################################################################


for k, (i, j) in enumerate(zip(month_list, day_list)):

    print 'Running month: '+i+' and day: '+j+'...'
   
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor.severe_load() 
    
    
    for s, b in enumerate(alltor.neutral_base_yrs):
        
        if s == 0:
            
            da = US_tors_ef1.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))

        elif s == 1:
            
            da2 = US_tors_ef1.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da, da2], dim=('time'))

        else:

            da2 = US_tors_ef1.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da_group, da2], dim=('time'))            
            

    for d, n in enumerate(alltor.el_nino_yrs):
        
        if d == 0:
            
            na = US_tors_ef1.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))      

        elif d == 1:

            na2 = US_tors_ef1.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na, na2], dim=('time'))
            
        else:
            
            na2 = US_tors_ef1.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na_group, na2], dim=('time')) 
            
        
    for f, m in enumerate(alltor.la_nina_yrs):
    
        if f == 0:
        
            la = US_tors_ef1.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))       
    
        elif f == 1:
            
            la2 = US_tors_ef1.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))            
            la_group = xr.concat([la, la2], dim=('time'))
            
        else:
            
            la2 = US_tors_ef1.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))
            la_group = xr.concat([la_group, la2], dim=('time'))             

            
    neut_comp[k] = da_group.sum(['x','y'])
    nino_comp[k] = na_group.sum(['x','y'])
    nina_comp[k] = la_group.sum(['x','y'])

   
    
###############################################################################
###############################################################################
###############################################################################
    

for k in xrange(12):
    
    num_yrs_neut = len(neut_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)

    elif k == 1:
        
        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
      
    else:

        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear'))  
        
        

sum_group_1 = sum_group_neut[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_neut[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
neutral_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_neut_concat)},
                              coords={'dayofyear': sum_coord_neut_concat})  

 

###############################################################################
###############################################################################
###############################################################################
  
    
for k in xrange(12):
    
    num_yrs_nino = len(nino_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nino = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
    
    elif k == 1:
        
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
      
        sum_group_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
     
    else:
          
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
    
        sum_group_nino = xr.concat([sum_group_nino, sum_tors_nino_2], dim=('dayofyear'))   



sum_group_1 = sum_group_nino[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nino[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
elninos_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nino_concat)},
                              coords={'dayofyear': sum_coord_nino_concat})  

     

###############################################################################        
###############################################################################
###############################################################################

    
for k in xrange(12):

    num_yrs_nina = len(nina_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nina = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:
    
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2], dim=('dayofyear'))  
        
        
 
sum_group_1 = sum_group_nina[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nina[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
laninas_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nina_concat)},
                              coords={'dayofyear': sum_coord_nina_concat})  


###############################################################################
###############################################################################
###############################################################################


Gauss_SmoothN = xr.concat([neutral_concats,neutral_concats,neutral_concats], dim='dayofyear')
Gauss_SmoothEN = xr.concat([elninos_concats,elninos_concats,elninos_concats], dim='dayofyear')
Gauss_SmoothLN = xr.concat([laninas_concats,laninas_concats,laninas_concats], dim='dayofyear')

Gauss_SmoothN = Gauss_SmoothN.grid.values
Gauss_SmoothEN = Gauss_SmoothEN.grid.values
Gauss_SmoothLN = Gauss_SmoothLN.grid.values
    
Gauss_SmoothN = gfilt(Gauss_SmoothN*1.0,sigma=15.0)
Gauss_SmoothEN = gfilt(Gauss_SmoothEN*1.0,sigma=15.0)
Gauss_SmoothLN = gfilt(Gauss_SmoothLN*1.0,sigma=15.0)

sliced_gaussN = Gauss_SmoothN[len(neutral_concats.grid):len(neutral_concats.grid)*2]
sliced_gaussEN = Gauss_SmoothEN[len(elninos_concats.grid):len(elninos_concats.grid)*2]
sliced_gaussLN = Gauss_SmoothLN[len(laninas_concats.grid):len(laninas_concats.grid)*2]


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)

neut_comp = {}
nino_comp = {}
nina_comp = {}


###############################################################################
###############################################################################
###############################################################################


for k, (i, j) in enumerate(zip(month_list, day_list)):

    print 'Running month: '+i+' and day: '+j+'...'
   
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor.severe_load() 
    
    
    for s, b in enumerate(alltor.neutral_base_yrs):
        
        if s == 0:
            
            da = US_tors_ef2.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))

        elif s == 1:
            
            da2 = US_tors_ef2.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da, da2], dim=('time'))

        else:

            da2 = US_tors_ef2.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da_group, da2], dim=('time'))            
            

    for d, n in enumerate(alltor.el_nino_yrs):
        
        if d == 0:
            
            na = US_tors_ef2.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))      

        elif d == 1:

            na2 = US_tors_ef2.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na, na2], dim=('time'))
            
        else:
            
            na2 = US_tors_ef2.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na_group, na2], dim=('time')) 
            
        
    for f, m in enumerate(alltor.la_nina_yrs):
    
        if f == 0:
        
            la = US_tors_ef2.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))       
    
        elif f == 1:
            
            la2 = US_tors_ef2.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))            
            la_group = xr.concat([la, la2], dim=('time'))
            
        else:
            
            la2 = US_tors_ef2.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))
            la_group = xr.concat([la_group, la2], dim=('time'))             

            
    neut_comp[k] = da_group.sum(['x','y'])
    nino_comp[k] = na_group.sum(['x','y'])
    nina_comp[k] = la_group.sum(['x','y'])  
    
    
###############################################################################
###############################################################################
###############################################################################
    

for k in xrange(12):
    
    num_yrs_neut = len(neut_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)

    elif k == 1:
        
        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
      
    else:

        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear'))  
        
        

sum_group_1 = sum_group_neut[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_neut[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
neutral_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_neut_concat)},
                              coords={'dayofyear': sum_coord_neut_concat})  

 

###############################################################################
###############################################################################
###############################################################################
  
    
for k in xrange(12):
    
    num_yrs_nino = len(nino_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nino = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
    
    elif k == 1:
        
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
      
        sum_group_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
     
    else:
          
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
    
        sum_group_nino = xr.concat([sum_group_nino, sum_tors_nino_2], dim=('dayofyear'))   



sum_group_1 = sum_group_nino[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nino[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
elninos_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nino_concat)},
                              coords={'dayofyear': sum_coord_nino_concat})  

     

###############################################################################        
###############################################################################
###############################################################################

    
for k in xrange(12):

    num_yrs_nina = len(nina_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nina = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:
    
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2], dim=('dayofyear'))  
        
        
 
sum_group_1 = sum_group_nina[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nina[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
laninas_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nina_concat)},
                              coords={'dayofyear': sum_coord_nina_concat})  


###############################################################################
###############################################################################
###############################################################################


Gauss_SmoothN = xr.concat([neutral_concats,neutral_concats,neutral_concats], dim='dayofyear')
Gauss_SmoothEN = xr.concat([elninos_concats,elninos_concats,elninos_concats], dim='dayofyear')
Gauss_SmoothLN = xr.concat([laninas_concats,laninas_concats,laninas_concats], dim='dayofyear')

Gauss_SmoothN = Gauss_SmoothN.grid.values
Gauss_SmoothEN = Gauss_SmoothEN.grid.values
Gauss_SmoothLN = Gauss_SmoothLN.grid.values
    
Gauss_SmoothN = gfilt(Gauss_SmoothN*1.0,sigma=15.0)
Gauss_SmoothEN = gfilt(Gauss_SmoothEN*1.0,sigma=15.0)
Gauss_SmoothLN = gfilt(Gauss_SmoothLN*1.0,sigma=15.0)

sliced_gaussN2 = Gauss_SmoothN[len(neutral_concats.grid):len(neutral_concats.grid)*2]
sliced_gaussEN2 = Gauss_SmoothEN[len(elninos_concats.grid):len(elninos_concats.grid)*2]
sliced_gaussLN2 = Gauss_SmoothLN[len(laninas_concats.grid):len(laninas_concats.grid)*2]


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)

neut_comp = {}
nino_comp = {}
nina_comp = {}


###############################################################################
###############################################################################
###############################################################################


for k, (i, j) in enumerate(zip(month_list, day_list)):

    print 'Running month: '+i+' and day: '+j+'...'
   
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor.severe_load() 
    
    
    for s, b in enumerate(alltor.neutral_base_yrs):
        
        if s == 0:
            
            da = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))

        elif s == 1:
            
            da2 = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da, da2], dim=('time'))

        else:

            da2 = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da_group, da2], dim=('time'))            
            

    for d, n in enumerate(alltor.el_nino_yrs):
        
        if d == 0:
            
            na = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))      

        elif d == 1:

            na2 = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na, na2], dim=('time'))
            
        else:
            
            na2 = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na_group, na2], dim=('time')) 
            
        
    for f, m in enumerate(alltor.la_nina_yrs):
    
        if f == 0:
        
            la = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))       
    
        elif f == 1:
            
            la2 = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))            
            la_group = xr.concat([la, la2], dim=('time'))
            
        else:
            
            la2 = SE_tors_ef1.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))
            la_group = xr.concat([la_group, la2], dim=('time'))             

            
    neut_comp[k] = da_group.sum(['x','y'])
    nino_comp[k] = na_group.sum(['x','y'])
    nina_comp[k] = la_group.sum(['x','y']) 
    
    
###############################################################################
###############################################################################
###############################################################################
    

for k in xrange(12):
    
    num_yrs_neut = len(neut_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)

    elif k == 1:
        
        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
      
    else:

        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear'))  
        
        

sum_group_1 = sum_group_neut[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_neut[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
neutral_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_neut_concat)},
                              coords={'dayofyear': sum_coord_neut_concat})  

 

###############################################################################
###############################################################################
###############################################################################
  
    
for k in xrange(12):
    
    num_yrs_nino = len(nino_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nino = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
    
    elif k == 1:
        
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
      
        sum_group_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
     
    else:
          
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
    
        sum_group_nino = xr.concat([sum_group_nino, sum_tors_nino_2], dim=('dayofyear'))   



sum_group_1 = sum_group_nino[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nino[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
elninos_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nino_concat)},
                              coords={'dayofyear': sum_coord_nino_concat})  

     

###############################################################################        
###############################################################################
###############################################################################

    
for k in xrange(12):

    num_yrs_nina = len(nina_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nina = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:
    
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2], dim=('dayofyear'))  
        
        
 
sum_group_1 = sum_group_nina[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nina[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
laninas_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nina_concat)},
                              coords={'dayofyear': sum_coord_nina_concat})  


###############################################################################
###############################################################################
###############################################################################


Gauss_SmoothN = xr.concat([neutral_concats,neutral_concats,neutral_concats], dim='dayofyear')
Gauss_SmoothEN = xr.concat([elninos_concats,elninos_concats,elninos_concats], dim='dayofyear')
Gauss_SmoothLN = xr.concat([laninas_concats,laninas_concats,laninas_concats], dim='dayofyear')

Gauss_SmoothN = Gauss_SmoothN.grid.values
Gauss_SmoothEN = Gauss_SmoothEN.grid.values
Gauss_SmoothLN = Gauss_SmoothLN.grid.values
    
Gauss_SmoothN = gfilt(Gauss_SmoothN*1.0,sigma=15.0)
Gauss_SmoothEN = gfilt(Gauss_SmoothEN*1.0,sigma=15.0)
Gauss_SmoothLN = gfilt(Gauss_SmoothLN*1.0,sigma=15.0)

sliced_gaussN3 = Gauss_SmoothN[len(neutral_concats.grid):len(neutral_concats.grid)*2]
sliced_gaussEN3 = Gauss_SmoothEN[len(elninos_concats.grid):len(elninos_concats.grid)*2]
sliced_gaussLN3 = Gauss_SmoothLN[len(laninas_concats.grid):len(laninas_concats.grid)*2]


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)

neut_comp = {}
nino_comp = {}
nina_comp = {}


###############################################################################
###############################################################################
###############################################################################


for k, (i, j) in enumerate(zip(month_list, day_list)):

    print 'Running month: '+i+' and day: '+j+'...'
   
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor.severe_load() 
    
    
    for s, b in enumerate(alltor.neutral_base_yrs):
        
        if s == 0:
            
            da = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))

        elif s == 1:
            
            da2 = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da, da2], dim=('time'))

        else:

            da2 = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da_group, da2], dim=('time'))            
            

    for d, n in enumerate(alltor.el_nino_yrs):
        
        if d == 0:
            
            na = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))      

        elif d == 1:

            na2 = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na, na2], dim=('time'))
            
        else:
            
            na2 = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na_group, na2], dim=('time')) 
            
        
    for f, m in enumerate(alltor.la_nina_yrs):
    
        if f == 0:
        
            la = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))       
    
        elif f == 1:
            
            la2 = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))            
            la_group = xr.concat([la, la2], dim=('time'))
            
        else:
            
            la2 = SE_tors_ef2.sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))
            la_group = xr.concat([la_group, la2], dim=('time'))             

            
    neut_comp[k] = da_group.sum(['x','y'])
    nino_comp[k] = na_group.sum(['x','y'])
    nina_comp[k] = la_group.sum(['x','y'])
    
    
###############################################################################
###############################################################################
###############################################################################
    

for k in xrange(12):
    
    num_yrs_neut = len(neut_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)

    elif k == 1:
        
        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
      
    else:

        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear'))  
        
        

for k in xrange(12):
    
    num_yrs_neut = len(neut_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)

    elif k == 1:
        
        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
      
    else:

        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear'))  
        
        

sum_group_1 = sum_group_neut[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_neut[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_neut_concat = np.hstack([sum_group_neut[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_neut[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
neutral_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_neut_concat)},
                              coords={'dayofyear': sum_coord_neut_concat})  

 

###############################################################################
###############################################################################
###############################################################################
  
    
for k in xrange(12):
    
    num_yrs_nino = len(nino_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nino = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
    
    elif k == 1:
        
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
      
        sum_group_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
     
    else:
          
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
    
        sum_group_nino = xr.concat([sum_group_nino, sum_tors_nino_2], dim=('dayofyear'))   



sum_group_1 = sum_group_nino[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nino[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nino_concat = np.hstack([sum_group_nino[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nino[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
elninos_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nino_concat)},
                              coords={'dayofyear': sum_coord_nino_concat})  

     

###############################################################################        
###############################################################################
###############################################################################

    
for k in xrange(12):

    num_yrs_nina = len(nina_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nina = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:
    
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2], dim=('dayofyear'))  
        
        
 
sum_group_1 = sum_group_nina[(31+28+32):(31+28+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].values,
                                   sum_group_1.values,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].values,
                                   sum_group_2.values,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].values,
                                   sum_group_3.values,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].values,
                                   sum_group_4.values,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].values,
                                   sum_group_5.values,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].values,
                                   sum_group_6.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].values,
                                   sum_group_7.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].values,
                                   sum_group_8.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].values,
                                   sum_group_9.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].values])


coord_group_1 = sum_group_nina[(31+28+32):(31+28+34)].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33)].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34)].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33)].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34)].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34)].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33)].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34)].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33)].coords['dayofyear'].values[1]


sum_coord_nina_concat = np.hstack([sum_group_nina[:(31+28+32)].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nina[(31+28+34):(31+28+32+31)].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32)].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32)].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32)].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31)].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32)].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31)].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32)].coords['dayofyear'].values])

    
laninas_concats = xr.Dataset({'grid': (['dayofyear'], sum_group_nina_concat)},
                              coords={'dayofyear': sum_coord_nina_concat})  


###############################################################################
###############################################################################
###############################################################################


Gauss_SmoothN = xr.concat([neutral_concats,neutral_concats,neutral_concats], dim='dayofyear')
Gauss_SmoothEN = xr.concat([elninos_concats,elninos_concats,elninos_concats], dim='dayofyear')
Gauss_SmoothLN = xr.concat([laninas_concats,laninas_concats,laninas_concats], dim='dayofyear')

Gauss_SmoothN = Gauss_SmoothN.grid.values
Gauss_SmoothEN = Gauss_SmoothEN.grid.values
Gauss_SmoothLN = Gauss_SmoothLN.grid.values
    
Gauss_SmoothN = gfilt(Gauss_SmoothN*1.0,sigma=15.0)
Gauss_SmoothEN = gfilt(Gauss_SmoothEN*1.0,sigma=15.0)
Gauss_SmoothLN = gfilt(Gauss_SmoothLN*1.0,sigma=15.0)

sliced_gaussN4 = Gauss_SmoothN[len(neutral_concats.grid):len(neutral_concats.grid)*2]
sliced_gaussEN4 = Gauss_SmoothEN[len(elninos_concats.grid):len(elninos_concats.grid)*2]
sliced_gaussLN4 = Gauss_SmoothLN[len(laninas_concats.grid):len(laninas_concats.grid)*2]


###############################################################################
###############################################################################
###############################################################################

'''
def myticks(x,pos):

    if x == 0: return "$0$"

    exponent = int(np.log10(x))
    coeff = x/10**exponent

    return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent)
'''

grey_patch = mpatches.Patch(color='grey', alpha=0.4, label='All')


fig = plt.figure(figsize=(8,12))


ax1 = fig.add_axes([0.0, 0.75, 0.95, 0.225]) 

p1, = ax1.plot(range(0,len(sliced_gaussEN[:-1])),sliced_gaussEN[:-1]/np.sum(sliced_gaussEN[:-1]),'r-',linewidth=2.0)
p2, = ax1.plot(range(0,len(sliced_gaussLN[:-1])),sliced_gaussLN[:-1]/np.sum(sliced_gaussLN[:-1]),'b-',linewidth=2.0)
p3, = ax1.plot(range(0,len(sliced_gaussN[:-1])),sliced_gaussN[:-1]/np.sum(sliced_gaussN[:-1]),'k-',linewidth=2.0)    
#p4, = ax1.plot(range(0,total_days),Gauss_SmoothAN1/np.sum(Gauss_SmoothAN1),'--',color='grey',linewidth=2.0)  

p5 = ax1.fill_between(range(0,len(lo_thresh1)),lo_thresh1,hi_thresh1,color='grey',linewidth=1.0,alpha=0.5)

ax1.set_ylabel('Fraction of Tornado Days (EF1+)', fontsize=10)

ax1.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
plt.setp(ax1.get_yticklabels(), fontsize=10, rotation=35)
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_title('a) Annual Cycle of CONUS EF1+ Tornado Days')

ax1.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3,grey_patch],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "IQR"],
                    loc="upper right",
                    fancybox=True, fontsize=12)



ax2 = fig.add_axes([0.0, 0.5, 0.95, 0.225]) 

p1, = ax2.plot(range(0,len(sliced_gaussEN2)),sliced_gaussEN2/np.sum(sliced_gaussEN2),'r-',linewidth=2.0)
p2, = ax2.plot(range(0,len(sliced_gaussLN2)),sliced_gaussLN2/np.sum(sliced_gaussLN2),'b-',linewidth=2.0)
p3, = ax2.plot(range(0,len(sliced_gaussN2)),sliced_gaussN2/np.sum(sliced_gaussN2),'k-',linewidth=2.0)    
#p4, = ax2.plot(range(0,total_days),Gauss_SmoothAN2/np.sum(Gauss_SmoothAN2),'--',color='grey',linewidth=2.0)  

p5 = ax2.fill_between(range(0,len(lo_thresh2)),lo_thresh2,hi_thresh2,color='grey',linewidth=1.0,alpha=0.5)

ax2.set_ylabel('Fraction of Tornado Days (EF2+)', fontsize=10)

#ax2.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
ax2.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009])
plt.setp(ax2.get_yticklabels(), fontsize=10, rotation=35)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax2.set_title('b) Annual Cycle of CONUS EF2+ Tornado Days')

ax2.grid(True, linestyle='--', alpha=0.5)




ax3 = fig.add_axes([0.0, 0.25, 0.95, 0.225]) 

p1, = ax3.plot(range(0,len(sliced_gaussEN3)),sliced_gaussEN3/np.sum(sliced_gaussEN3),'r-',linewidth=2.0)
p2, = ax3.plot(range(0,len(sliced_gaussLN3)),sliced_gaussLN3/np.sum(sliced_gaussLN3),'b-',linewidth=2.0)
p3, = ax3.plot(range(0,len(sliced_gaussN3)),sliced_gaussN3/np.sum(sliced_gaussN3),'k-',linewidth=2.0)    
#p4, = ax3.plot(range(0,total_days),Gauss_SmoothAN3/np.sum(Gauss_SmoothAN3),'--',color='grey',linewidth=2.0)  

p5 = ax3.fill_between(range(0,len(lo_thresh3)),lo_thresh3,hi_thresh3,color='grey',linewidth=1.0,alpha=0.5)

ax3.set_ylabel('Fraction of Tornado Days (EF1+)', fontsize=10)

ax3.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
plt.setp(ax3.get_yticklabels(), fontsize=10, rotation=35)
#ax3.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax3.set_title('c) Annual Cycle of Southeast EF1+ Tornado Days')

ax3.grid(True, linestyle='--', alpha=0.5)




ax4 = fig.add_axes([0.0, 0.0, 0.95, 0.225]) 

p1, = ax4.plot(range(0,len(sliced_gaussEN4)),sliced_gaussEN4/np.sum(sliced_gaussEN4),'r-',linewidth=2.0)
p2, = ax4.plot(range(0,len(sliced_gaussLN4)),sliced_gaussLN4/np.sum(sliced_gaussLN4),'b-',linewidth=2.0)
p3, = ax4.plot(range(0,len(sliced_gaussN4)),sliced_gaussN4/np.sum(sliced_gaussN4),'k-',linewidth=2.0)    
#p4, = ax4.plot(range(0,total_days),Gauss_SmoothAN4/np.sum(Gauss_SmoothAN4),'--',color='grey',linewidth=2.0)  

p5 = ax4.fill_between(range(0,len(lo_thresh4)),lo_thresh4,hi_thresh4,color='grey',linewidth=1.0,alpha=0.5)

ax4.set_ylabel('Fraction of Tornado Days (EF2+)', fontsize=10)
ax4.set_xlabel('Day of Year', fontsize=10)

#ax4.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
ax4.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009])
plt.setp(ax4.get_yticklabels(), fontsize=10, rotation=35)
#ax4.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax4.set_xticks(tick_locs) 
ax4.set_xticklabels(tick_lbls)
ax4.set_title('d) Annual Cycle of Southeast EF2+ Tornado Days')

ax4.grid(True, linestyle='--', alpha=0.5)



plt.savefig('wut.png', bbox_inches='tight', dpi=200)

#plt.show()


###############################################################################
###############################################################################
###############################################################################

'''
US_tors_ef1 = xr.open_dataset('tors_5316_ef1', decode_cf=True)
US_tors_ef2 = xr.open_dataset('tors_5316_ef2', decode_cf=True)
SE_tors_ef1 = xr.open_dataset('tors_5316_ef1_SE', decode_cf=True)
SE_tors_ef2 = xr.open_dataset('tors_5316_ef2_SE', decode_cf=True)

US_nuyr_ef1 = len(US_tors_ef1.grid.groupby('time.year').sum('time').year.values)
US_nuyr_ef2 = len(US_tors_ef2.grid.groupby('time.year').sum('time').year.values)
SE_nuyr_ef1 = len(SE_tors_ef1.grid.groupby('time.year').sum('time').year.values)
SE_nuyr_ef2 = len(SE_tors_ef2.grid.groupby('time.year').sum('time').year.values)

US_sumtors_ef1 = US_tors_ef1.grid.sum(['x','y'])
US_sumtors_ef2 = US_tors_ef2.grid.sum(['x','y'])
SE_sumtors_ef1 = SE_tors_ef1.grid.sum(['x','y'])
SE_sumtors_ef2 = SE_tors_ef2.grid.sum(['x','y'])

years_count = US_tors_ef1.grid.groupby('time.year').sum('time').year.values


temp_US_ef1 = {}
temp_US_ef2 = {}
temp_SE_ef1 = {}
temp_SE_ef2 = {}

for u, i in enumerate(years_count):

    temp_US_ef1[u] = US_sumtors_ef1.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))
    temp_US_ef2[u] = US_sumtors_ef2.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))
    temp_SE_ef1[u] = SE_sumtors_ef1.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))
    temp_SE_ef2[u] = SE_sumtors_ef2.sel(time=slice(pd.to_datetime(str(i)+'-01-01'),pd.to_datetime(str(i)+'-12-31')))
    
    if u == 0:
        
        files_1 = temp_US_ef1[u]
        files_2 = temp_US_ef2[u]
        files_3 = temp_SE_ef1[u]
        files_4 = temp_SE_ef2[u]
    
    if u == 1:
    
        files1 = np.dstack([files_1, temp_US_ef1[u]])    
        files2 = np.dstack([files_2, temp_US_ef2[u]]) 
        files3 = np.dstack([files_3, temp_SE_ef1[u]]) 
        files4 = np.dstack([files_4, temp_SE_ef2[u]])
        
    if u > 1:
        
        files1 = np.dstack([files1, temp_US_ef1[u][:365]])    
        files2 = np.dstack([files2, temp_US_ef2[u][:365]]) 
        files3 = np.dstack([files3, temp_SE_ef1[u][:365]]) 
        files4 = np.dstack([files4, temp_SE_ef2[u][:365]])
        
    
file1 = np.squeeze(files1)
file2 = np.squeeze(files2)
file3 = np.squeeze(files3)
file4 = np.squeeze(files4)


hi_thresh1 = np.zeros(len(file1[:,0]))
hi_thresh2 = np.zeros(len(file1[:,0]))
hi_thresh3 = np.zeros(len(file1[:,0]))
hi_thresh4 = np.zeros(len(file1[:,0]))

lo_thresh1 = np.zeros(len(file1[:,0]))
lo_thresh2 = np.zeros(len(file1[:,0]))
lo_thresh3 = np.zeros(len(file1[:,0]))
lo_thresh4 = np.zeros(len(file1[:,0]))

for i in xrange(len(file1[:,0])):
    
    hi_thresh1[i] = np.nanpercentile(file1[i,:], 97.5)
    hi_thresh2[i] = np.nanpercentile(file2[i,:], 97.5)
    hi_thresh3[i] = np.nanpercentile(file3[i,:], 97.5)
    hi_thresh4[i] = np.nanpercentile(file4[i,:], 97.5)

    lo_thresh1[i] = np.nanpercentile(file1[i,:], 2.5)
    lo_thresh2[i] = np.nanpercentile(file2[i,:], 2.5)
    lo_thresh3[i] = np.nanpercentile(file3[i,:], 2.5)
    lo_thresh4[i] = np.nanpercentile(file4[i,:], 2.5)
    
    




for i in xrange(len(temp_US_ef1[0])):
    
    
    
    

    three_sum_ef1_US = xr.concat([temp_US_ef1,temp_US_ef1,temp_US_ef1], dim='time')
    three_sum_ef2_US = xr.concat([temp_US_ef2,temp_US_ef2,temp_US_ef2], dim='time')
    three_sum_ef1_SE = xr.concat([temp_SE_ef1,temp_SE_ef1,temp_SE_ef1], dim='time')
    three_sum_ef2_SE = xr.concat([temp_SE_ef2,temp_SE_ef2,temp_SE_ef2], dim='time')

    Gauss_Smooth_ef1_US = gfilt(three_sum_ef1_US*1.0, sigma=15.0)
    Gauss_Smooth_ef2_US = gfilt(three_sum_ef2_US*1.0, sigma=15.0)
    Gauss_Smooth_ef1_SE = gfilt(three_sum_ef1_SE*1.0, sigma=15.0)
    Gauss_Smooth_ef2_SE = gfilt(three_sum_ef2_SE*1.0, sigma=15.0)

    sliced_ef1_US[u] = Gauss_Smooth_ef1_US[len(temp_US_ef1):len(temp_US_ef1)*2]
    sliced_ef2_US[u] = Gauss_Smooth_ef2_US[len(temp_US_ef2):len(temp_US_ef2)*2]
    sliced_ef1_SE[u] = Gauss_Smooth_ef1_SE[len(temp_SE_ef1):len(temp_SE_ef1)*2]
    sliced_ef2_SE[u] = Gauss_Smooth_ef2_SE[len(temp_SE_ef2):len(temp_SE_ef2)*2]


for i in xrange(len(sliced_ef1_US)):   
    
    if i == 0:
        
        files_1 = sliced_ef1_US[i]
        files_2 = sliced_ef2_US[i]
        files_3 = sliced_ef1_SE[i]
        files_4 = sliced_ef2_SE[i]
    
    if i == 1:
    
        files1 = np.dstack([files_1, sliced_ef1_US[i]])    
        files2 = np.dstack([files_2, sliced_ef2_US[i]]) 
        files3 = np.dstack([files_3, sliced_ef1_SE[i]]) 
        files4 = np.dstack([files_4, sliced_ef2_SE[i]])
        
    if i > 1:
        
        files1 = np.dstack([files1, sliced_ef1_US[i][:365]])    
        files2 = np.dstack([files2, sliced_ef2_US[i][:365]]) 
        files3 = np.dstack([files3, sliced_ef1_SE[i][:365]]) 
        files4 = np.dstack([files4, sliced_ef2_SE[i][:365]])          
        
'''

