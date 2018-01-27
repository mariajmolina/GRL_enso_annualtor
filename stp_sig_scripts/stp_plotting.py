#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 11:39:16 2017

Maria J. Molina
Ph.D. Student
Central Michigan University

"""

###############################################################################
###############################################################################
###############################################################################


from __future__ import division

import mm_pkg as pk
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gfilt
from mpl_toolkits.basemap import Basemap
from itertools import product
import pickle
from shapely.geometry import MultiPoint, Point, Polygon
import shapefile


###############################################################################
###############################################################################
###############################################################################



def make_colormap(colors):
    
    from matplotlib.colors import LinearSegmentedColormap, ColorConverter

    z  = np.array(sorted(colors.keys()))
    n  = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / (zn - z1)

    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        Ci = colors[z[i]]      
        if type(Ci) == str:
            RGB = CC.to_rgb(Ci)
        else:
            RGB = Ci
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])

    cmap_dict = {}
    cmap_dict['red']   = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue']  = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)
    
    return mymap    
    
    
###############################################################################
############################ALL YEARS##########################################
###############################################################################


datas = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)

num_yrs = len(datas.stp.groupby('time.year').sum('time').year.values)

sum_tors = datas.stp.groupby('time.dayofyear').mean('time')

three_sum_tors = xr.concat([sum_tors,sum_tors,sum_tors], dim='dayofyear')

Gauss_SmoothAN = np.divide(three_sum_tors, num_yrs)

for i, j in product(xrange(len(Gauss_SmoothAN[0,:,0])),xrange(len(Gauss_SmoothAN[0,0,:]))):
    
    Gauss_SmoothAN[:,i,j] = gfilt(Gauss_SmoothAN[:,i,j]*1.0, sigma=15.0)

sliced_gauss = Gauss_SmoothAN[len(sum_tors[:,0,0]):len(sum_tors[:,0,0])*2,:,:]

sliced_gauss = np.divide(sliced_gauss,np.sum(sliced_gauss, axis=0))

gauss_peak_AN = np.ndarray.argmax(sliced_gauss.values, axis=0)

with open('gauss_peak_AN', 'wb') as output:
    pickle.dump(gauss_peak_AN, output, pickle.HIGHEST_PROTOCOL)


###############################################################################
###############################################################################
###############################################################################


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str) 

  
data = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)


neut_comp = {}
nino_comp = {}
nina_comp = {}


for k, (i, j) in enumerate(zip(month_list, day_list)):

    print 'Running month: '+i+' and day: '+j+'...'
   
    time_1 = '1979-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=True, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor.severe_load() 
    
    
    for s, b in enumerate(alltor.neutral_base_yrs):
        
        if s == 0:
            
            da = data['stp'].sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))

        elif s == 1:
            
            da2 = data['stp'].sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da, da2], dim=('time'))

        else:

            da2 = data['stp'].sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da_group, da2], dim=('time'))            
            

    for d, n in enumerate(alltor.el_nino_yrs):
        
        if d == 0:
            
            na = data['stp'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))      

        elif d == 1:

            na2 = data['stp'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na, na2], dim=('time'))
            
        else:
            
            na2 = data['stp'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na_group, na2], dim=('time')) 
            
        
    for f, m in enumerate(alltor.la_nina_yrs):
    
        if f == 0:
        
            la = data['stp'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))       
    
        elif f == 1:
            
            la2 = data['stp'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))            
            la_group = xr.concat([la, la2], dim=('time'))
            
        else:
            
            la2 = data['stp'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))
            la_group = xr.concat([la_group, la2], dim=('time'))             

            
    neut_comp[k] = da_group
    nino_comp[k] = na_group
    nina_comp[k] = la_group      

    
neut_co = {}
nino_co = {}
nina_co = {}
    
for k in xrange(12):
    
    neut_co[k] = neut_comp[k]
    nino_co[k] = nino_comp[k]
    nina_co[k] = nina_comp[k]
    

for k in xrange(12):

    neut_comp[k].to_netcdf('neut_co_'+str(k))
    
    nino_comp[k].to_netcdf('nino_co_'+str(k))
    
    nina_comp[k].to_netcdf('nina_co_'+str(k))
    
    
#############################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################
#############################################################################################################################################################################################################################################
##############load pregenerated files########################################################################################################################################################################################################
#############################################################################################################################################################################################################################################
 
    
neut_co = {}
 
for k in xrange(12):

    neut_co[k] = xr.open_dataarray('neut_co_'+str(k), decode_cf=True)
    neut_co[k].values
    print k
      
    
print 'entering loop..'
    
for k in xrange(12):
    
    num_yrs_neut = len(neut_co[k].groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)
        
        print 'done 0'

    elif k == 1:
        
        sum_tors_neut_2 = neut_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))

        print 'done 1'        
      
    else:

        sum_tors_neut_2 = neut_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear')) 
        
        print 'done '+str(k)
        
print 'exiting loop..'        

sum_group_1 = sum_group_neut[(31+28+32):(31+28+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_neut_concat = np.vstack([sum_group_neut[:(31+28+32),:,:].values,
                                   sum_group_1.values,
                                   sum_group_neut[(31+28+34):(31+28+32+31),:,:].values,
                                   sum_group_2.values,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32),:,:].values,
                                   sum_group_3.values,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31),:,:].values,
                                   sum_group_4.values,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32),:,:].values,
                                   sum_group_5.values,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32),:,:].values,
                                   sum_group_6.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31),:,:].values,
                                   sum_group_7.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32),:,:].values,
                                   sum_group_8.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31),:,:].values,
                                   sum_group_9.values,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32),:,:].values])


coord_group_1 = sum_group_neut[(31+28+32):(31+28+34),:,:].coords['dayofyear'].values[1]
coord_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33),:,:].coords['dayofyear'].values[1]
coord_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33),:,:].coords['dayofyear'].values[1]
coord_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34),:,:].coords['dayofyear'].values[1]
coord_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33),:,:].coords['dayofyear'].values[1]
coord_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33),:,:].coords['dayofyear'].values[1]


sum_coord_neut_concat = np.hstack([sum_group_neut[:(31+28+32),:,:].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_neut[(31+28+34):(31+28+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_neut[(31+28+32+33):(31+28+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_neut[(31+28+32+31+34):(31+28+32+31+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_neut[(31+28+32+31+32+33):(31+28+32+31+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_neut[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32),:,:].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_neut[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_neut[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32),:,:].coords['dayofyear'].values])

    
neutral_concats = xr.Dataset({'grid': (['dayofyear','y','x'], sum_group_neut_concat)},
                              coords={'dayofyear': sum_coord_neut_concat})  

 

###############################################################################
###############################################################################
###############################################################################
  


nino_co = {}
    
for k in xrange(12):    
    
    nino_co[k] = xr.open_dataarray('nino_co_'+str(k), decode_cf=True)
    nino_co[k].values
    print k

    
for k in xrange(12):
    
    num_yrs_nino = len(nino_co[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nino = nino_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
    
    elif k == 1:
        
        sum_tors_nino_2 = nino_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
      
        sum_group_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
     
    else:
          
        sum_tors_nino_2 = nino_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
    
        sum_group_nino = xr.concat([sum_group_nino, sum_tors_nino_2], dim=('dayofyear'))   



sum_group_1 = sum_group_nino[(31+28+32):(31+28+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
#sum_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nino_concat = np.vstack([sum_group_nino[:(31+28+32),:,:].values,
                                   sum_group_1.values,
                                   sum_group_nino[(31+28+34):(31+28+32+31),:,:].values,
                                   sum_group_2.values,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32),:,:].values,
                                   sum_group_3.values,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31),:,:].values,
                                   sum_group_4.values,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+34),:,:].values,
#                                   sum_group_5.values,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32),:,:].values,
                                   sum_group_6.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31),:,:].values,
                                   sum_group_7.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32),:,:].values,
                                   sum_group_8.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31),:,:].values,
                                   sum_group_9.values,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32),:,:].values])


coord_group_1 = sum_group_nino[(31+28+32):(31+28+34),:,:].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nino[(31+28+32+31):(31+28+32+33),:,:].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nino[(31+28+32+31+32):(31+28+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nino[(31+28+32+31+32+31):(31+28+32+31+32+33),:,:].coords['dayofyear'].values[1]
#coord_group_5 = sum_group_nino[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nino[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34),:,:].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nino[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33),:,:].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nino[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33),:,:].coords['dayofyear'].values[1]


sum_coord_nino_concat = np.hstack([sum_group_nino[:(31+28+32),:,:].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nino[(31+28+34):(31+28+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nino[(31+28+32+33):(31+28+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nino[(31+28+32+31+34):(31+28+32+31+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nino[(31+28+32+31+32+33):(31+28+32+31+32+31+34),:,:].coords['dayofyear'].values,
#                                   coord_group_5,
                                   sum_group_nino[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32),:,:].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nino[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nino[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32),:,:].coords['dayofyear'].values])

    
elninos_concats = xr.Dataset({'grid': (['dayofyear','y','x'], sum_group_nino_concat)},
                              coords={'dayofyear': sum_coord_nino_concat})  

     

###############################################################################        
###############################################################################
###############################################################################

    
nina_co = {}

for k in xrange(12):
    
    nina_co[k] = xr.open_dataarray('nina_co_'+str(k), decode_cf=True)
    nina_co[k].values
    print k
    

for k in xrange(12):

    num_yrs_nina = len(nina_co[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nina = nina_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
        print 'done '+str(k)
    
    elif k == 1:
        
        sum_tors_nina_2 = nina_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear')) 
        print 'done '+str(k)
        
    else:
    
        sum_tors_nina_2 = nina_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2], dim=('dayofyear')) 
        print 'done '+str(k)
        
        
 
sum_group_1 = sum_group_nina[(31+28+32):(31+28+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33),:,:].mean('dayofyear', skipna=True).expand_dims('time',axis=0)
 

sum_group_nina_concat = np.vstack([sum_group_nina[:(31+28+32),:,:].values,
                                   sum_group_1.values,
                                   sum_group_nina[(31+28+34):(31+28+32+31),:,:].values,
                                   sum_group_2.values,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32),:,:].values,
                                   sum_group_3.values,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31),:,:].values,
                                   sum_group_4.values,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32),:,:].values,
                                   sum_group_5.values,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32),:,:].values,
                                   sum_group_6.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31),:,:].values,
                                   sum_group_7.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32),:,:].values,
                                   sum_group_8.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31),:,:].values,
                                   sum_group_9.values,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32),:,:].values])


coord_group_1 = sum_group_nina[(31+28+32):(31+28+34),:,:].coords['dayofyear'].values[1]
coord_group_2 = sum_group_nina[(31+28+32+31):(31+28+32+33),:,:].coords['dayofyear'].values[1]
coord_group_3 = sum_group_nina[(31+28+32+31+32):(31+28+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_4 = sum_group_nina[(31+28+32+31+32+31):(31+28+32+31+32+33),:,:].coords['dayofyear'].values[1]
coord_group_5 = sum_group_nina[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_6 = sum_group_nina[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34),:,:].coords['dayofyear'].values[1]
coord_group_7 = sum_group_nina[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33),:,:].coords['dayofyear'].values[1]
coord_group_8 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34),:,:].coords['dayofyear'].values[1]
coord_group_9 = sum_group_nina[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33),:,:].coords['dayofyear'].values[1]


sum_coord_nina_concat = np.hstack([sum_group_nina[:(31+28+32),:,:].coords['dayofyear'].values,
                                   coord_group_1,
                                   sum_group_nina[(31+28+34):(31+28+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_2,
                                   sum_group_nina[(31+28+32+33):(31+28+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_3,
                                   sum_group_nina[(31+28+32+31+34):(31+28+32+31+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_4,
                                   sum_group_nina[(31+28+32+31+32+33):(31+28+32+31+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_5,
                                   sum_group_nina[(31+28+32+31+32+31+34):(31+28+32+31+32+31+32+32),:,:].coords['dayofyear'].values,
                                   coord_group_6,
                                   sum_group_nina[(31+28+32+31+32+31+32+34):(31+28+32+31+32+31+32+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_7,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+33):(31+28+32+31+32+31+32+32+31+32),:,:].coords['dayofyear'].values,
                                   coord_group_8,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+34):(31+28+32+31+32+31+32+32+31+32+31),:,:].coords['dayofyear'].values,
                                   coord_group_9,
                                   sum_group_nina[(31+28+32+31+32+31+32+32+31+32+33):(31+28+32+31+32+31+32+32+31+32+31+32),:,:].coords['dayofyear'].values])

    
laninas_concats = xr.Dataset({'grid': (['dayofyear','y','x'], sum_group_nina_concat)},
                              coords={'dayofyear': sum_coord_nina_concat})  

       
###############################################################################
###############################################################################
###############################################################################

  
neutral_concats.to_netcdf('neutral_concats')
elninos_concats.to_netcdf('elninos_concats')
laninas_concats.to_netcdf('laninas_concats')


###############################################################################
###############################################################################
###############################################################################


with open('gauss_peak_AN', 'rb') as f:
    gauss_peak_AN = pickle.load(f)
    
neutral_concats = xr.open_dataarray('neutral_concats', decode_cf=True)
elninos_concats = xr.open_dataarray('elninos_concats', decode_cf=True)
laninas_concats = xr.open_dataarray('laninas_concats', decode_cf=True)

neutral_concats.values
elninos_concats.values
laninas_concats.values


###############################################################################
###############################################################################
###############################################################################
    
                 
Gauss_SmoothN = xr.concat([neutral_concats,neutral_concats,neutral_concats], dim='dayofyear')
Gauss_SmoothEN = xr.concat([elninos_concats,elninos_concats,elninos_concats], dim='dayofyear')
Gauss_SmoothLN = xr.concat([laninas_concats,laninas_concats,laninas_concats], dim='dayofyear')

Gauss_SmoothN = Gauss_SmoothN.values
Gauss_SmoothEN = Gauss_SmoothEN.values
Gauss_SmoothLN = Gauss_SmoothLN.values

    
for i, j in product(xrange(len(Gauss_SmoothN[0,:,0])),xrange(len(Gauss_SmoothN[0,0,:]))):
    
    Gauss_SmoothN[:,i,j] = gfilt(Gauss_SmoothN[:,i,j]*1.0,sigma=15.0)
    Gauss_SmoothEN[:,i,j] = gfilt(Gauss_SmoothEN[:,i,j]*1.0,sigma=15.0)
    Gauss_SmoothLN[:,i,j] = gfilt(Gauss_SmoothLN[:,i,j]*1.0,sigma=15.0)


sliced_gaussN = Gauss_SmoothN[len(neutral_concats[:,0,0]):len(neutral_concats[:,0,0])*2,:,:]
sliced_gaussEN = Gauss_SmoothEN[len(elninos_concats[:,0,0]):len(elninos_concats[:,0,0])*2,:,:]
sliced_gaussLN = Gauss_SmoothLN[len(laninas_concats[:,0,0]):len(laninas_concats[:,0,0])*2,:,:]

sliced_gaussN = np.divide(sliced_gaussN,np.sum(sliced_gaussN, axis=0))
sliced_gaussEN = np.divide(sliced_gaussEN,np.sum(sliced_gaussEN, axis=0))
sliced_gaussLN = np.divide(sliced_gaussLN,np.sum(sliced_gaussLN, axis=0))

gauss_peak_N = np.ndarray.argmax(sliced_gaussN,axis=0)
gauss_peak_EN = np.ndarray.argmax(sliced_gaussEN,axis=0)
gauss_peak_LN = np.ndarray.argmax(sliced_gaussLN,axis=0)

gauss_smooth = gfilt(gauss_peak_AN*1.0, sigma=1.5)
gauss_smooth_N = gfilt(gauss_peak_N*1.0, sigma=1.5)
gauss_smooth_EN = gfilt(gauss_peak_EN*1.0, sigma=1.5)
gauss_smooth_LN = gfilt(gauss_peak_LN*1.0, sigma=1.5)


###############################################################################
###############################################################################
###############################################################################


data_alls = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_ensoannualpaper', decode_cf=True)

datas_grid_load = data_alls.grid.values

data_alls_num_yrs = len(np.unique(data_alls.grid.coords['time.year'].values))

sumalls_tors = data_alls.grid.groupby('time.dayofyear').sum('time')

data_allS = np.divide(sumalls_tors, data_alls_num_yrs)

data_neut = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_neutral', decode_cf=True)
data_nino = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_nino', decode_cf=True)
data_nina = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_nina', decode_cf=True)

data_neut.grid.values
data_nino.grid.values
data_nina.grid.values

stpmask_alls = data_allS.sum('dayofyear')
stpmask_neut = data_neut.grid.sum('dayofyear')
stpmask_nino = data_nino.grid.sum('dayofyear')
stpmask_nina = data_nina.grid.sum('dayofyear')


###############################################################################
################load significance##############################################
###############################################################################


neut_stp_stack = np.load('neut_stp_smooth.npy')
nina_stp_stack = np.load('nina_stp_smooth.npy')
nino_stp_stack = np.load('nino_stp_smooth.npy')


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
        
        
###############################################################################
###############################################################################
###############################################################################
        
        
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
 

for i, j in product(xrange(len(gauss_smooth[:,0])),xrange(len(gauss_smooth[0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    xpt, ypt = m(x,y,inverse=True)


    if not m.is_land(x,y):
        
        gauss_smooth[i,j] = None
        gauss_smooth_N[i,j] = None
        gauss_smooth_EN[i,j] = None
        gauss_smooth_LN[i,j] = None
        neut_stp_stack[:,i,j] = None
        nina_stp_stack[:,i,j] = None
        nino_stp_stack[:,i,j] = None
        
    
    if not in_us(ypt, xpt):

        gauss_smooth[i,j] = None
        gauss_smooth_N[i,j] = None
        gauss_smooth_EN[i,j] = None
        gauss_smooth_LN[i,j] = None
        neut_stp_stack[:,i,j] = None
        nina_stp_stack[:,i,j] = None
        nino_stp_stack[:,i,j] = None
        
    
    if np.isfinite(gauss_smooth[i,j]) and ypt > 50.0:

        gauss_smooth[i,j] = None
        neut_stp_stack[:,i,j] = None
        nina_stp_stack[:,i,j] = None
        nino_stp_stack[:,i,j] = None        
        
    if np.isfinite(gauss_smooth_N[i,j]) and ypt > 50.0:
        
        gauss_smooth_N[i,j] = None
        
    if np.isfinite(gauss_smooth_EN[i,j]) and ypt > 50.0:
        
        gauss_smooth_EN[i,j] = None
        
    if np.isfinite(gauss_smooth_LN[i,j]) and ypt > 50.0:
        
        gauss_smooth_LN[i,j] = None
        
    print i, j


###############################################################################
###############################################################################
###############################################################################
        
        
g_mask = np.ma.masked_invalid(gauss_smooth)
g_mask_all = np.ma.masked_array(g_mask, stpmask_alls<=2.5)
        
g_mask3 = np.ma.masked_invalid(gauss_smooth_N)
g_mask_neut = np.ma.masked_array(g_mask3, stpmask_neut<=2.5)

g_mask = np.ma.masked_invalid(gauss_smooth_EN)
g_mask_nino = np.ma.masked_array(g_mask, stpmask_nino<=2.5)

g_nino_neut = g_mask_nino - g_mask_neut

g_mask = np.ma.masked_invalid(gauss_smooth_LN)
g_mask_nina = np.ma.masked_array(g_mask, stpmask_nina<=2.5)

g_nina_neut = g_mask_nina - g_mask_neut

        
###############################################################################
###############################################################################
###############################################################################


nino_diff = nino_stp_stack - neut_stp_stack

nina_diff = nina_stp_stack - neut_stp_stack


nino_upper = np.nanpercentile(nino_diff, 97.5, axis=0)
nino_lower = np.nanpercentile(nino_diff, 2.5, axis=0)
nina_upper = np.nanpercentile(nina_diff, 97.5, axis=0)
nina_lower = np.nanpercentile(nina_diff, 2.5, axis=0)


###############################################################################
###############################################################################
###############################################################################


nino_sig = np.zeros(g_nino_neut.data.shape)
nina_sig = np.zeros(g_nina_neut.data.shape)

for i, j in product(xrange(len(g_nino_neut.data[:,0])),xrange(len(g_nino_neut.data[0,:]))):
        
    if np.isfinite(g_nino_neut.data[i,j]):
        
        if g_nino_neut.data[i,j] >= nino_upper[i,j]:
            
            nino_sig[i,j] = 1
            
        if g_nino_neut.data[i,j] <= nino_lower[i,j]:

            nino_sig[i,j] = 1
        
    if np.isfinite(g_nina_neut.data[i,j]):
        
        if g_nina_neut.data[i,j] >= nina_upper[i,j]:
            
            nina_sig[i,j] = 1
            
        if g_nina_neut.data[i,j] <= nina_lower[i,j]:

            nina_sig[i,j] = 1
        
    print i, j


###############################################################################
###############################################################################
###############################################################################


nino_sig_mask = np.ma.masked_invalid(nino_sig)
g_mask_nino_mask = np.ma.masked_array(nino_sig_mask, stpmask_nino<=2.5)

nina_sig_mask = np.ma.masked_invalid(nina_sig)
g_mask_nina_mask = np.ma.masked_array(nina_sig_mask, stpmask_nina<=2.5)


###############################################################################
###############################################################################
###############################################################################
    

maria_color = ({0.:'midnightblue',
                0.2:'mediumblue',
                0.3:'cornflowerblue',
                0.35:'lightsteelblue',
                0.4:'darkgrey',
                0.45:'darkgoldenrod',
                0.5:'goldenrod',
                0.55:'lightsalmon',
                0.6:'salmon',
                0.7:'red',
                0.8:'darkred',
                0.9166:'k',1.0:'k'})
                

cmap = make_colormap(maria_color)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
######################stp mask with significance###############################
###############################################################################


fig = plt.figure(figsize=(9.25,12))


ax1 = fig.add_axes([0.0, 0.666, 0.5, 0.33])

cs = ax1.pcolormesh(x1, y1, g_mask_all, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)
ax1.set_title(u'a) All Years', fontsize=13, loc='center')


ax2 = fig.add_axes([0.5, 0.666, 0.5, 0.33])

cs = ax2.pcolormesh(x1, y1, g_mask_neut, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
ax2.set_title(u'b) ENSO-Neutral Years', fontsize=13, loc='center')


ax3 = fig.add_axes([0.0, 0.3425, 0.5, 0.33])

cs = ax3.pcolormesh(x1, y1, g_mask_nino, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)
ax3.set_title(u'c) El Ni\xf1o Years', fontsize=13, loc='center')


ax4 = fig.add_axes([0.5, 0.3425, 0.5, 0.33])

cs3 = ax4.pcolormesh(x1, y1, np.divide((g_nino_neut),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

ax4.scatter(x1, y1, np.divide(g_mask_nino_mask,8), facecolors='k', edgecolors='k', zorder=10)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)
ax4.set_title(u'd) El Ni\xf1o - ENSO-Neutral Years', fontsize=13, loc='center')


ax5 = fig.add_axes([0.0, 0.005, 0.5, 0.33])

cs2 = ax5.pcolormesh(x1, y1, g_mask_nina, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)
ax5.set_title(u'e) La Ni\xf1a Years', fontsize=13, loc='center')

    
ax6 = fig.add_axes([0.5, 0.005, 0.5, 0.33])

cs = ax6.pcolormesh(x1, y1, np.divide(g_nina_neut,7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

ax6.scatter(x1, y1, np.divide(g_mask_nina_mask,8), facecolors='k', edgecolors='k', zorder=10)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
ax6.set_title(u'f) La Ni\xf1a - ENSO-Neutral Years', fontsize=13, loc='center')
    

cbar_ax = fig.add_axes([0.01, 0.0, 0.48, 0.015])
ticks_1 = [0,32,61,92,122,152,183,214,245,275,305,335]
tick_1 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
cbar = fig.colorbar(cs2, cax=cbar_ax, ticks=ticks_1, orientation='horizontal')
cbar.ax.set_xticklabels(tick_1)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Annual Peak Fraction of STP $\geq$ 1', fontsize=12) 

cbar_ax = fig.add_axes([0.51, 0.0, 0.48, 0.015])
cbar = fig.colorbar(cs3, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=11)
cbar.set_label('STP Peak Fraction Difference (weeks)', fontsize=12) 


plt.savefig('wut_3.png', bbox_inches='tight', dpi=200)


#plt.show()



###############################################################################
###############################################################################
###############################################################################
###############################################################################
######################stp mask#################################################
###############################################################################
'''

fig = plt.figure(figsize=(9.25,12))


ax1 = fig.add_axes([0.0, 0.666, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth)

g_mask2 = np.ma.masked_array(g_mask, stpmask_alls<=2.5)

cs = ax1.pcolormesh(x1, y1, g_mask2, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)

ax1.set_title(u'a) All Years', fontsize=13, loc='center')


ax2 = fig.add_axes([0.5, 0.666, 0.5, 0.33])

g_mask3 = np.ma.masked_invalid(gauss_smooth_N)

g_mask_3 = np.ma.masked_array(g_mask3, stpmask_neut<=2.5)

cs = ax2.pcolormesh(x1, y1, g_mask_3, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax2.set_title(u'b) Non-ENSO Years', fontsize=13, loc='center')


ax3 = fig.add_axes([0.0, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)

g_mask2 = np.ma.masked_array(g_mask, stpmask_nino<=2.5)

cs = ax3.pcolormesh(x1, y1, g_mask2, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)

ax3.set_title(u'c) El Ni\xf1o Years', fontsize=13, loc='center')


ax4 = fig.add_axes([0.5, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)

g_mask2 = np.ma.masked_array(g_mask, stpmask_nino<=2.5)

cs3 = ax4.pcolormesh(x1, y1, np.divide((g_mask2-g_mask_3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)

ax4.set_title(u'd) El Ni\xf1o - Non-ENSO Years', fontsize=13, loc='center')


ax5 = fig.add_axes([0.0, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)

g_mask2 = np.ma.masked_array(g_mask, stpmask_nina<=2.5)

cs2 = ax5.pcolormesh(x1, y1, g_mask2, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)

ax5.set_title(u'e) La Ni\xf1a Years', fontsize=13, loc='center')

    
ax6 = fig.add_axes([0.5, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)

g_mask2 = np.ma.masked_array(g_mask, stpmask_nina<=2.5)

cs = ax6.pcolormesh(x1, y1, np.divide((g_mask2-g_mask_3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax6.set_title(u'f) La Ni\xf1a - Non-ENSO Years', fontsize=13, loc='center')
    

cbar_ax = fig.add_axes([0.01, 0.0, 0.48, 0.015])

ticks_1 = [0,32,61,92,122,152,183,214,245,275,305,335]
tick_1 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


cbar = fig.colorbar(cs2, cax=cbar_ax, ticks=ticks_1, orientation='horizontal')

cbar.ax.set_xticklabels(tick_1)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Annual Peak Fraction of STP $\geq$ 1', fontsize=12) 


cbar_ax = fig.add_axes([0.51, 0.0, 0.48, 0.015])


cbar = fig.colorbar(cs3, cax=cbar_ax, orientation='horizontal')

cbar.ax.tick_params(labelsize=11)
cbar.set_label('STP Peak Fraction Difference (weeks)', fontsize=12) 


plt.savefig('wut_3.png', bbox_inches='tight', dpi=200)


#plt.show()



###############################################################################
###############################################################################
###############################################################################


###############################################################################
######################stp mask#################################################
###############################################################################


fig = plt.figure(figsize=(9.25,12))


ax1 = fig.add_axes([0.0, 0.666, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth)


cs = ax1.pcolormesh(x1, y1, g_mask, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)

ax1.set_title(u'a) All Years', fontsize=13, loc='center')


ax2 = fig.add_axes([0.5, 0.666, 0.5, 0.33])

g_mask3 = np.ma.masked_invalid(gauss_smooth_N)


cs = ax2.pcolormesh(x1, y1, g_mask3, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax2.set_title(u'b) ENSO-Neutral Years', fontsize=13, loc='center')


ax3 = fig.add_axes([0.0, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)

cs = ax3.pcolormesh(x1, y1, g_mask, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)

ax3.set_title(u'c) El Ni\xf1o Years', fontsize=13, loc='center')


ax4 = fig.add_axes([0.5, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)


cs3 = ax4.pcolormesh(x1, y1, np.divide((g_mask-g_mask3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)

ax4.set_title(u'd) El Ni\xf1o - ENSO-Neutral Years', fontsize=13, loc='center')


ax5 = fig.add_axes([0.0, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)


cs2 = ax5.pcolormesh(x1, y1, g_mask, vmin=0, vmax=365, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)

ax5.set_title(u'e) La Ni\xf1a Years', fontsize=13, loc='center')

    
ax6 = fig.add_axes([0.5, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)


cs = ax6.pcolormesh(x1, y1, np.divide((g_mask-g_mask3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax6.set_title(u'f) La Ni\xf1a - ENSO-Neutral Years', fontsize=13, loc='center')
    

cbar_ax = fig.add_axes([0.01, 0.0, 0.48, 0.015])

ticks_1 = [0,32,61,92,122,152,183,214,245,275,305,335]
tick_1 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


cbar = fig.colorbar(cs2, cax=cbar_ax, ticks=ticks_1, orientation='horizontal')

cbar.ax.set_xticklabels(tick_1)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Annual Peak Fraction of STP $\geq$ 1', fontsize=12) 


cbar_ax = fig.add_axes([0.51, 0.0, 0.48, 0.015])


cbar = fig.colorbar(cs3, cax=cbar_ax, orientation='horizontal')

cbar.ax.tick_params(labelsize=11)
cbar.set_label('STP Peak Fraction Difference (weeks)', fontsize=12) 


plt.savefig('wut_3.png', bbox_inches='tight', dpi=200)


#plt.show()



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#####################generating tor masks######################################
###############################################################################
    
    
data = xr.open_dataset('tors_5316_ef1_obs', decode_cf=True)

time_1 = pd.to_datetime('1979-01-01 00:00:00')
time_2 = pd.to_datetime('2016-12-31 21:00:00')

datas = data.grid.sel(time=slice(time_1,time_2))

num_yrs = len(datas.groupby('time.year').sum('time').year.values)

sum_tors = datas.groupby('time.dayofyear').sum('time')


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str) 


neut_comp = {}
nino_comp = {}
nina_comp = {}


for k, (i, j) in enumerate(zip(month_list, day_list)):

    print 'Running month: '+i+' and day: '+j+'...'
   
    time_1 = '1979-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,                ##############
                              enso_neutral=True, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor.severe_load() 
    
    
    for s, b in enumerate(alltor.neutral_base_yrs):
        
        if s == 0:
            
            da = data['grid'].sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))

        elif s == 1:
            
            da2 = data['grid'].sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da, da2], dim=('time'))

        else:

            da2 = data['grid'].sel(time=slice(pd.to_datetime(str(b)+'-'+str(i)+'-01'),pd.to_datetime(str(b)+'-'+str(i)+'-'+str(j))))
            da_group = xr.concat([da_group, da2], dim=('time'))            
            

    for d, n in enumerate(alltor.el_nino_yrs):
        
        if d == 0:
            
            na = data['grid'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))      

        elif d == 1:

            na2 = data['grid'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na, na2], dim=('time'))
            
        else:
            
            na2 = data['grid'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na_group, na2], dim=('time')) 
            
        
    for f, m in enumerate(alltor.la_nina_yrs):
    
        if f == 0:
        
            la = data['grid'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))       
    
        elif f == 1:
            
            la2 = data['grid'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))            
            la_group = xr.concat([la, la2], dim=('time'))
            
        else:
            
            la2 = data['grid'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))
            la_group = xr.concat([la_group, la2], dim=('time'))             
      

    neut_comp[k] = da_group
    nino_comp[k] = na_group
    nina_comp[k] = la_group      

    
    
for k in xrange(12):
    
    num_yrs_neut = len(neut_comp[k].groupby('time.year').sum('time').year.values)
    num_yrs_nino = len(nino_comp[k].groupby('time.year').sum('time').year.values)
    num_yrs_nina = len(nina_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_neut = neut_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)
        
        sum_tors_nino = nino_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
        
        sum_tors_nina = nina_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
        
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_torn_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
        sum_torn_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
        sum_torn_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:

        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
        
        wut1 = sum_tors_neut_2[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
        sum_tors_neut_stack = np.vstack([sum_tors_neut_2[:-2,:,:].values,wut1.values])
        sum_tors_neut_2 = xr.Dataset({'grid': (['dayofyear','y','x'], sum_tors_neut_stack)},
                                      coords={'dayofyear': sum_tors_neut_2.coords['dayofyear'].values[:-1]})     
                
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
        
        wut2 = sum_tors_nino_2[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
        sum_tors_nino_stack = np.vstack([sum_tors_nino_2[:-2,:,:].values,wut2.values])
        sum_tors_nino_2 = xr.Dataset({'grid': (['dayofyear','y','x'], sum_tors_nino_stack)},
                                      coords={'dayofyear': sum_tors_nino_2.coords['dayofyear'].values[:-1]})         
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)
        
        wut3 = sum_tors_nina_2[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
        sum_tors_nina_stack = np.vstack([sum_tors_nina_2[:-2,:,:].values,wut3.values])
        sum_tors_nina_2 = xr.Dataset({'grid': (['dayofyear','y','x'], sum_tors_nina_stack)},
                                      coords={'dayofyear': sum_tors_nina_2.coords['dayofyear'].values[:-1]}) 

        sum_torn_neut = xr.concat([sum_torn_neut, sum_tors_neut_2.grid], dim=('dayofyear'))        
        sum_torn_nino = xr.concat([sum_torn_nino, sum_tors_nino_2.grid], dim=('dayofyear'))   
        sum_torn_nina = xr.concat([sum_torn_nina, sum_tors_nina_2.grid], dim=('dayofyear'))         
    
    

###############################################################################
###############################################################################
###############################################################################
        
        
latlon_tor = xr.open_dataset('tor_grid_latlon', decode_cf=False)

x_tor, y_tor = latlon_tor.lons.values, latlon_tor.lats.values

y_tors = np.zeros(len(y_tor)-1)
x_tors = np.zeros(len(x_tor)-1)

for a in xrange(len(y_tors)):
    
    y_tors[a] = np.nanmean([y_tor[a],y_tor[a+1]])

for b in xrange(len(x_tors)):
    
    x_tors[b] = np.nanmean([x_tor[b],x_tor[b+1]])

x_to, y_to = np.meshgrid(x_tors, y_tors)


tor_mask_alls = sum_tors[:,:,:].sum(dim='dayofyear')
tor_mask_neut = sum_torn_neut[:,:,:].sum(dim='dayofyear')
tor_mask_nino = sum_torn_nino[:,:,:].sum(dim='dayofyear')
tor_mask_nina = sum_torn_nina[:,:,:].sum(dim='dayofyear')


toralls_mask = np.zeros(tor_mask_alls.shape)
torneut_mask = np.zeros(tor_mask_neut.shape)
tornino_mask = np.zeros(tor_mask_nino.shape)
tornina_mask = np.zeros(tor_mask_nina.shape)


for i, j in product(xrange(len(tor_mask_alls[:,0])),xrange(len(tor_mask_alls[0,:]))):
    
    if tor_mask_alls[i,j] != 0:
        
        toralls_mask[i,j] = np.nan
        
    if tor_mask_alls[i,j] == 0:
        
        toralls_mask[i,j] = 1  
        
    if tor_mask_neut[i,j] != 0:
        
        torneut_mask[i,j] = np.nan
        
    if tor_mask_neut[i,j] == 0:
        
        torneut_mask[i,j] = 1 

    if tor_mask_nino[i,j] != 0:
        
        tornino_mask[i,j] = np.nan
        
    if tor_mask_nino[i,j] == 0:
        
        tornino_mask[i,j] = 1 

    if tor_mask_nina[i,j] != 0:
        
        tornina_mask[i,j] = np.nan
        
    if tor_mask_nina[i,j] == 0:
        
        tornina_mask[i,j] = 1         
      
        
###############################################################################
###############################tor ob mask#####################################
###############################################################################


fig = plt.figure(figsize=(9.25,12))


ax1 = fig.add_axes([0.0, 0.666, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth)

cs = ax1.pcolormesh(x1, y1, g_mask, vmin=0, vmax=365, cmap=cmap)

ax1.scatter(x_to, y_to, toralls_mask*20, c='w', marker='s')

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)

ax1.set_title(u'a) All Years', fontsize=13, loc='center')


ax2 = fig.add_axes([0.5, 0.666, 0.5, 0.33])

g_mask3 = np.ma.masked_invalid(gauss_smooth_N)

cs = ax2.pcolormesh(x1, y1, g_mask3, vmin=0, vmax=365, cmap=cmap)

ax2.scatter(x_to, y_to, torneut_mask*20, c='w', marker='s')

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax2.set_title(u'b) Non-ENSO Years', fontsize=13, loc='center')


ax3 = fig.add_axes([0.0, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)

cs = ax3.pcolormesh(x1, y1, g_mask, vmin=0, vmax=365, cmap=cmap)

ax3.scatter(x_to, y_to, tornino_mask*20, c='w', marker='s')

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)

ax3.set_title(u'c) El Ni\xf1o Years', fontsize=13, loc='center')


ax4 = fig.add_axes([0.5, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)

cs3 = ax4.pcolormesh(x1, y1, np.divide((g_mask-g_mask3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

ax4.scatter(x_to, y_to, tornino_mask*20, c='w', marker='s')

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.01, fontsize=11)

ax4.set_title(u'd) El Ni\xf1o - Non-ENSO Years', fontsize=13, loc='center')


ax5 = fig.add_axes([0.0, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)

cs2 = ax5.pcolormesh(x1, y1, g_mask, vmin=0, vmax=365, cmap=cmap)

ax5.scatter(x_to, y_to, tornina_mask*20, c='w', marker='s')

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.01, fontsize=11)

ax5.set_title(u'e) La Ni\xf1a Years', fontsize=13, loc='center')

    
ax6 = fig.add_axes([0.5, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)

cs = ax6.pcolormesh(x1, y1, np.divide((g_mask-g_mask3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

ax6.scatter(x_to, y_to, tornina_mask*20, c='w', marker='s')

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax6.set_title(u'f) La Ni\xf1a - Non-ENSO Years', fontsize=13, loc='center')
    

cbar_ax = fig.add_axes([0.01, 0.0, 0.48, 0.015])

ticks_1 = [0,32,61,92,122,152,183,214,245,275,305,335]
tick_1 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


cbar = fig.colorbar(cs2, cax=cbar_ax, ticks=ticks_1, orientation='horizontal')

cbar.ax.set_xticklabels(tick_1)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Annual Peak Fraction of STP $\geq$ 1', fontsize=12) 


cbar_ax = fig.add_axes([0.51, 0.0, 0.48, 0.015])


cbar = fig.colorbar(cs3, cax=cbar_ax, orientation='horizontal')

cbar.ax.tick_params(labelsize=11)
cbar.set_label('STP Peak Fraction Difference (weeks)', fontsize=12) 


plt.savefig('wut_3.png', bbox_inches='tight', dpi=200)


#plt.show()

'''

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


