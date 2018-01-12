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
import matplotlib.patches as mpatches


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

#gauss_peak_AN = np.ndarray.argmax(sliced_gauss.values, axis=0)

with open('sliced_gauss', 'wb') as output:
    pickle.dump(sliced_gauss, output, pickle.HIGHEST_PROTOCOL)


###############################################################################
###############################################################################
###############################################################################
    
    
    
with open('sliced_gauss', 'rb') as f:
    sliced_gauss = pickle.load(f)
    
    
neut_co = {}
 
for k in xrange(12):

    neut_co[k] = xr.open_dataarray('neut_co_'+str(k), decode_cf=True)

nino_co = {}
    
for k in xrange(12):    
    
    nino_co[k] = xr.open_dataarray('nino_co_'+str(k), decode_cf=True)

nina_co = {}

for k in xrange(12):
    
    nina_co[k] = xr.open_dataarray('nina_co_'+str(k), decode_cf=True)
    
    
###############################################################################
###############################################################################
###############################################################################


for k in xrange(12):
    
    neut_co[k].values
    nino_co[k].values
    nina_co[k].values


###############################################################################
###############################################################################
###############################################################################
      
    
for k in xrange(12):
    
    num_yrs_neut = len(neut_co[k].groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)

    elif k == 1:
        
        sum_tors_neut_2 = neut_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
      
    else:

        sum_tors_neut_2 = neut_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear'))  
        
    print k
        
        

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

    print k


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

    
for k in xrange(12):

    num_yrs_nina = len(nina_co[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nina = nina_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_nina_2 = nina_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:
    
        sum_tors_nina_2 = nina_co[k].groupby('time.dayofyear').mean('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2], dim=('dayofyear'))  

    print k        
        
 
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

        
data_alls = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_ensoannualpaper', decode_cf=True)

data_alls_num_yrs = len(data_alls.grid.groupby('time.year').sum('time').year.values)

sumalls_tors = data_alls.grid.groupby('time.dayofyear').sum('time')

data_allS = np.divide(sumalls_tors, data_alls_num_yrs)

data_neut = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_neutral', decode_cf=True)
data_nino = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_nino', decode_cf=True)
data_nina = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_nina', decode_cf=True)

stpmask_alls = data_allS.sum('dayofyear')
stpmask_neut = data_neut.grid.sum('dayofyear')
stpmask_nino = data_nino.grid.sum('dayofyear')
stpmask_nina = data_nina.grid.sum('dayofyear')


###############################################################################
###############################################################################
###############################################################################


neutral_concat = neutral_concats.grid
elninos_concat = elninos_concats.grid
laninas_concat = laninas_concats.grid


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

for i, j in product(xrange(len(neutral_concat[0,:,0])),xrange(len(neutral_concat[0,0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    if not m.is_land(x,y):
        
        neutral_concat[:,i,j] = None
        elninos_concat[:,i,j] = None
        laninas_concat[:,i,j] = None
        
               
        
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


for i, j in product(xrange(len(neutral_concat[0,:,0])),xrange(len(neutral_concat[0,0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    xpt, ypt = m(x,y,inverse=True)
    
    if not in_us(ypt, xpt):

        neutral_concat[:,i,j] = None
        elninos_concat[:,i,j] = None
        laninas_concat[:,i,j] = None


for i, j in product(xrange(len(neutral_concat[0,:,0])),xrange(len(neutral_concat[0,0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    xpt, ypt = m(x,y,inverse=True)
    
        
    if np.all(np.isfinite(neutral_concat[:,i,j])) and ypt > 50.0:
        
        neutral_concat[:,i,j] = None
        
    if np.all(np.isfinite(elninos_concat[:,i,j])) and ypt > 50.0:
        
        elninos_concat[:,i,j] = None
        
    if np.all(np.isfinite(laninas_concat[:,i,j])) and ypt > 50.0:
        
        laninas_concat[:,i,j] = None


###############################################################################
###############################################################################
###############################################################################
        
        
for i, j in product(xrange(len(neutral_concat[0,:,0])),xrange(len(neutral_concat[0,0,:]))):
    
    if stpmask_neut[i,j] <= 2.5:
        
        neutral_concat[:,i,j] = None        

    if stpmask_nino[i,j] <= 2.5:
        
        elninos_concat[:,i,j] = None   
        
    if stpmask_nina[i,j] <= 2.5:
        
        laninas_concat[:,i,j] = None   


###############################################################################
###############################################################################
###############################################################################


neutral_con = neutral_concat.mean(['x','y'], skipna=True)
elninos_con = elninos_concat.mean(['x','y'], skipna=True)
laninas_con = laninas_concat.mean(['x','y'], skipna=True)

                   
Gauss_SmoothN = xr.concat([neutral_con,neutral_con,neutral_con], dim='dayofyear')
Gauss_SmoothEN = xr.concat([elninos_con,elninos_con,elninos_con], dim='dayofyear')
Gauss_SmoothLN = xr.concat([laninas_con,laninas_con,laninas_con], dim='dayofyear')


Gauss_SmoothN = gfilt(Gauss_SmoothN*1.0,sigma=15.0)
Gauss_SmoothEN = gfilt(Gauss_SmoothEN*1.0,sigma=15.0)
Gauss_SmoothLN = gfilt(Gauss_SmoothLN*1.0,sigma=15.0)


sliced_gaussN = Gauss_SmoothN[len(neutral_con):len(neutral_con)*2]
sliced_gaussEN = Gauss_SmoothEN[len(elninos_con):len(elninos_con)*2]
sliced_gaussLN = Gauss_SmoothLN[len(laninas_con):len(laninas_con)*2]


sliced_gaussN = np.divide(sliced_gaussN,np.sum(sliced_gaussN))
sliced_gaussEN = np.divide(sliced_gaussEN,np.sum(sliced_gaussEN))
sliced_gaussLN = np.divide(sliced_gaussLN,np.sum(sliced_gaussLN))



###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#sliced_g = xr.Dataset({'grid': (['dayofyear','y','x'], sliced_gauss)})  
#sliced_gN = xr.Dataset({'grid': (['dayofyear','y','x'], sliced_gaussN)})  
#sliced_gEN = xr.Dataset({'grid': (['dayofyear','y','x'], sliced_gaussEN)})  
#sliced_gLN = xr.Dataset({'grid': (['dayofyear','y','x'], sliced_gaussLN)})  

#sliced_G = sliced_g.grid.mean(['x','y'], skipna=True)
#sliced_GN = sliced_gN.grid.mean(['x','y'], skipna=True)
#sliced_GEN = sliced_gEN.grid.mean(['x','y'], skipna=True)
#sliced_GLN = sliced_gLN.grid.mean(['x','y'], skipna=True)


###############################################################################
#######IQR#####################################################################
###############################################################################


data = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)

iqr_stp = data.stp.resample(freq='D', dim='time', how='mean', skipna=True)


for i, j in product(xrange(len(iqr_stp[0,:,0])),xrange(len(iqr_stp[0,0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    if not m.is_land(x,y):
        
        iqr_stp[:,i,j] = None


for i, j in product(xrange(len(iqr_stp[0,:,0])),xrange(len(iqr_stp[0,0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    xpt, ypt = m(x,y,inverse=True)
    
    if not in_us(ypt, xpt):

        iqr_stp[:,i,j] = None

    if np.all(np.isfinite(iqr_stp[:,i,j])) and ypt > 50.0:

        iqr_stp[:,i,j] = None



for i, j in product(xrange(len(iqr_stp[0,:,0])),xrange(len(iqr_stp[0,0,:]))):
    
    if stpmask_alls[i,j] <= 2.5:
        
        iqr_stp[:,i,j] = None 



stp_years = np.unique(iqr_stp.coords['time'].dt.year.values)


iqr_files = {}

for i, j in enumerate(stp_years):
    
    iqr_filing = iqr_stp.sel(time=slice(pd.to_datetime(str(j)+'-01-01'),pd.to_datetime(str(j)+'-12-31')))
    
    iqr_filin = iqr_filing.mean(['lon','lat'], skipna=True)
    
    three_iqr = xr.concat([iqr_filin,iqr_filin,iqr_filin], dim='time')

    Gauss_three_iqr = gfilt(three_iqr*1.0, sigma=15.0)
    
    iqr_files[i] = Gauss_three_iqr[len(iqr_filin):len(iqr_filin)*2]
    


for i in xrange(len(iqr_files)):   
    
    if i == 0:
        
        files_1 = iqr_files[i]
    
    if i == 1:
    
        files1 = np.dstack([files_1, iqr_files[i][:365]])    
        
    if i > 1:
        
        files1 = np.dstack([files1, iqr_files[i][:365]]) 


file1 = np.squeeze(files1)



iqr_stp_forG = iqr_stp.groupby('time.dayofyear').mean()

three_iqr_stp = xr.concat([iqr_stp_forG,iqr_stp_forG,iqr_stp_forG], dim='dayofyear')

Gauss_Smooth_ef1_US = gfilt(three_iqr_stp*1.0, sigma=15.0)

Gauss_SmoothAN1 = Gauss_Smooth_ef1_US[len(iqr_stp_forG):len(iqr_stp_forG)*2]



hi_thresh1 = np.zeros(len(file1[:,0]))

lo_thresh1 = np.zeros(len(file1[:,0]))

for i in xrange(len(file1[:,0])):
    
    hi_thresh1[i] = np.divide(np.nanpercentile(file1[i,:], 75.), np.nansum(Gauss_SmoothAN1))

    lo_thresh1[i] = np.divide(np.nanpercentile(file1[i,:], 25.), np.nansum(Gauss_SmoothAN1))


    
###############################################################################
###############################################################################
###############################################################################


grey_patch = mpatches.Patch(color='grey', alpha=0.4, label='All')


fig = plt.figure(figsize=(8,4))


ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 

p1, = ax1.plot(range(0,len(sliced_gaussEN[:-1])),sliced_gaussEN[:-1],'r-',linewidth=2.0)
p2, = ax1.plot(range(0,len(sliced_gaussLN[:-1])),sliced_gaussLN[:-1],'b-',linewidth=2.0)
p3, = ax1.plot(range(0,len(sliced_gaussN[:-1])),sliced_gaussN[:-1],'k-',linewidth=2.0)    
#p4, = ax1.plot(range(0,len(sliced_G[:-1])),sliced_G[:-1],'--',color='grey',linewidth=2.0)  

p5 = ax1.fill_between(range(0,len(lo_thresh1)),lo_thresh1,hi_thresh1,color='grey',linewidth=1.0,alpha=0.5)

ax1.set_ylabel('Fraction of STP', fontsize=10)

ax1.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
plt.setp(ax1.get_yticklabels(), fontsize=10, rotation=35)
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

#ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_title('Annual Cycle of STP')

ax1.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3,grey_patch],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "IQR"],
                    loc="upper right",
                    fancybox=True, fontsize=12)


tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax1.set_xticks(tick_locs) 
ax1.set_xticklabels(tick_lbls)
ax1.set_xlabel('Day of Year', fontsize=10)


#plt.savefig('wut.png', bbox_inches='tight', dpi=200)


plt.show()


###############################################################################
###############################################################################
###############################################################################
