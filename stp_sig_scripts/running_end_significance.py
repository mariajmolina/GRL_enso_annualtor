#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Created on Tue Jan 23 10:31:21 2018

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


month_choice = 1
#1-12 int


with open('num_neut_yrs', 'rb') as f:
    num_neut_yrs = pickle.load(f)
    
with open('num_nina_yrs', 'rb') as f:
    num_nina_yrs = pickle.load(f)
    
with open('num_nino_yrs', 'rb') as f:
    num_nino_yrs = pickle.load(f)

month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

month_name = np.array(['jan','feb','mar','apr','may','jun',
                       'jul','aug','sep','oct','nov','dec'], dtype=str)
    
day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str) 


if month_choice == 1:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_1', decode_cf=True)    
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[0]
    num_nina_yrs = num_nina_yrs[0]
    num_nino_yrs = num_nino_yrs[0]
    month_list = month_list[0].astype(int)
    day_list = day_list[0].astype(int)
    month_name = month_name[0]

if month_choice == 2:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_2', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[1]
    num_nina_yrs = num_nina_yrs[1]
    num_nino_yrs = num_nino_yrs[1]
    month_list = month_list[1].astype(int)
    day_list = day_list[1].astype(int)
    month_name = month_name[1]

if month_choice == 3:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_3', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[2]
    num_nina_yrs = num_nina_yrs[2]
    num_nino_yrs = num_nino_yrs[2]
    month_list = month_list[2].astype(int)
    day_list = day_list[2].astype(int)
    month_name = month_name[2]

if month_choice == 4:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_4', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[3]
    num_nina_yrs = num_nina_yrs[3]
    num_nino_yrs = num_nino_yrs[3]
    month_list = month_list[3].astype(int)
    day_list = day_list[3].astype(int)
    month_name = month_name[3]

if month_choice == 5:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_5', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[4]
    num_nina_yrs = num_nina_yrs[4]
    num_nino_yrs = num_nino_yrs[4]
    month_list = month_list[4].astype(int)
    day_list = day_list[4].astype(int)
    month_name = month_name[4]

if month_choice == 6:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_6', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[5]
    num_nina_yrs = num_nina_yrs[5]
    num_nino_yrs = num_nino_yrs[5]
    month_list = month_list[5].astype(int)
    day_list = day_list[5].astype(int)
    month_name = month_name[5]

if month_choice == 7:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_7', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[6]
    num_nina_yrs = num_nina_yrs[6]
    num_nino_yrs = num_nino_yrs[6]
    month_list = month_list[6].astype(int)
    day_list = day_list[6].astype(int)
    month_name = month_name[6]

if month_choice == 8:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_8', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[7]
    num_nina_yrs = num_nina_yrs[7]
    num_nino_yrs = num_nino_yrs[7]
    month_list = month_list[7].astype(int)
    day_list = day_list[7].astype(int)
    month_name = month_name[7]

if month_choice == 9:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_9', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[8]
    num_nina_yrs = num_nina_yrs[8]
    num_nino_yrs = num_nino_yrs[8]
    month_list = month_list[8].astype(int)
    day_list = day_list[8].astype(int)
    month_name = month_name[8]

if month_choice == 10:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_10', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[9]
    num_nina_yrs = num_nina_yrs[9]
    num_nino_yrs = num_nino_yrs[9]
    month_list = month_list[9].astype(int)
    day_list = day_list[9].astype(int)
    month_name = month_name[9]

if month_choice == 11:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_11', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[10]
    num_nina_yrs = num_nina_yrs[10]
    num_nino_yrs = num_nino_yrs[10]
    month_list = month_list[10].astype(int)
    day_list = day_list[10].astype(int)
    month_name = month_name[10]

if month_choice == 12:
    print 'Calculating for month number '+str(month_choice)
    monthlies_stp = xr.open_dataarray('monthlies_stp_12', decode_cf=True)
    monthlies_stp.values
    num_neut_yrs = num_neut_yrs[11]
    num_nina_yrs = num_nina_yrs[11]
    num_nino_yrs = num_nino_yrs[11]
    month_list = month_list[11].astype(int)
    day_list = day_list[11].astype(int)
    month_name = month_name[11]


###############################################################################
###############################################################################
###############################################################################


neutnina_diff = np.zeros([1,day_list,277,349])
nina_diff = np.zeros([1,day_list,277,349])
nino_diff = np.zeros([1,day_list,277,349])


for _ in xrange(3334):

    yrs_choice = pd.date_range('1979','2016',freq='AS').year
       
    rand_neutnina = np.array([yrs_choice[np.random.choice(len(yrs_choice))] for i in xrange(num_neut_yrs)])        
    rand_nina = np.array([yrs_choice[np.random.choice(len(yrs_choice))] for i in xrange(num_nina_yrs)])
    rand_nino = np.array([yrs_choice[np.random.choice(len(yrs_choice))] for i in xrange(num_nino_yrs)])  

    for e, i in enumerate(rand_neutnina):
        
        if e == 0:
        
            ninaneut_neutnina_1 = monthlies_stp.sel(time=str(i)).expand_dims(dim='year_add').values[:,:day_list,:,:]
            
        elif e == 1:

            ninaneut_neutnina_2 = monthlies_stp.sel(time=str(i)).expand_dims(dim='year_add').values[:,:day_list,:,:]       
            ninaneut_stack = np.vstack([ninaneut_neutnina_1,ninaneut_neutnina_2])
                      
        else:
        
            ninaneut_neutnina_2 = monthlies_stp.sel(time=str(i)).expand_dims(dim='year_add').values[:,:day_list,:,:]           
            ninaneut_stack = np.vstack([ninaneut_neutnina_2,ninaneut_stack])
     
    for e, r in enumerate(rand_nina):        

        if e == 0:
        
            ninaneut_nina_1 = monthlies_stp.sel(time=str(r)).expand_dims(dim='year_add').values[:,:day_list,:,:]
           
        elif e == 1:

            ninaneut_nina_2 = monthlies_stp.sel(time=str(r)).expand_dims(dim='year_add').values[:,:day_list,:,:]      
            ninaneut_nina_stack = np.vstack([ninaneut_nina_1,ninaneut_nina_2])
                     
        else:
        
            ninaneut_nina_2 = monthlies_stp.sel(time=str(r)).expand_dims(dim='year_add').values[:,:day_list,:,:]          
            ninaneut_nina_stack = np.vstack([ninaneut_nina_2,ninaneut_nina_stack])
            
    for e, l in enumerate(rand_nino):        
            
        if e == 0:
        
            ninoneut_nino_1 = monthlies_stp.sel(time=str(l)).expand_dims(dim='year_add').values[:,:day_list,:,:]
            
        elif e == 1:

            ninoneut_nino_2 = monthlies_stp.sel(time=str(l)).expand_dims(dim='year_add').values[:,:day_list,:,:]
            ninoneut_nino_stack = np.vstack([ninoneut_nino_1,ninoneut_nino_2])
                      
        else:
        
            ninoneut_nino_2 = monthlies_stp.sel(time=str(l)).expand_dims(dim='year_add').values[:,:day_list,:,:]
            ninoneut_nino_stack = np.vstack([ninoneut_nino_2,ninoneut_nino_stack])                        


    init_files_concata = np.nanmean(ninaneut_stack, axis=0)
    init_files_concat_a = np.nanmean(ninaneut_nina_stack, axis=0)
    init_files_concatao = np.nanmean(ninoneut_nino_stack, axis=0)
                        
    init_files_concata = np.divide(init_files_concata, num_neut_yrs)
    init_files_concat_a = np.divide(init_files_concat_a, num_nina_yrs)
    init_files_concatao = np.divide(init_files_concatao, num_nino_yrs)
            
    neutnina_diff[0,:,:,:] = init_files_concata
    nina_diff[0,:,:,:] = init_files_concat_a
    nino_diff[0,:,:,:] = init_files_concatao
    
    
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+month_name+'/neutnina_month_'+month_name+'_'+str(_+1), 'wb') as output:
        pickle.dump(neutnina_diff, output, pickle.HIGHEST_PROTOCOL)
        
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+month_name+'/nina_month_'+month_name+'_'+str(_+1), 'wb') as output:
        pickle.dump(nina_diff, output, pickle.HIGHEST_PROTOCOL)
        
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+month_name+'/nino_month_'+month_name+'_'+str(_+1), 'wb') as output:
        pickle.dump(nino_diff, output, pickle.HIGHEST_PROTOCOL)
    
    print str(_)+' complete...'
    
    
###############################################################################
###############################################################################
###############################################################################

