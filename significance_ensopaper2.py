#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:36:59 2018

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
from scipy.ndimage import gaussian_filter as gfilt
from itertools import product


###############################################################################
###############################################################################
###############################################################################


que_num = 500


data = xr.open_dataset('tors_5316_ef1', decode_cf=True)


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str) 


###############################################################################
###############################################################################
###############################################################################


neut_comp = {}
nino_comp = {}
nina_comp = {}


for k, (i, j) in enumerate(zip(month_list, day_list)):

    print 'Running month: '+i+' and day: '+j+'...'
   
    time_1 = '1953-'+i+'-01 00:00:00'
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
    
    
###############################################################################
###############################################################################
###############################################################################



for _ in xrange(500):

    for k in xrange(12):

        neut_yrs_choice = neut_comp[k].groupby('time.year').sum('time').year.values
        nina_yrs_choice = nina_comp[k].groupby('time.year').sum('time').year.values
        nino_yrs_choice = nino_comp[k].groupby('time.year').sum('time').year.values

        joint_yrs_choice = np.hstack([nina_yrs_choice, neut_yrs_choice, nino_yrs_choice])

        num_neut_yrs_choice = len(neut_yrs_choice)      
        num_nina_yrs_choice = len(nina_yrs_choice)
        num_nino_yrs_choice = len(nino_yrs_choice)     
        
        randomizer = num_neut_yrs_choice + num_nina_yrs_choice + num_nino_yrs_choice
                
        num_ninaneut_yrs_choice = num_nina_yrs_choice + num_neut_yrs_choice
        num_ninoneut_yrs_choice = num_nino_yrs_choice + num_neut_yrs_choice
        
        rand_neut = np.array([joint_yrs_choice[np.random.choice(randomizer)] for i in xrange(num_neut_yrs_choice)])                
        rand_nina = np.array([joint_yrs_choice[np.random.choice(randomizer)] for i in xrange(num_nina_yrs_choice)])
        rand_nino = np.array([joint_yrs_choice[np.random.choice(randomizer)] for i in xrange(num_nino_yrs_choice)])  
        
        years_concat = xr.concat([nina_comp[k], neut_comp[k], nino_comp[k]], dim='time')
        
        neutneut_neut = {}
        ninaneut_nina = {}
        ninoneut_nino = {}
        
        for e, i in enumerate(rand_neut):
            
            neutneut_neut[e] = years_concat.sel(time=str(i))  
            
            if e == 0:
                
                files_1a = neutneut_neut[e]
                
            elif e == 1:
                
                files_2a = neutneut_neut[e]
                files_neut = xr.concat([files_1a,files_2a], dim='time')
                          
            elif e != 0 and e != 1:
                
                files_2a = neutneut_neut[e]
                files_neut = xr.concat([files_neut,files_2a], dim='time')
                     
        for e, r in enumerate(rand_nina):        
            
            ninaneut_nina[e] = years_concat.sel(time=str(r))

            if e == 0:
                
                files_1a = ninaneut_nina[e]
              
            elif e == 1:
                
                files_2a = ninaneut_nina[e]
                files_nina = xr.concat([files_1a,files_2a], dim='time')

            elif e != 0 and e != 1:
                
                files_2a = ninaneut_nina[e]
                files_nina = xr.concat([files_nina,files_2a], dim='time')          
            
        for e, l in enumerate(rand_nino):        
            
            ninoneut_nino[e] = years_concat.sel(time=str(l))

            if e == 0:
                
                files_1a = ninoneut_nino[e]
              
            elif e == 1:
                
                files_2a = ninoneut_nino[e]
                files_nino = xr.concat([files_1a,files_2a], dim='time')

            elif e != 0 and e != 1:
                
                files_2a = ninoneut_nino[e]
                files_nino = xr.concat([files_nino,files_2a], dim='time')                          

        if k == 0:

            init_files_neut = files_neut.groupby('time.dayofyear').sum('time')
            init_files_nina = files_nina.groupby('time.dayofyear').sum('time')
            init_files_nino = files_nino.groupby('time.dayofyear').sum('time')
                        
            init_files_neut = np.divide(init_files_neut, num_neut_yrs_choice)
            init_files_nina = np.divide(init_files_nina, num_nina_yrs_choice)
            init_files_nino = np.divide(init_files_nino, num_nino_yrs_choice)
                
        elif k == 1:            

            more_files_neut = files_neut.groupby('time.dayofyear').sum('time')
            more_files_nina = files_nina.groupby('time.dayofyear').sum('time')
            more_files_nino = files_nino.groupby('time.dayofyear').sum('time')
                        
            more_files_neut = np.divide(more_files_neut, num_neut_yrs_choice)
            more_files_nina = np.divide(more_files_nina, num_nina_yrs_choice)
            more_files_nino = np.divide(more_files_nino, num_nino_yrs_choice)    
            
            grp_files_neut = xr.concat([init_files_neut, more_files_neut], dim=('dayofyear'))
            grp_files_nina = xr.concat([init_files_nina, more_files_nina], dim=('dayofyear'))
            grp_files_nino = xr.concat([init_files_nino, more_files_nino], dim=('dayofyear'))                        
            
        elif k != 0 and k != 1:
                        
            more_files_neut = files_neut.groupby('time.dayofyear').sum('time')
            more_files_nina = files_nina.groupby('time.dayofyear').sum('time')
            more_files_nino = files_nino.groupby('time.dayofyear').sum('time')            
            
            more_files_neut = np.divide(more_files_neut, num_neut_yrs_choice)
            more_files_nina = np.divide(more_files_nina, num_nina_yrs_choice)
            more_files_nino = np.divide(more_files_nino, num_nino_yrs_choice)    
 
            wut1 = more_files_neut[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
            concata_stack = np.vstack([more_files_neut[:-2,:,:].values,wut1.values])
            more_files_concata = xr.Dataset({'grid': (['dayofyear','y','x'], concata_stack)},
                                             coords={'dayofyear': more_files_neut.coords['dayofyear'].values[:-1]})   
            
            wut3 = more_files_nina[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
            concat_a_stack = np.vstack([more_files_nina[:-2,:,:].values,wut3.values])
            more_files_concat_a = xr.Dataset({'grid': (['dayofyear','y','x'], concat_a_stack)},
                                             coords={'dayofyear': more_files_nina.coords['dayofyear'].values[:-1]}) 

            wut4 = more_files_nino[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
            concatao_stack = np.vstack([more_files_nino[:-2,:,:].values,wut4.values])
            more_files_concatao = xr.Dataset({'grid': (['dayofyear','y','x'], concatao_stack)},
                                             coords={'dayofyear': more_files_nino.coords['dayofyear'].values[:-1]})             
           
            grp_files_neut = xr.concat([grp_files_neut, more_files_concata.grid], dim=('dayofyear'))
            grp_files_nina = xr.concat([grp_files_nina, more_files_concat_a.grid], dim=('dayofyear'))
            grp_files_nino = xr.concat([grp_files_nino, more_files_concatao.grid], dim=('dayofyear')) 
        
    sum_files_neut = xr.concat([grp_files_neut,grp_files_neut,grp_files_neut], dim='dayofyear')
    sum_files_nina = xr.concat([grp_files_nina,grp_files_nina,grp_files_nina], dim='dayofyear')
    sum_files_nino = xr.concat([grp_files_nino,grp_files_nino,grp_files_nino], dim='dayofyear')
       
    for c, v in product(xrange(len(sum_files_neut[0,:,0])),xrange(len(sum_files_neut[0,0,:]))):
        
        sum_files_neut[:,c,v] = gfilt(sum_files_neut[:,c,v]*1.0,sigma=15.0)
        sum_files_nina[:,c,v] = gfilt(sum_files_nina[:,c,v]*1.0,sigma=15.0)
        sum_files_nino[:,c,v] = gfilt(sum_files_nino[:,c,v]*1.0,sigma=15.0)
           
    sliced_neut = sum_files_neut[len(grp_files_neut[:,0,0]):len(grp_files_neut[:,0,0])*2,:,:]
    sliced_nina = sum_files_nina[len(grp_files_nina[:,0,0]):len(grp_files_nina[:,0,0])*2,:,:]
    sliced_nino = sum_files_nino[len(grp_files_nino[:,0,0]):len(grp_files_nino[:,0,0])*2,:,:]
                
    sliced_neut = np.divide(sliced_neut,np.sum(sliced_neut, axis=0))
    sliced_nina = np.divide(sliced_nina,np.sum(sliced_nina, axis=0))
    sliced_nino = np.divide(sliced_nino,np.sum(sliced_nino, axis=0))
        
    np.save('/storage/timme1mj/maria_pysplit/torobs_sig/sliced_neut_'+str(_+que_num), sliced_neut)
    np.save('/storage/timme1mj/maria_pysplit/torobs_sig/sliced_nina_'+str(_+que_num), sliced_nina)
    np.save('/storage/timme1mj/maria_pysplit/torobs_sig/sliced_nino_'+str(_+que_num), sliced_nino)
    
    print str(_)+' completed...'
    



###############################################################################
###############################################################################
###############################################################################        
###############################################################################
###############################################################################
###############################################################################
        
        
