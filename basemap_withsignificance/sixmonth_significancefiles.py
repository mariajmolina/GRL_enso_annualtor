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
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gfilt
from mpl_toolkits.basemap import Basemap
from itertools import product
import pickle


###############################################################################
###############################################################################
###############################################################################


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


#neut_diff_concata = np.zeros([10000,51,66])
#neut_diff_concato = np.zeros([10000,51,66])
#nina_diff_concat_a = np.zeros([10000,51,66])
#nino_diff_concatao = np.zeros([10000,51,66])

neut_diff_concata_1 = np.zeros([500,51,66])
neut_diff_concato_1 = np.zeros([500,51,66])
nina_diff_concat_a_1 = np.zeros([500,51,66])
nino_diff_concatao_1 = np.zeros([500,51,66])

neut_diff_concata_2 = np.zeros([500,51,66])
neut_diff_concato_2 = np.zeros([500,51,66])
nina_diff_concat_a_2 = np.zeros([500,51,66])
nino_diff_concatao_2 = np.zeros([500,51,66])

#nina_diff = np.zeros([10000,51,66])
#nino_diff = np.zeros([10000,51,66])

for _ in xrange(500):

    for k in xrange(12):

        neut_yrs_choice = neut_comp[k].groupby('time.year').sum('time').year.values
        nina_yrs_choice = nina_comp[k].groupby('time.year').sum('time').year.values
        nino_yrs_choice = nino_comp[k].groupby('time.year').sum('time').year.values
        
        #nina_joint_yrs_choice = np.hstack([nina_yrs_choice, neut_yrs_choice])
        #nino_joint_yrs_choice = np.hstack([nino_yrs_choice, neut_yrs_choice])

        nina_joint_yrs_choice = np.hstack([nina_yrs_choice, neut_yrs_choice, nino_yrs_choice])
        nino_joint_yrs_choice = np.hstack([nino_yrs_choice, neut_yrs_choice, nina_yrs_choice])
        
        num_neut_yrs_choice = len(neut_comp[k].groupby('time.year').sum('time').year.values)      
        num_nina_yrs_choice = len(nina_comp[k].groupby('time.year').sum('time').year.values)
        num_nino_yrs_choice = len(nino_comp[k].groupby('time.year').sum('time').year.values)     
        
        num_ninaneut_yrs_choice = num_nina_yrs_choice + num_neut_yrs_choice
        num_ninoneut_yrs_choice = num_nino_yrs_choice + num_neut_yrs_choice
        
        rand_neutnina = np.array([nina_joint_yrs_choice[np.random.choice(num_ninaneut_yrs_choice)] for i in xrange(num_neut_yrs_choice)])        
        rand_neutnino = np.array([nino_joint_yrs_choice[np.random.choice(num_ninoneut_yrs_choice)] for i in xrange(num_neut_yrs_choice)])
        
        rand_nina = np.array([nina_joint_yrs_choice[np.random.choice(num_ninaneut_yrs_choice)] for i in xrange(num_nina_yrs_choice)])
        rand_nino = np.array([nino_joint_yrs_choice[np.random.choice(num_ninoneut_yrs_choice)] for i in xrange(num_nino_yrs_choice)])  
        
        ninaneut_concat = xr.concat([nina_comp[k], neut_comp[k]], dim='time')
        ninoneut_concat = xr.concat([nino_comp[k], neut_comp[k]], dim='time')
        
        ninaneut_neutnina = {}
        ninoneut_neutnino = {}
        ninaneut_nina = {}
        ninoneut_nino = {}
        
        for e, (i, j) in enumerate(zip(rand_neutnina, rand_neutnino)):
            
            ninaneut_neutnina[e] = ninaneut_concat.sel(time=str(i))
            ninoneut_neutnino[e] = ninoneut_concat.sel(time=str(j))   
            
            if e == 0:
                
                files_1a = ninaneut_neutnina[e]
                files_1o = ninoneut_neutnino[e]
                
            elif e == 1:
                
                files_2a = ninaneut_neutnina[e]
                files_concata = xr.concat([files_1a,files_2a], dim='time')
                
                files_2o = ninoneut_neutnino[e]
                files_concato = xr.concat([files_1o,files_2o], dim='time')
                
            elif e != 0 and e != 1:
                
                files_2a = ninaneut_neutnina[e]
                files_concata = xr.concat([files_concata,files_2a], dim='time')
                
                files_2o = ninoneut_neutnino[e]
                files_concato = xr.concat([files_concato,files_2o], dim='time')
            
        for e, r in enumerate(rand_nina):        
            
            ninaneut_nina[e] = ninaneut_concat.sel(time=str(r))

            if e == 0:
                
                files_1a = ninaneut_nina[e]
              
            elif e == 1:
                
                files_2a = ninaneut_nina[e]
                files_concat_a = xr.concat([files_1a,files_2a], dim='time')

            elif e != 0 and e != 1:
                
                files_2a = ninaneut_nina[e]
                files_concat_a = xr.concat([files_concat_a,files_2a], dim='time')          
            
        for e, l in enumerate(rand_nino):        
            
            ninoneut_nino[e] = ninoneut_concat.sel(time=str(l))

            if e == 0:
                
                files_1a = ninoneut_nino[e]
              
            elif e == 1:
                
                files_2a = ninoneut_nino[e]
                files_concatao = xr.concat([files_1a,files_2a], dim='time')

            elif e != 0 and e != 1:
                
                files_2a = ninoneut_nino[e]
                files_concatao = xr.concat([files_concatao,files_2a], dim='time')                          

        if k == 0:

            init_files_concata = files_concata.groupby('time.dayofyear').sum('time')
            init_files_concato = files_concato.groupby('time.dayofyear').sum('time')
            init_files_concat_a = files_concat_a.groupby('time.dayofyear').sum('time')
            init_files_concatao = files_concatao.groupby('time.dayofyear').sum('time')
                        
            init_files_concata = np.divide(init_files_concata, num_neut_yrs_choice)
            init_files_concato = np.divide(init_files_concato, num_neut_yrs_choice)
            init_files_concat_a = np.divide(init_files_concat_a, num_nina_yrs_choice)
            init_files_concatao = np.divide(init_files_concatao, num_nino_yrs_choice)
                
        elif k == 1:            

            more_files_concata = files_concata.groupby('time.dayofyear').sum('time')
            more_files_concato = files_concato.groupby('time.dayofyear').sum('time')
            more_files_concat_a = files_concat_a.groupby('time.dayofyear').sum('time')
            more_files_concatao = files_concatao.groupby('time.dayofyear').sum('time')
                        
            more_files_concata = np.divide(more_files_concata, num_neut_yrs_choice)
            more_files_concato = np.divide(more_files_concato, num_neut_yrs_choice)
            more_files_concat_a = np.divide(more_files_concat_a, num_nina_yrs_choice)
            more_files_concatao = np.divide(more_files_concatao, num_nino_yrs_choice)    
            
            grp_files_concata = xr.concat([init_files_concata, more_files_concata], dim=('dayofyear'))
            grp_files_concato = xr.concat([init_files_concato, more_files_concato], dim=('dayofyear'))
            grp_files_concat_a = xr.concat([init_files_concat_a, more_files_concat_a], dim=('dayofyear'))
            grp_files_concatao = xr.concat([init_files_concatao, more_files_concatao], dim=('dayofyear'))                        
            
        elif k != 0 and k != 1:
                        
            more_files_concata = files_concata.groupby('time.dayofyear').sum('time')
            more_files_concato = files_concato.groupby('time.dayofyear').sum('time')
            more_files_concat_a = files_concat_a.groupby('time.dayofyear').sum('time')
            more_files_concatao = files_concatao.groupby('time.dayofyear').sum('time')            
            
            more_files_concata = np.divide(more_files_concata, num_neut_yrs_choice)
            more_files_concato = np.divide(more_files_concato, num_neut_yrs_choice)
            more_files_concat_a = np.divide(more_files_concat_a, num_nina_yrs_choice)
            more_files_concatao = np.divide(more_files_concatao, num_nino_yrs_choice)    
 
            wut1 = more_files_concata[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
            concata_stack = np.vstack([more_files_concata[:-2,:,:].values,wut1.values])
            more_files_concata = xr.Dataset({'grid': (['dayofyear','y','x'], concata_stack)},
                                             coords={'dayofyear': more_files_concata.coords['dayofyear'].values[:-1]})   

            wut2 = more_files_concato[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
            concato_stack = np.vstack([more_files_concato[:-2,:,:].values,wut2.values])
            more_files_concato = xr.Dataset({'grid': (['dayofyear','y','x'], concato_stack)},
                                             coords={'dayofyear': more_files_concato.coords['dayofyear'].values[:-1]}) 
            
            wut3 = more_files_concat_a[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
            concat_a_stack = np.vstack([more_files_concat_a[:-2,:,:].values,wut3.values])
            more_files_concat_a = xr.Dataset({'grid': (['dayofyear','y','x'], concat_a_stack)},
                                             coords={'dayofyear': more_files_concat_a.coords['dayofyear'].values[:-1]}) 

            wut4 = more_files_concatao[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
            concatao_stack = np.vstack([more_files_concatao[:-2,:,:].values,wut4.values])
            more_files_concatao = xr.Dataset({'grid': (['dayofyear','y','x'], concatao_stack)},
                                             coords={'dayofyear': more_files_concatao.coords['dayofyear'].values[:-1]})             
           
            grp_files_concata = xr.concat([grp_files_concata, more_files_concata.grid], dim=('dayofyear'))
            grp_files_concato = xr.concat([grp_files_concato, more_files_concato.grid], dim=('dayofyear'))
            grp_files_concat_a = xr.concat([grp_files_concat_a, more_files_concat_a.grid], dim=('dayofyear'))
            grp_files_concatao = xr.concat([grp_files_concatao, more_files_concatao.grid], dim=('dayofyear')) 
        
    sum_files_concata = xr.concat([grp_files_concata,grp_files_concata,grp_files_concata], dim='dayofyear')
    sum_files_concato = xr.concat([grp_files_concato,grp_files_concato,grp_files_concato], dim='dayofyear')
    sum_files_concat_a = xr.concat([grp_files_concat_a,grp_files_concat_a,grp_files_concat_a], dim='dayofyear')
    sum_files_concatao = xr.concat([grp_files_concatao,grp_files_concatao,grp_files_concatao], dim='dayofyear')
       
    for c, v in product(xrange(len(sum_files_concata[0,:,0])),xrange(len(sum_files_concata[0,0,:]))):
        
        sum_files_concata[:,c,v] = gfilt(sum_files_concata[:,c,v]*1.0,sigma=15.0)
        sum_files_concato[:,c,v] = gfilt(sum_files_concato[:,c,v]*1.0,sigma=15.0)
        sum_files_concat_a[:,c,v] = gfilt(sum_files_concat_a[:,c,v]*1.0,sigma=15.0)
        sum_files_concatao[:,c,v] = gfilt(sum_files_concatao[:,c,v]*1.0,sigma=15.0)
           
        
    sliced_concata_1 = sum_files_concata[(len(grp_files_concata[:,0,0])+32):(len(grp_files_concata[:,0,0])+214),:,:]
    sliced_concato_1 = sum_files_concato[(len(grp_files_concato[:,0,0])+32):(len(grp_files_concato[:,0,0])+214),:,:]
    sliced_concat_a_1 = sum_files_concat_a[(len(grp_files_concat_a[:,0,0])+32):(len(grp_files_concat_a[:,0,0])+214),:,:]
    sliced_concatao_1 = sum_files_concatao[(len(grp_files_concatao[:,0,0])+32):(len(grp_files_concatao[:,0,0])+214),:,:]

    sliced_concata_2 = sum_files_concata[(549+31):((len(grp_files_concata[:,0,0])*2)+31),:,:] 
    sliced_concato_2 = sum_files_concato[(549+31):((len(grp_files_concato[:,0,0])*2)+31),:,:]
    sliced_concat_a_2 = sum_files_concat_a[(549+31):((len(grp_files_concat_a[:,0,0])*2)+31),:,:]
    sliced_concatao_2 = sum_files_concatao[(549+31):((len(grp_files_concatao[:,0,0])*2)+31),:,:]
    
                
    sliced_concata_1 = np.divide(sliced_concata_1,np.sum(sliced_concata_1, axis=0))
    sliced_concato_1 = np.divide(sliced_concato_1,np.sum(sliced_concato_1, axis=0))
    sliced_concat_a_1 = np.divide(sliced_concat_a_1,np.sum(sliced_concat_a_1, axis=0))
    sliced_concatao_1 = np.divide(sliced_concatao_1,np.sum(sliced_concatao_1, axis=0))

    sliced_concata_2 = np.divide(sliced_concata_2,np.sum(sliced_concata_2, axis=0))
    sliced_concato_2 = np.divide(sliced_concato_2,np.sum(sliced_concato_2, axis=0))
    sliced_concat_a_2 = np.divide(sliced_concat_a_2,np.sum(sliced_concat_a_2, axis=0))
    sliced_concatao_2 = np.divide(sliced_concatao_2,np.sum(sliced_concatao_2, axis=0))
        
    gauss_concata_1 = np.ndarray.argmax(sliced_concata_1.values,axis=0)
    gauss_concato_1 = np.ndarray.argmax(sliced_concato_1.values,axis=0)
    gauss_concat_a_1 = np.ndarray.argmax(sliced_concat_a_1.values,axis=0)
    gauss_concatao_1 = np.ndarray.argmax(sliced_concatao_1.values,axis=0)

    gauss_concata_2 = np.ndarray.argmax(sliced_concata_2.values,axis=0)
    gauss_concato_2 = np.ndarray.argmax(sliced_concato_2.values,axis=0)
    gauss_concat_a_2 = np.ndarray.argmax(sliced_concat_a_2.values,axis=0)
    gauss_concatao_2 = np.ndarray.argmax(sliced_concatao_2.values,axis=0)


    neut_diff_concata_1[_,:,:] = gauss_concata_1
    neut_diff_concato_1[_,:,:] = gauss_concato_1
    nina_diff_concat_a_1[_,:,:] = gauss_concat_a_1
    nino_diff_concatao_1[_,:,:] = gauss_concatao_1

    neut_diff_concata_2[_,:,:] = gauss_concata_2
    neut_diff_concato_2[_,:,:] = gauss_concato_2
    nina_diff_concat_a_2[_,:,:] = gauss_concat_a_2
    nino_diff_concatao_2[_,:,:] = gauss_concatao_2


    #nina_diff[_,:,:] = gauss_concat_a - gauss_concata
    #nino_diff[_,:,:] = gauss_concatao - gauss_concato
    
    print str(_)+' completed...'
    
 
#neut_diff_concata = neut_diff_concata[:,:,:]
#neut_diff_concato = neut_diff_concato[:,:,:]
#nina_diff_concat_a = nina_diff_concat_a[:,:,:]
#nino_diff_concatao = nino_diff_concatao[:,:,:]
    

#neut_diff_concata = neut_diff_concata[0:500,:,:]
#neut_diff_concato = neut_diff_concato[0:500,:,:]
#nina_diff_concat_a = nina_diff_concat_a[0:500,:,:]
#nino_diff_concatao = nino_diff_concatao[0:500,:,:]


with open('neut_diffs_concata_1_'+str(i), 'wb') as output:
    pickle.dump(neut_diff_concata_1, output, pickle.HIGHEST_PROTOCOL)

with open('neut_diffs_concato_1_'+str(i), 'wb') as output:
    pickle.dump(neut_diff_concato_1, output, pickle.HIGHEST_PROTOCOL)

with open('nina_diffs_concat_a_1_'+str(i), 'wb') as output:
    pickle.dump(nina_diff_concat_a_1, output, pickle.HIGHEST_PROTOCOL)

with open('nino_diffs_concatao_1_'+str(i), 'wb') as output:
    pickle.dump(nino_diff_concatao_1, output, pickle.HIGHEST_PROTOCOL)
    
    
with open('neut_diffs_concata_2_'+str(i), 'wb') as output:
    pickle.dump(neut_diff_concata_2, output, pickle.HIGHEST_PROTOCOL)

with open('neut_diffs_concato_2_'+str(i), 'wb') as output:
    pickle.dump(neut_diff_concato_2, output, pickle.HIGHEST_PROTOCOL)

with open('nina_diffs_concat_a_2_'+str(i), 'wb') as output:
    pickle.dump(nina_diff_concat_a_2, output, pickle.HIGHEST_PROTOCOL)

with open('nino_diffs_concatao_2_'+str(i), 'wb') as output:
    pickle.dump(nino_diff_concatao_2, output, pickle.HIGHEST_PROTOCOL)


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


###############################################################################
############first six months###################################################
###############################################################################


with open('neut_diffs_concata_1_1', 'rb') as f:
     neut_diff_concata_1 = pickle.load(f)

with open('neut_diffs_concato_1_1', 'rb') as f:
     neut_diff_concato_1 = pickle.load(f)

with open('nina_diffs_concat_a_1_1', 'rb') as f:
     nina_diff_concat_a_1 = pickle.load(f)

with open('nino_diffs_concatao_1_1', 'rb') as f:
     nino_diff_concatao_1 = pickle.load(f)
     
with open('neut_diffs_concata_1_2', 'rb') as f:
     neut_diff_concata_2 = pickle.load(f)

with open('neut_diffs_concato_1_2', 'rb') as f:
     neut_diff_concato_2 = pickle.load(f)

with open('nina_diffs_concat_a_1_2', 'rb') as f:
     nina_diff_concat_a_2 = pickle.load(f)

with open('nino_diffs_concatao_1_2', 'rb') as f:
     nino_diff_concatao_2 = pickle.load(f)

with open('neut_diffs_concata_1_3', 'rb') as f:
     neut_diff_concata_3 = pickle.load(f)

with open('neut_diffs_concato_1_3', 'rb') as f:
     neut_diff_concato_3 = pickle.load(f)

with open('nina_diffs_concat_a_1_3', 'rb') as f:
     nina_diff_concat_a_3 = pickle.load(f)

with open('nino_diffs_concatao_1_3', 'rb') as f:
     nino_diff_concatao_3 = pickle.load(f)

with open('neut_diffs_concata_1_4', 'rb') as f:
     neut_diff_concata_4 = pickle.load(f)

with open('neut_diffs_concato_1_4', 'rb') as f:
     neut_diff_concato_4 = pickle.load(f)

with open('nina_diffs_concat_a_1_4', 'rb') as f:
     nina_diff_concat_a_4 = pickle.load(f)

with open('nino_diffs_concatao_1_4', 'rb') as f:
     nino_diff_concatao_4 = pickle.load(f)

with open('neut_diffs_concata_1_5', 'rb') as f:
     neut_diff_concata_5 = pickle.load(f)

with open('neut_diffs_concato_1_5', 'rb') as f:
     neut_diff_concato_5 = pickle.load(f)

with open('nina_diffs_concat_a_1_5', 'rb') as f:
     nina_diff_concat_a_5 = pickle.load(f)

with open('nino_diffs_concatao_1_5', 'rb') as f:
     nino_diff_concatao_5 = pickle.load(f)

with open('neut_diffs_concata_1_6', 'rb') as f:
     neut_diff_concata_6 = pickle.load(f)

with open('neut_diffs_concato_1_6', 'rb') as f:
     neut_diff_concato_6 = pickle.load(f)

with open('nina_diffs_concat_a_1_6', 'rb') as f:
     nina_diff_concat_a_6 = pickle.load(f)

with open('nino_diffs_concatao_1_6', 'rb') as f:
     nino_diff_concatao_6 = pickle.load(f)

with open('neut_diffs_concata_1_7', 'rb') as f:
     neut_diff_concata_7 = pickle.load(f)

with open('neut_diffs_concato_1_7', 'rb') as f:
     neut_diff_concato_7 = pickle.load(f)

with open('nina_diffs_concat_a_1_7', 'rb') as f:
     nina_diff_concat_a_7 = pickle.load(f)

with open('nino_diffs_concatao_1_7', 'rb') as f:
     nino_diff_concatao_7 = pickle.load(f)

with open('neut_diffs_concata_1_8', 'rb') as f:
     neut_diff_concata_8 = pickle.load(f)

with open('neut_diffs_concato_1_8', 'rb') as f:
     neut_diff_concato_8 = pickle.load(f)

with open('nina_diffs_concat_a_1_8', 'rb') as f:
     nina_diff_concat_a_8 = pickle.load(f)

with open('nino_diffs_concatao_1_8', 'rb') as f:
     nino_diff_concatao_8 = pickle.load(f)

with open('neut_diffs_concata_1_9', 'rb') as f:
     neut_diff_concata_9 = pickle.load(f)

with open('neut_diffs_concato_1_9', 'rb') as f:
     neut_diff_concato_9 = pickle.load(f)

with open('nina_diffs_concat_a_1_9', 'rb') as f:
     nina_diff_concat_a_9 = pickle.load(f)

with open('nino_diffs_concatao_1_9', 'rb') as f:
     nino_diff_concatao_9 = pickle.load(f)

with open('neut_diffs_concata_1_10', 'rb') as f:
     neut_diff_concata_10 = pickle.load(f)

with open('neut_diffs_concato_1_10', 'rb') as f:
     neut_diff_concato_10 = pickle.load(f)

with open('nina_diffs_concat_a_1_10', 'rb') as f:
     nina_diff_concat_a_10 = pickle.load(f)

with open('nino_diffs_concatao_1_10', 'rb') as f:
     nino_diff_concatao_10 = pickle.load(f)

with open('neut_diffs_concata_1_11', 'rb') as f:
     neut_diff_concata_11 = pickle.load(f)

with open('neut_diffs_concato_1_11', 'rb') as f:
     neut_diff_concato_11 = pickle.load(f)

with open('nina_diffs_concat_a_1_11', 'rb') as f:
     nina_diff_concat_a_11 = pickle.load(f)

with open('nino_diffs_concatao_1_11', 'rb') as f:
     nino_diff_concatao_11 = pickle.load(f)

with open('neut_diffs_concata_1_12', 'rb') as f:
     neut_diff_concata_12 = pickle.load(f)

with open('neut_diffs_concato_1_12', 'rb') as f:
     neut_diff_concato_12 = pickle.load(f)

with open('nina_diffs_concat_a_1_12', 'rb') as f:
     nina_diff_concat_a_12 = pickle.load(f)

with open('nino_diffs_concatao_1_12', 'rb') as f:
     nino_diff_concatao_12 = pickle.load(f)

with open('neut_diffs_concata_1_13', 'rb') as f:
     neut_diff_concata_13 = pickle.load(f)

with open('neut_diffs_concato_1_13', 'rb') as f:
     neut_diff_concato_13 = pickle.load(f)

with open('nina_diffs_concat_a_1_13', 'rb') as f:
     nina_diff_concat_a_13 = pickle.load(f)

with open('nino_diffs_concatao_1_13', 'rb') as f:
     nino_diff_concatao_13 = pickle.load(f)

with open('neut_diffs_concata_1_14', 'rb') as f:
     neut_diff_concata_14 = pickle.load(f)

with open('neut_diffs_concato_1_14', 'rb') as f:
     neut_diff_concato_14 = pickle.load(f)

with open('nina_diffs_concat_a_1_14', 'rb') as f:
     nina_diff_concat_a_14 = pickle.load(f)

with open('nino_diffs_concatao_1_14', 'rb') as f:
     nino_diff_concatao_14 = pickle.load(f)

with open('neut_diffs_concata_1_15', 'rb') as f:
     neut_diff_concata_15 = pickle.load(f)

with open('neut_diffs_concato_1_15', 'rb') as f:
     neut_diff_concato_15 = pickle.load(f)

with open('nina_diffs_concat_a_1_15', 'rb') as f:
     nina_diff_concat_a_15 = pickle.load(f)

with open('nino_diffs_concatao_1_15', 'rb') as f:
     nino_diff_concatao_15 = pickle.load(f)

with open('neut_diffs_concata_1_16', 'rb') as f:
     neut_diff_concata_16 = pickle.load(f)

with open('neut_diffs_concato_1_16', 'rb') as f:
     neut_diff_concato_16 = pickle.load(f)

with open('nina_diffs_concat_a_1_16', 'rb') as f:
     nina_diff_concat_a_16 = pickle.load(f)

with open('nino_diffs_concatao_1_16', 'rb') as f:
     nino_diff_concatao_16 = pickle.load(f)

with open('neut_diffs_concata_1_17', 'rb') as f:
     neut_diff_concata_17 = pickle.load(f)

with open('neut_diffs_concato_1_17', 'rb') as f:
     neut_diff_concato_17 = pickle.load(f)

with open('nina_diffs_concat_a_1_17', 'rb') as f:
     nina_diff_concat_a_17 = pickle.load(f)

with open('nino_diffs_concatao_1_17', 'rb') as f:
     nino_diff_concatao_17 = pickle.load(f)

with open('neut_diffs_concata_1_18', 'rb') as f:
     neut_diff_concata_18 = pickle.load(f)

with open('neut_diffs_concato_1_18', 'rb') as f:
     neut_diff_concato_18 = pickle.load(f)

with open('nina_diffs_concat_a_1_18', 'rb') as f:
     nina_diff_concat_a_18 = pickle.load(f)

with open('nino_diffs_concatao_1_18', 'rb') as f:
     nino_diff_concatao_18 = pickle.load(f)

with open('neut_diffs_concata_1_19', 'rb') as f:
     neut_diff_concata_19 = pickle.load(f)

with open('neut_diffs_concato_1_19', 'rb') as f:
     neut_diff_concato_19 = pickle.load(f)

with open('nina_diffs_concat_a_1_19', 'rb') as f:
     nina_diff_concat_a_19 = pickle.load(f)

with open('nino_diffs_concatao_1_19', 'rb') as f:
     nino_diff_concatao_19 = pickle.load(f)

with open('neut_diffs_concata_1_20', 'rb') as f:
     neut_diff_concata_20 = pickle.load(f)

with open('neut_diffs_concato_1_20', 'rb') as f:
     neut_diff_concato_20 = pickle.load(f)

with open('nina_diffs_concat_a_1_20', 'rb') as f:
     nina_diff_concat_a_20 = pickle.load(f)

with open('nino_diffs_concatao_1_20', 'rb') as f:
     nino_diff_concatao_20 = pickle.load(f)
     
     
     
neut_diff_concata = np.vstack([neut_diff_concata_1,neut_diff_concata_2,neut_diff_concata_3,neut_diff_concata_4,neut_diff_concata_5,
                               neut_diff_concata_6,neut_diff_concata_7,neut_diff_concata_8,neut_diff_concata_9,neut_diff_concata_10,
                               neut_diff_concata_11,neut_diff_concata_12,neut_diff_concata_13,neut_diff_concata_14,neut_diff_concata_15,
                               neut_diff_concata_16,neut_diff_concata_17,neut_diff_concata_18,neut_diff_concata_19,neut_diff_concata_20])
    
neut_diff_concato = np.vstack([neut_diff_concato_1,neut_diff_concato_2,neut_diff_concato_3,neut_diff_concato_4,neut_diff_concato_5,
                               neut_diff_concato_6,neut_diff_concato_7,neut_diff_concato_8,neut_diff_concato_9,neut_diff_concato_10,
                               neut_diff_concato_11,neut_diff_concato_12,neut_diff_concato_13,neut_diff_concato_14,neut_diff_concato_15,
                               neut_diff_concato_16,neut_diff_concato_17,neut_diff_concato_18,neut_diff_concato_19,neut_diff_concato_20])
    
nina_diff_concat_a = np.vstack([nina_diff_concat_a_1,nina_diff_concat_a_2,nina_diff_concat_a_3,nina_diff_concat_a_4,nina_diff_concat_a_5,
                                nina_diff_concat_a_6,nina_diff_concat_a_7,nina_diff_concat_a_8,nina_diff_concat_a_9,nina_diff_concat_a_10,
                                nina_diff_concat_a_11,nina_diff_concat_a_12,nina_diff_concat_a_13,nina_diff_concat_a_14,nina_diff_concat_a_15,
                                nina_diff_concat_a_16,nina_diff_concat_a_17,nina_diff_concat_a_18,nina_diff_concat_a_19,nina_diff_concat_a_20])
    
nino_diff_concatao = np.vstack([nino_diff_concatao_1,nino_diff_concatao_2,nino_diff_concatao_3,nino_diff_concatao_4,nino_diff_concatao_5,
                                nino_diff_concatao_6,nino_diff_concatao_7,nino_diff_concatao_8,nino_diff_concatao_9,nino_diff_concatao_10,
                                nino_diff_concatao_11,nino_diff_concatao_12,nino_diff_concatao_13,nino_diff_concatao_14,nino_diff_concatao_15,
                                nino_diff_concatao_16,nino_diff_concatao_17,nino_diff_concatao_18,nino_diff_concatao_19,nino_diff_concatao_20])
     
   
    
for i in xrange(10000):
    
    neut_diff_concata[i,:,:] = gfilt(neut_diff_concata[i,:,:]*1.0, sigma=1.5)
    neut_diff_concato[i,:,:] = gfilt(neut_diff_concato[i,:,:]*1.0, sigma=1.5)
    nina_diff_concat_a[i,:,:] = gfilt(nina_diff_concat_a[i,:,:]*1.0, sigma=1.5)
    nino_diff_concatao[i,:,:] = gfilt(nino_diff_concatao[i,:,:]*1.0, sigma=1.5)
    
    print str(i)+' completed...'
    


neut_concata_upper = np.nanpercentile(neut_diff_concata,97.5,axis=0)
neut_concata_lower = np.nanpercentile(neut_diff_concata,2.5,axis=0)
neut_concato_upper = np.nanpercentile(neut_diff_concato,97.5,axis=0)
neut_concato_lower = np.nanpercentile(neut_diff_concato,2.5,axis=0)
nina_concat_a_upper = np.nanpercentile(nina_diff_concat_a,97.5,axis=0)
nina_concat_a_lower = np.nanpercentile(nina_diff_concat_a,2.5,axis=0)
nino_concatao_upper = np.nanpercentile(nino_diff_concatao,97.5,axis=0)
nino_concatao_lower = np.nanpercentile(nino_diff_concatao,2.5,axis=0)


###############################################################################
###############################################################################
###############################################################################


with open('neut_diffs_concata_2_1', 'rb') as f:
     neut_diff_concata_1 = pickle.load(f)

with open('neut_diffs_concato_2_1', 'rb') as f:
     neut_diff_concato_1 = pickle.load(f)

with open('nina_diffs_concat_a_2_1', 'rb') as f:
     nina_diff_concat_a_1 = pickle.load(f)

with open('nino_diffs_concatao_2_1', 'rb') as f:
     nino_diff_concatao_1 = pickle.load(f)
     
with open('neut_diffs_concata_2_2', 'rb') as f:
     neut_diff_concata_2 = pickle.load(f)

with open('neut_diffs_concato_2_2', 'rb') as f:
     neut_diff_concato_2 = pickle.load(f)

with open('nina_diffs_concat_a_2_2', 'rb') as f:
     nina_diff_concat_a_2 = pickle.load(f)

with open('nino_diffs_concatao_2_2', 'rb') as f:
     nino_diff_concatao_2 = pickle.load(f)

with open('neut_diffs_concata_2_3', 'rb') as f:
     neut_diff_concata_3 = pickle.load(f)

with open('neut_diffs_concato_2_3', 'rb') as f:
     neut_diff_concato_3 = pickle.load(f)

with open('nina_diffs_concat_a_2_3', 'rb') as f:
     nina_diff_concat_a_3 = pickle.load(f)

with open('nino_diffs_concatao_2_3', 'rb') as f:
     nino_diff_concatao_3 = pickle.load(f)

with open('neut_diffs_concata_2_4', 'rb') as f:
     neut_diff_concata_4 = pickle.load(f)

with open('neut_diffs_concato_2_4', 'rb') as f:
     neut_diff_concato_4 = pickle.load(f)

with open('nina_diffs_concat_a_2_4', 'rb') as f:
     nina_diff_concat_a_4 = pickle.load(f)

with open('nino_diffs_concatao_2_4', 'rb') as f:
     nino_diff_concatao_4 = pickle.load(f)

with open('neut_diffs_concata_2_5', 'rb') as f:
     neut_diff_concata_5 = pickle.load(f)

with open('neut_diffs_concato_2_5', 'rb') as f:
     neut_diff_concato_5 = pickle.load(f)

with open('nina_diffs_concat_a_2_5', 'rb') as f:
     nina_diff_concat_a_5 = pickle.load(f)

with open('nino_diffs_concatao_2_5', 'rb') as f:
     nino_diff_concatao_5 = pickle.load(f)

with open('neut_diffs_concata_2_6', 'rb') as f:
     neut_diff_concata_6 = pickle.load(f)

with open('neut_diffs_concato_2_6', 'rb') as f:
     neut_diff_concato_6 = pickle.load(f)

with open('nina_diffs_concat_a_2_6', 'rb') as f:
     nina_diff_concat_a_6 = pickle.load(f)

with open('nino_diffs_concatao_2_6', 'rb') as f:
     nino_diff_concatao_6 = pickle.load(f)

with open('neut_diffs_concata_2_7', 'rb') as f:
     neut_diff_concata_7 = pickle.load(f)

with open('neut_diffs_concato_2_7', 'rb') as f:
     neut_diff_concato_7 = pickle.load(f)

with open('nina_diffs_concat_a_2_7', 'rb') as f:
     nina_diff_concat_a_7 = pickle.load(f)

with open('nino_diffs_concatao_2_7', 'rb') as f:
     nino_diff_concatao_7 = pickle.load(f)

with open('neut_diffs_concata_2_8', 'rb') as f:
     neut_diff_concata_8 = pickle.load(f)

with open('neut_diffs_concato_2_8', 'rb') as f:
     neut_diff_concato_8 = pickle.load(f)

with open('nina_diffs_concat_a_2_8', 'rb') as f:
     nina_diff_concat_a_8 = pickle.load(f)

with open('nino_diffs_concatao_2_8', 'rb') as f:
     nino_diff_concatao_8 = pickle.load(f)

with open('neut_diffs_concata_2_9', 'rb') as f:
     neut_diff_concata_9 = pickle.load(f)

with open('neut_diffs_concato_2_9', 'rb') as f:
     neut_diff_concato_9 = pickle.load(f)

with open('nina_diffs_concat_a_2_9', 'rb') as f:
     nina_diff_concat_a_9 = pickle.load(f)

with open('nino_diffs_concatao_2_9', 'rb') as f:
     nino_diff_concatao_9 = pickle.load(f)

with open('neut_diffs_concata_2_10', 'rb') as f:
     neut_diff_concata_10 = pickle.load(f)

with open('neut_diffs_concato_2_10', 'rb') as f:
     neut_diff_concato_10 = pickle.load(f)

with open('nina_diffs_concat_a_2_10', 'rb') as f:
     nina_diff_concat_a_10 = pickle.load(f)

with open('nino_diffs_concatao_2_10', 'rb') as f:
     nino_diff_concatao_10 = pickle.load(f)

with open('neut_diffs_concata_2_11', 'rb') as f:
     neut_diff_concata_11 = pickle.load(f)

with open('neut_diffs_concato_2_11', 'rb') as f:
     neut_diff_concato_11 = pickle.load(f)

with open('nina_diffs_concat_a_2_11', 'rb') as f:
     nina_diff_concat_a_11 = pickle.load(f)

with open('nino_diffs_concatao_2_11', 'rb') as f:
     nino_diff_concatao_11 = pickle.load(f)

with open('neut_diffs_concata_2_12', 'rb') as f:
     neut_diff_concata_12 = pickle.load(f)

with open('neut_diffs_concato_2_12', 'rb') as f:
     neut_diff_concato_12 = pickle.load(f)

with open('nina_diffs_concat_a_2_12', 'rb') as f:
     nina_diff_concat_a_12 = pickle.load(f)

with open('nino_diffs_concatao_2_12', 'rb') as f:
     nino_diff_concatao_12 = pickle.load(f)

with open('neut_diffs_concata_2_13', 'rb') as f:
     neut_diff_concata_13 = pickle.load(f)

with open('neut_diffs_concato_2_13', 'rb') as f:
     neut_diff_concato_13 = pickle.load(f)

with open('nina_diffs_concat_a_2_13', 'rb') as f:
     nina_diff_concat_a_13 = pickle.load(f)

with open('nino_diffs_concatao_2_13', 'rb') as f:
     nino_diff_concatao_13 = pickle.load(f)

with open('neut_diffs_concata_2_14', 'rb') as f:
     neut_diff_concata_14 = pickle.load(f)

with open('neut_diffs_concato_2_14', 'rb') as f:
     neut_diff_concato_14 = pickle.load(f)

with open('nina_diffs_concat_a_2_14', 'rb') as f:
     nina_diff_concat_a_14 = pickle.load(f)

with open('nino_diffs_concatao_2_14', 'rb') as f:
     nino_diff_concatao_14 = pickle.load(f)

with open('neut_diffs_concata_2_15', 'rb') as f:
     neut_diff_concata_15 = pickle.load(f)

with open('neut_diffs_concato_2_15', 'rb') as f:
     neut_diff_concato_15 = pickle.load(f)

with open('nina_diffs_concat_a_2_15', 'rb') as f:
     nina_diff_concat_a_15 = pickle.load(f)

with open('nino_diffs_concatao_2_15', 'rb') as f:
     nino_diff_concatao_15 = pickle.load(f)

with open('neut_diffs_concata_2_16', 'rb') as f:
     neut_diff_concata_16 = pickle.load(f)

with open('neut_diffs_concato_2_16', 'rb') as f:
     neut_diff_concato_16 = pickle.load(f)

with open('nina_diffs_concat_a_2_16', 'rb') as f:
     nina_diff_concat_a_16 = pickle.load(f)

with open('nino_diffs_concatao_2_16', 'rb') as f:
     nino_diff_concatao_16 = pickle.load(f)

with open('neut_diffs_concata_2_17', 'rb') as f:
     neut_diff_concata_17 = pickle.load(f)

with open('neut_diffs_concato_2_17', 'rb') as f:
     neut_diff_concato_17 = pickle.load(f)

with open('nina_diffs_concat_a_2_17', 'rb') as f:
     nina_diff_concat_a_17 = pickle.load(f)

with open('nino_diffs_concatao_2_17', 'rb') as f:
     nino_diff_concatao_17 = pickle.load(f)

with open('neut_diffs_concata_2_18', 'rb') as f:
     neut_diff_concata_18 = pickle.load(f)

with open('neut_diffs_concato_2_18', 'rb') as f:
     neut_diff_concato_18 = pickle.load(f)

with open('nina_diffs_concat_a_2_18', 'rb') as f:
     nina_diff_concat_a_18 = pickle.load(f)

with open('nino_diffs_concatao_2_18', 'rb') as f:
     nino_diff_concatao_18 = pickle.load(f)

with open('neut_diffs_concata_2_19', 'rb') as f:
     neut_diff_concata_19 = pickle.load(f)

with open('neut_diffs_concato_2_19', 'rb') as f:
     neut_diff_concato_19 = pickle.load(f)

with open('nina_diffs_concat_a_2_19', 'rb') as f:
     nina_diff_concat_a_19 = pickle.load(f)

with open('nino_diffs_concatao_2_19', 'rb') as f:
     nino_diff_concatao_19 = pickle.load(f)

with open('neut_diffs_concata_2_20', 'rb') as f:
     neut_diff_concata_20 = pickle.load(f)

with open('neut_diffs_concato_2_20', 'rb') as f:
     neut_diff_concato_20 = pickle.load(f)

with open('nina_diffs_concat_a_2_20', 'rb') as f:
     nina_diff_concat_a_20 = pickle.load(f)

with open('nino_diffs_concatao_2_20', 'rb') as f:
     nino_diff_concatao_20 = pickle.load(f)

     
     
neut_diff_concata = np.vstack([neut_diff_concata_1,neut_diff_concata_2,neut_diff_concata_3,neut_diff_concata_4,neut_diff_concata_5,
                               neut_diff_concata_6,neut_diff_concata_7,neut_diff_concata_8,neut_diff_concata_9,neut_diff_concata_10,
                               neut_diff_concata_11,neut_diff_concata_12,neut_diff_concata_13,neut_diff_concata_14,neut_diff_concata_15,
                               neut_diff_concata_16,neut_diff_concata_17,neut_diff_concata_18,neut_diff_concata_19,neut_diff_concata_20])
    
neut_diff_concato = np.vstack([neut_diff_concato_1,neut_diff_concato_2,neut_diff_concato_3,neut_diff_concato_4,neut_diff_concato_5,
                               neut_diff_concato_6,neut_diff_concato_7,neut_diff_concato_8,neut_diff_concato_9,neut_diff_concato_10,
                               neut_diff_concato_11,neut_diff_concato_12,neut_diff_concato_13,neut_diff_concato_14,neut_diff_concato_15,
                               neut_diff_concato_16,neut_diff_concato_17,neut_diff_concato_18,neut_diff_concato_19,neut_diff_concato_20])
    
nina_diff_concat_a = np.vstack([nina_diff_concat_a_1,nina_diff_concat_a_2,nina_diff_concat_a_3,nina_diff_concat_a_4,nina_diff_concat_a_5,
                                nina_diff_concat_a_6,nina_diff_concat_a_7,nina_diff_concat_a_8,nina_diff_concat_a_9,nina_diff_concat_a_10,
                                nina_diff_concat_a_11,nina_diff_concat_a_12,nina_diff_concat_a_13,nina_diff_concat_a_14,nina_diff_concat_a_15,
                                nina_diff_concat_a_16,nina_diff_concat_a_17,nina_diff_concat_a_18,nina_diff_concat_a_19,nina_diff_concat_a_20])
    
nino_diff_concatao = np.vstack([nino_diff_concatao_1,nino_diff_concatao_2,nino_diff_concatao_3,nino_diff_concatao_4,nino_diff_concatao_5,
                                nino_diff_concatao_6,nino_diff_concatao_7,nino_diff_concatao_8,nino_diff_concatao_9,nino_diff_concatao_10,
                                nino_diff_concatao_11,nino_diff_concatao_12,nino_diff_concatao_13,nino_diff_concatao_14,nino_diff_concatao_15,
                                nino_diff_concatao_16,nino_diff_concatao_17,nino_diff_concatao_18,nino_diff_concatao_19,nino_diff_concatao_20])
     
   
    
for i in xrange(10000):
    
    neut_diff_concata[i,:,:] = gfilt(neut_diff_concata[i,:,:]*1.0, sigma=1.5)
    neut_diff_concato[i,:,:] = gfilt(neut_diff_concato[i,:,:]*1.0, sigma=1.5)
    nina_diff_concat_a[i,:,:] = gfilt(nina_diff_concat_a[i,:,:]*1.0, sigma=1.5)
    nino_diff_concatao[i,:,:] = gfilt(nino_diff_concatao[i,:,:]*1.0, sigma=1.5)
    
    print str(i)+' completed...'
    


neut_concata_upper = np.nanpercentile(neut_diff_concata,97.5,axis=0)
neut_concata_lower = np.nanpercentile(neut_diff_concata,2.5,axis=0)
neut_concato_upper = np.nanpercentile(neut_diff_concato,97.5,axis=0)
neut_concato_lower = np.nanpercentile(neut_diff_concato,2.5,axis=0)
nina_concat_a_upper = np.nanpercentile(nina_diff_concat_a,97.5,axis=0)
nina_concat_a_lower = np.nanpercentile(nina_diff_concat_a,2.5,axis=0)
nino_concatao_upper = np.nanpercentile(nino_diff_concatao,97.5,axis=0)
nino_concatao_lower = np.nanpercentile(nino_diff_concatao,2.5,axis=0)


###############################################################################
###############################################################################
###############################################################################
    

    
    
