#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:20:01 2018

Maria J. Molina

"""

##############################################################################
##############################################################################
##############################################################################

import xarray as xr
import numpy as np

##############################################################################
##############################################################################
##############################################################################

          
data = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)


data_cut = data['stp']

da = np.asarray(data_cut)

mask = ~np.isnan(da)
mask[mask] &= da[mask] >= 1
da[mask] = 1
            
mask = ~np.isnan(da)
mask[mask] &= da[mask] < 1
da[mask] = 0
    
             
if np.nanmin(da) == 0 and np.nanmax(da) == 1:
    print 'Success!'
    

data_stphrs = xr.Dataset({'grid':(['time','y','x'],da)}, 
                          coords={'time':data_cut['time'].values},
                          attrs={'File Created By':'Maria J. Molina, Central Michigan University',
                                 'File Contents':'STP hours'})
                                           
  
data_stphrs.to_netcdf(path='/storage/timme1mj/maria_pysplit/stphrs_ensoannualpaper')


##############################################################################
##############################################################################
##############################################################################




##############################################################################
##############################################################################
##############################################################################


import mm_pkg as pk
import pandas as pd
import numpy as np
import xarray as xr



month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str) 


  
data = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)


neut_comp = {}


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


    neut_comp[k] = da_group
         
    
neut_co = {}
   
for k in xrange(12):
    
    da = neut_comp[k].values

    mask = ~np.isnan(da)
    mask[mask] &= da[mask] >= 1
    da[mask] = 1
            
    mask = ~np.isnan(da)
    mask[mask] &= da[mask] < 1
    da[mask] = 0
    
    data_stphrs = xr.Dataset({'grid':(['time','y','x'],da)}, 
                              coords={'time':neut_comp[k]['time'].values})
    
    neut_co[k] = data_stphrs


for k in xrange(12):
    
    num_yrs_neut = len(neut_co[k].grid.groupby('time.year').sum('time').year.values)

    if k == 0:
        
        sum_tors_neut = neut_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)

    elif k == 1:
        
        sum_tors_neut_2 = neut_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
      
    else:

        sum_tors_neut_2 = neut_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
       
        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2], dim=('dayofyear'))  
        
        

sum_group_1 = sum_group_neut[(31+28+32):(31+28+34),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_2 = sum_group_neut[(31+28+32+31):(31+28+32+33),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_3 = sum_group_neut[(31+28+32+31+32):(31+28+32+31+34),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_4 = sum_group_neut[(31+28+32+31+32+31):(31+28+32+31+32+33),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_5 = sum_group_neut[(31+28+32+31+32+31+32):(31+28+32+31+32+31+34),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_6 = sum_group_neut[(31+28+32+31+32+31+32+32):(31+28+32+31+32+31+32+34),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_7 = sum_group_neut[(31+28+32+31+32+31+32+32+31):(31+28+32+31+32+31+32+32+33),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_8 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32):(31+28+32+31+32+31+32+32+31+34),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)
sum_group_9 = sum_group_neut[(31+28+32+31+32+31+32+32+31+32+31):(31+28+32+31+32+31+32+32+31+32+33),:,:].sum('dayofyear', skipna=True).expand_dims('time',axis=0)


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
                                           
  
neutral_concats.to_netcdf(path='/storage/timme1mj/maria_pysplit/stphrs_neutral')


    
###############################################################################
###############################################################################
###############################################################################



##############################################################################
##############################################################################
##############################################################################


import mm_pkg as pk
import pandas as pd
import numpy as np
import xarray as xr



month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str) 


  
data = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)


nino_comp = {}



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
    
            

    for d, n in enumerate(alltor.el_nino_yrs):
        
        if d == 0:
            
            na = data['stp'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))      

        elif d == 1:

            na2 = data['stp'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na, na2], dim=('time'))
            
        else:
            
            na2 = data['stp'].sel(time=slice(pd.to_datetime(str(n)+'-'+str(i)+'-01'),pd.to_datetime(str(n)+'-'+str(i)+'-'+str(j))))
            na_group = xr.concat([na_group, na2], dim=('time')) 
            

    nino_comp[k] = na_group
   

nino_co = {}
    
for k in xrange(12):

    da = nino_comp[k].values

    mask = ~np.isnan(da)
    mask[mask] &= da[mask] >= 1
    da[mask] = 1
            
    mask = ~np.isnan(da)
    mask[mask] &= da[mask] < 1
    da[mask] = 0
    
    data_stphrs = xr.Dataset({'grid':(['time','y','x'],da)}, 
                              coords={'time':nino_comp[k]['time'].values})
    
    nino_co[k] = data_stphrs
    
    
for k in xrange(12):
    
    num_yrs_nino = len(nino_co[k].grid.groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nino = nino_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
    
    elif k == 1:
        
        sum_tors_nino_2 = nino_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
      
        sum_group_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
     
    else:
          
        sum_tors_nino_2 = nino_co[k].grid.groupby('time.dayofyear').sum('time')
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
    
  
elninos_concats.to_netcdf(path='/storage/timme1mj/maria_pysplit/stphrs_nino')

  
    
###############################################################################
###############################################################################
###############################################################################
    
    
##############################################################################
##############################################################################
##############################################################################


import mm_pkg as pk
import pandas as pd
import numpy as np
import xarray as xr


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str) 

  
data = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)



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
              
        
    for f, m in enumerate(alltor.la_nina_yrs):
    
        if f == 0:
        
            la = data['stp'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))       
    
        elif f == 1:
            
            la2 = data['stp'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))            
            la_group = xr.concat([la, la2], dim=('time'))
            
        else:
            
            la2 = data['stp'].sel(time=slice(pd.to_datetime(str(m)+'-'+str(i)+'-01'),pd.to_datetime(str(m)+'-'+str(i)+'-'+str(j))))
            la_group = xr.concat([la_group, la2], dim=('time'))             


    nina_comp[k] = la_group      

     
nina_co = {}
    
for k in xrange(12):

    da = nina_comp[k].values

    mask = ~np.isnan(da)
    mask[mask] &= da[mask] >= 1
    da[mask] = 1
            
    mask = ~np.isnan(da)
    mask[mask] &= da[mask] < 1
    da[mask] = 0
    
    data_stphrs = xr.Dataset({'grid':(['time','y','x'],da)}, 
                              coords={'time':nina_comp[k]['time'].values})
    
    nina_co[k] = data_stphrs
    
    
for k in xrange(12):

    num_yrs_nina = len(nina_co[k].grid.groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_nina = nina_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_nina_2 = nina_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:
    
        sum_tors_nina_2 = nina_co[k].grid.groupby('time.dayofyear').sum('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2], dim=('dayofyear'))  
        
        
 
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


laninas_concats.to_netcdf(path='/storage/timme1mj/maria_pysplit/stphrs_nina')

    
    
###############################################################################
###############################################################################
###############################################################################
