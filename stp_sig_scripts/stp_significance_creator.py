#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:09:23 2018

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
    
    
neut_comp = {}
 
for k in xrange(12):

    neut_comp[k] = xr.open_dataarray('neut_co_'+str(k), decode_cf=True)

nino_comp = {}
    
for k in xrange(12):    
    
    nino_comp[k] = xr.open_dataarray('nino_co_'+str(k), decode_cf=True)

nina_comp = {}

for k in xrange(12):
    
    nina_comp[k] = xr.open_dataarray('nina_co_'+str(k), decode_cf=True)
    
    
###############################################################################
###############################################################################
###############################################################################
    
    
num_neut_yrs_choice = {}
num_nina_yrs_choice = {}
num_nino_yrs_choice = {}
    
for k in xrange(12):
    
    num_neut_yrs_choice[k] = len(neut_comp[k].groupby('time.year').sum('time').year.values)      
    num_nina_yrs_choice[k] = len(nina_comp[k].groupby('time.year').sum('time').year.values)
    num_nino_yrs_choice[k] = len(nino_comp[k].groupby('time.year').sum('time').year.values)    
    
        
###############################################################################
###############################################################################
###############################################################################

 
num_neut_yrs = [num_neut_yrs_choice.items()[i][1] for i in xrange(12)]
num_nina_yrs = [num_nina_yrs_choice.items()[i][1] for i in xrange(12)]
num_nino_yrs = [num_nino_yrs_choice.items()[i][1] for i in xrange(12)]
    

###############################################################################
###############################################################################
###############################################################################
  
  
with open('num_neut_yrs', 'wb') as output:
    pickle.dump(num_neut_yrs, output, pickle.HIGHEST_PROTOCOL)

with open('num_nina_yrs', 'wb') as output:
    pickle.dump(num_nina_yrs, output, pickle.HIGHEST_PROTOCOL)

with open('num_nino_yrs', 'wb') as output:
    pickle.dump(num_nino_yrs, output, pickle.HIGHEST_PROTOCOL)
    
    
###############################################################################
###############################################################################
###############################################################################
    

datas = xr.open_dataset('/storage/Reanalysis_Processed/stp_cin_1979_2016.nc', decode_cf=True)

data = datas.stp.resample(time='D').mean(dim='time')


for i in xrange(12):
    
    monthlies_stp = data[data['time.month']==(i+1)]
    
    monthlies_stp.to_netcdf('monthlies_stp_'+str(i+1))
        

###############################################################################
############loading of data####################################################
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


for _ in xrange(5000):

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
            
    neutnina_diff[_,:,:,:] = init_files_concata
    nina_diff[_,:,:,:] = init_files_concat_a
    nino_diff[_,:,:,:] = init_files_concatao
    
    
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+month_name+'/neutnina_month_'+month_name+'_'+str(_+1), 'wb') as output:
        pickle.dump(neutnina_diff, output, pickle.HIGHEST_PROTOCOL)
        
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+month_name+'/nina_month_'+month_name+'_'+str(_+1), 'wb') as output:
        pickle.dump(nina_diff, output, pickle.HIGHEST_PROTOCOL)
        
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+month_name+'/nino_month_'+month_name+'_'+str(_+1), 'wb') as output:
        pickle.dump(nino_diff, output, pickle.HIGHEST_PROTOCOL)
    
    print str(_)+' complete...'
    
    
    break
    
    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


neutnina_diff = np.zeros([1000,day_list,277,349])
#neutnino_diff = np.zeros([5000,day_list,277,349])
nina_diff = np.zeros([1000,day_list,277,349])
nino_diff = np.zeros([1000,day_list,277,349])

#month_data = np.reshape(monthlies_stp.values,(38,31,277,349))

for _ in xrange(1000):

    yrs_choice = pd.date_range('1979','2016',freq='AS').year
    #yrs_choice = np.arange(0,len(pd.date_range('1979','2016',freq='AS').year),1)
    
    #num_ninaneut_yrs_choice = num_nina_yrs + num_neut_yrs
    #num_ninoneut_yrs_choice = num_nino_yrs + num_neut_yrs
        
    rand_neutnina = np.array([yrs_choice[np.random.choice(len(yrs_choice))] for i in xrange(num_neut_yrs)])        
    #rand_neutnino = np.array([yrs_choice[np.random.choice(len(yrs_choice))] for i in xrange(num_neut_yrs)])
        
    rand_nina = np.array([yrs_choice[np.random.choice(len(yrs_choice))] for i in xrange(num_nina_yrs)])
    rand_nino = np.array([yrs_choice[np.random.choice(len(yrs_choice))] for i in xrange(num_nino_yrs)])  

    #print 'rand_steps complete'
    
    #ninaneut_stack = month_data[rand_neutnina,:,:,:]
    #ninoneut_stack = month_data[rand_neutnino,:,:,:]
    #ninaneut_nina_stack = month_data[rand_nina,:,:,:]
    #ninoneut_nino_stack = month_data[rand_nino,:,:,:]

    #init_files_concata = np.nanmean(ninaneut_stack, axis=0)
    #init_files_concato = np.nanmean(ninoneut_stack, axis=0)
    #init_files_concat_a = np.nanmean(ninaneut_nina_stack, axis=0)
    #init_files_concatao = np.nanmean(ninoneut_nino_stack, axis=0)
                        
    #init_files_concata = np.divide(init_files_concata, num_neut_yrs)
    #init_files_concato = np.divide(init_files_concato, num_neut_yrs)
    #init_files_concat_a = np.divide(init_files_concat_a, num_nina_yrs)
    #init_files_concatao = np.divide(init_files_concatao, num_nino_yrs)
            
    #neutnina_diff[_,:,:,:] = init_files_concata[:day_list,:,:]
    #neutnino_diff[_,:,:,:] = init_files_concato[:day_list,:,:]
    #nina_diff[_,:,:,:] = init_files_concat_a[:day_list,:,:]
    #nino_diff[_,:,:,:] = init_files_concatao[:day_list,:,:]
    
    #print str(_)+' complete...'


    for e, i in enumerate(rand_neutnina):
        
        if e == 0:
        
            ninaneut_neutnina_1 = monthlies_stp.sel(time=str(i)).expand_dims(dim='year_add').values[:,:day_list,:,:]
            
        elif e == 1:

            ninaneut_neutnina_2 = monthlies_stp.sel(time=str(i)).expand_dims(dim='year_add').values[:,:day_list,:,:]       
            ninaneut_stack = np.vstack([ninaneut_neutnina_1,ninaneut_neutnina_2])
                      
        else:
        
            ninaneut_neutnina_2 = monthlies_stp.sel(time=str(i)).expand_dims(dim='year_add').values[:,:day_list,:,:]           
            ninaneut_stack = np.vstack([ninaneut_neutnina_2,ninaneut_stack])


#    for e, j in enumerate(rand_neutnino):
        
#        if e == 0:
        
#            ninoneut_neutnino_1 = monthlies_stp.sel(time=str(j)).expand_dims(dim='year_add').values
            
#        elif e == 1:

#            ninoneut_neutnino_2 = monthlies_stp.sel(time=str(j)).expand_dims(dim='year_add').values            
#            ninoneut_stack = np.vstack([ninoneut_neutnino_1,ninoneut_neutnino_2])
                      
#        else:
        
#            ninoneut_neutnino_2 = monthlies_stp.sel(time=str(j)).expand_dims(dim='year_add').values            
#            ninoneut_stack = np.vstack([ninoneut_neutnino_2,ninoneut_stack])
 
     
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

    #print 'years selecting complete'

    init_files_concata = np.nanmean(ninaneut_stack, axis=0)
    #init_files_concato = np.nanmean(ninoneut_stack, axis=0)
    init_files_concat_a = np.nanmean(ninaneut_nina_stack, axis=0)
    init_files_concatao = np.nanmean(ninoneut_nino_stack, axis=0)
                        
    init_files_concata = np.divide(init_files_concata, num_neut_yrs)
    #init_files_concato = np.divide(init_files_concato, num_neut_yrs)
    init_files_concat_a = np.divide(init_files_concat_a, num_nina_yrs)
    init_files_concatao = np.divide(init_files_concatao, num_nino_yrs)
            
    neutnina_diff[_,:,:,:] = init_files_concata
    #neutnino_diff[_,:,:,:] = init_files_concato
    nina_diff[_,:,:,:] = init_files_concat_a
    nino_diff[_,:,:,:] = init_files_concatao
    
    print str(_)+' complete...'
    
    
###############################################################################
###############################################################################
###############################################################################


        
        
    sum_files_concata = xr.concat([grp_files_concata,grp_files_concata,grp_files_concata], dim='dayofyear')
    sum_files_concato = xr.concat([grp_files_concato,grp_files_concato,grp_files_concato], dim='dayofyear')
    sum_files_concat_a = xr.concat([grp_files_concat_a,grp_files_concat_a,grp_files_concat_a], dim='dayofyear')
    sum_files_concatao = xr.concat([grp_files_concatao,grp_files_concatao,grp_files_concatao], dim='dayofyear')
       
    for c, v in product(xrange(len(sum_files_concata[0,:,0])),xrange(len(sum_files_concata[0,0,:]))):
        
        sum_files_concata[:,c,v] = gfilt(sum_files_concata[:,c,v]*1.0,sigma=15.0)
        sum_files_concato[:,c,v] = gfilt(sum_files_concato[:,c,v]*1.0,sigma=15.0)
        sum_files_concat_a[:,c,v] = gfilt(sum_files_concat_a[:,c,v]*1.0,sigma=15.0)
        sum_files_concatao[:,c,v] = gfilt(sum_files_concatao[:,c,v]*1.0,sigma=15.0)
           
    sliced_concata = sum_files_concata[len(grp_files_concata[:,0,0]):len(grp_files_concata[:,0,0])*2,:,:]
    sliced_concato = sum_files_concato[len(grp_files_concato[:,0,0]):len(grp_files_concato[:,0,0])*2,:,:]
    sliced_concat_a = sum_files_concat_a[len(grp_files_concat_a[:,0,0]):len(grp_files_concat_a[:,0,0])*2,:,:]
    sliced_concatao = sum_files_concatao[len(grp_files_concatao[:,0,0]):len(grp_files_concatao[:,0,0])*2,:,:]
                
    sliced_concata = np.divide(sliced_concata,np.sum(sliced_concata, axis=0))
    sliced_concato = np.divide(sliced_concato,np.sum(sliced_concato, axis=0))
    sliced_concat_a = np.divide(sliced_concat_a,np.sum(sliced_concat_a, axis=0))
    sliced_concatao = np.divide(sliced_concatao,np.sum(sliced_concatao, axis=0))
        
    gauss_concata = np.ndarray.argmax(sliced_concata.values,axis=0)
    gauss_concato = np.ndarray.argmax(sliced_concato.values,axis=0)
    gauss_concat_a = np.ndarray.argmax(sliced_concat_a.values,axis=0)
    gauss_concatao = np.ndarray.argmax(sliced_concatao.values,axis=0)

    neut_diff_concata[_,:,:] = gauss_concata
    neut_diff_concato[_,:,:] = gauss_concato
    nina_diff_concat_a[_,:,:] = gauss_concat_a
    nino_diff_concatao[_,:,:] = gauss_concatao

    #nina_diff[_,:,:] = gauss_concat_a - gauss_concata
    #nino_diff[_,:,:] = gauss_concatao - gauss_concato
    
    print str(_)+' completed...'
    
 
neut_diff_concata = neut_diff_concata[:,:,:]
neut_diff_concato = neut_diff_concato[:,:,:]
nina_diff_concat_a = nina_diff_concat_a[:,:,:]
nino_diff_concatao = nino_diff_concatao[:,:,:]
    

#neut_diff_concata = neut_diff_concata[0:500,:,:]
#neut_diff_concato = neut_diff_concato[0:500,:,:]
#nina_diff_concat_a = nina_diff_concat_a[0:500,:,:]
#nino_diff_concatao = nino_diff_concatao[0:500,:,:]


with open('neut_diffs_concata_'+str(i), 'wb') as output:
    pickle.dump(neut_diff_concata, output, pickle.HIGHEST_PROTOCOL)

with open('neut_diffs_concato_'+str(i), 'wb') as output:
    pickle.dump(neut_diff_concato, output, pickle.HIGHEST_PROTOCOL)

with open('nina_diffs_concat_a_'+str(i), 'wb') as output:
    pickle.dump(nina_diff_concat_a, output, pickle.HIGHEST_PROTOCOL)

with open('nino_diffs_concatao_'+str(i), 'wb') as output:
    pickle.dump(nino_diff_concatao, output, pickle.HIGHEST_PROTOCOL)


###############################################################################
###############################################################################
###############################################################################
    
    
    
    
    
    

