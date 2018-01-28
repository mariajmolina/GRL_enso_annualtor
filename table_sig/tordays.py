#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:04:01 2018

Maria J. Molina
Ph.D. Student
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


###############################################################################
###############################################################################
###############################################################################


tor_strength = 1


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)
    
year_neutral = {}
year_nino = {}
year_nina = {}


###############################################################################
###############################################################################
###############################################################################


for k, (i, j) in enumerate(zip(month_list, day_list)):

    
    print 'Running month: '+i+' and day: '+j+'...'
   
    
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor1 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=tor_strength,
                              enso_neutral=True, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor1.severe_load()    


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor2 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=tor_strength,
                              enso_neutral=False, enso_only=False, nino_only=True, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor2.severe_load()


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'
    
    alltor3 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=tor_strength,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=True,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor3.severe_load()



    print 'loop neutral...'

    year_neutral[k] = len(alltor1.neutral_base_yrs)


    print 'loop nina...'    
    
    year_nino[k] = len(alltor2.el_nino_yrs)


    print 'loop nina...'    

    year_nina[k] = len(alltor3.la_nina_yrs)



with open('yr_neut_values_days', 'wb') as output:
    pickle.dump(year_neutral.values(), output, pickle.HIGHEST_PROTOCOL)
        
with open('yr_nino_values_days', 'wb') as output:
    pickle.dump(year_nino.values(), output, pickle.HIGHEST_PROTOCOL)
        
with open('yr_nina_values_days', 'wb') as output:
    pickle.dump(year_nina.values(), output, pickle.HIGHEST_PROTOCOL)


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


###############################################################################
###############################################################################
###############################################################################


p = 10

#US_tors_ef1 = xr.open_dataset('tors_5316_ef1_obs', decode_cf=True)
US_tors_ef1 = xr.open_dataset('tors_5316_ef2_obs', decode_cf=True)

US_tors_ef1 = US_tors_ef1.grid.where(US_tors_ef1.grid==0, 1)
#US_tors_ef1 = US_tors_ef1.grid.where(US_tors_ef1.grid==0, 1)

US_sumtors_ef1 = US_tors_ef1.sum(['x','y'])
#US_sumtors_ef1 = US_tors_ef1.sum(['x','y'])


with open('yr_neut_values_days', 'rb') as f:
    num_neut_yrs = pickle.load(f)
    
with open('yr_nino_values_days', 'rb') as f:
    num_nina_yrs = pickle.load(f)
    
with open('yr_nina_values_days', 'rb') as f:
    num_nino_yrs = pickle.load(f)
    

###############################################################################
###############################################################################
###############################################################################


yr_choices = pd.date_range('1953-01-01','2016-12-01',freq='AS').year.values


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
            
                file_1 = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                
            elif e == 1:
            
                file_2 = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                file_neut = np.stack([file_1[:day_list[x].astype('int64')],file_2[:day_list[x].astype('int64')]])

            else:
            
                file_more = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                file_more = np.expand_dims(file_more, axis=0)
                file_neut = np.vstack([file_neut, file_more[:,:day_list[x].astype('int64')]])

        
        for e, i in enumerate(rand_nina):
        
            if e == 0:
            
                file_1 = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                
            elif e == 1:
            
                file_2 = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                file_nina = np.stack([file_1[:day_list[x].astype('int64')],file_2[:day_list[x].astype('int64')]])

            else:
            
                file_more = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                file_more = np.expand_dims(file_more, axis=0)
                file_nina = np.vstack([file_nina, file_more[:,:day_list[x].astype('int64')]])
            
            
        for e, i in enumerate(rand_nino):
        
            if e == 0:
                
                file_1 = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                
            elif e == 1:
            
                file_2 = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
                file_nino = np.stack([file_1[:day_list[x].astype('int64')],file_2[:day_list[x].astype('int64')]])

            else:
            
                file_more = US_sumtors_ef1[(US_sumtors_ef1['time.month']==x+1) & (US_sumtors_ef1['time.year']==i)]
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
    

#np.save('summed_neut_days_ef1_'+str(p), summed_neut)
#np.save('summed_nina_days_ef1_'+str(p), summed_nina)
#np.save('summed_nino_days_ef1_'+str(p), summed_nino)

np.save('summed_neut_days_ef2_'+str(p), summed_neut)
np.save('summed_nina_days_ef2_'+str(p), summed_nina)
np.save('summed_nino_days_ef2_'+str(p), summed_nino)

    
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



import numpy as np


neut_linesig_1 = np.load('summed_neut_days_ef1_1.npy')
neut_linesig_2 = np.load('summed_neut_days_ef1_2.npy')
neut_linesig_3 = np.load('summed_neut_days_ef1_3.npy')
neut_linesig_4 = np.load('summed_neut_days_ef1_4.npy')
neut_linesig_5 = np.load('summed_neut_days_ef1_5.npy')
neut_linesig_6 = np.load('summed_neut_days_ef1_6.npy')
neut_linesig_7 = np.load('summed_neut_days_ef1_7.npy')
neut_linesig_8 = np.load('summed_neut_days_ef1_8.npy')
neut_linesig_9 = np.load('summed_neut_days_ef1_9.npy')
neut_linesig_10 = np.load('summed_neut_days_ef1_10.npy')

neut_linesig = np.vstack([neut_linesig_1,neut_linesig_2,neut_linesig_3,
                          neut_linesig_4,neut_linesig_5,neut_linesig_6,
                          neut_linesig_7,neut_linesig_8,neut_linesig_9,
                          neut_linesig_10])


nino_linesig_1 = np.load('summed_nino_days_ef1_1.npy')
nino_linesig_2 = np.load('summed_nino_days_ef1_2.npy')
nino_linesig_3 = np.load('summed_nino_days_ef1_3.npy')
nino_linesig_4 = np.load('summed_nino_days_ef1_4.npy')
nino_linesig_5 = np.load('summed_nino_days_ef1_5.npy')
nino_linesig_6 = np.load('summed_nino_days_ef1_6.npy')
nino_linesig_7 = np.load('summed_nino_days_ef1_7.npy')
nino_linesig_8 = np.load('summed_nino_days_ef1_8.npy')
nino_linesig_9 = np.load('summed_nino_days_ef1_9.npy')
nino_linesig_10 = np.load('summed_nino_days_ef1_10.npy')

nino_linesig = np.vstack([nino_linesig_1,nino_linesig_2,nino_linesig_3,
                          nino_linesig_4,nino_linesig_5,nino_linesig_6,
                          nino_linesig_7,nino_linesig_8,nino_linesig_9,
                          nino_linesig_10])
    
    
nina_linesig_1 = np.load('summed_nina_days_ef1_1.npy')
nina_linesig_2 = np.load('summed_nina_days_ef1_2.npy')
nina_linesig_3 = np.load('summed_nina_days_ef1_3.npy')
nina_linesig_4 = np.load('summed_nina_days_ef1_4.npy')
nina_linesig_5 = np.load('summed_nina_days_ef1_5.npy')
nina_linesig_6 = np.load('summed_nina_days_ef1_6.npy')
nina_linesig_7 = np.load('summed_nina_days_ef1_7.npy')
nina_linesig_8 = np.load('summed_nina_days_ef1_8.npy')
nina_linesig_9 = np.load('summed_nina_days_ef1_9.npy')
nina_linesig_10 = np.load('summed_nina_days_ef1_10.npy')

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
###############################################################################   
###############################################################################
###############################################################################
###############################################################################



import numpy as np


neut_linesig_1 = np.load('summed_neut_days_ef2_1.npy')
neut_linesig_2 = np.load('summed_neut_days_ef2_2.npy')
neut_linesig_3 = np.load('summed_neut_days_ef2_3.npy')
neut_linesig_4 = np.load('summed_neut_days_ef2_4.npy')
neut_linesig_5 = np.load('summed_neut_days_ef2_5.npy')
neut_linesig_6 = np.load('summed_neut_days_ef2_6.npy')
neut_linesig_7 = np.load('summed_neut_days_ef2_7.npy')
neut_linesig_8 = np.load('summed_neut_days_ef2_8.npy')
neut_linesig_9 = np.load('summed_neut_days_ef2_9.npy')
neut_linesig_10 = np.load('summed_neut_days_ef2_10.npy')

neut_linesig = np.vstack([neut_linesig_1,neut_linesig_2,neut_linesig_3,
                          neut_linesig_4,neut_linesig_5,neut_linesig_6,
                          neut_linesig_7,neut_linesig_8,neut_linesig_9,
                          neut_linesig_10])


nino_linesig_1 = np.load('summed_nino_days_ef2_1.npy')
nino_linesig_2 = np.load('summed_nino_days_ef2_2.npy')
nino_linesig_3 = np.load('summed_nino_days_ef2_3.npy')
nino_linesig_4 = np.load('summed_nino_days_ef2_4.npy')
nino_linesig_5 = np.load('summed_nino_days_ef2_5.npy')
nino_linesig_6 = np.load('summed_nino_days_ef2_6.npy')
nino_linesig_7 = np.load('summed_nino_days_ef2_7.npy')
nino_linesig_8 = np.load('summed_nino_days_ef2_8.npy')
nino_linesig_9 = np.load('summed_nino_days_ef2_9.npy')
nino_linesig_10 = np.load('summed_nino_days_ef2_10.npy')

nino_linesig = np.vstack([nino_linesig_1,nino_linesig_2,nino_linesig_3,
                          nino_linesig_4,nino_linesig_5,nino_linesig_6,
                          nino_linesig_7,nino_linesig_8,nino_linesig_9,
                          nino_linesig_10])
    
    
nina_linesig_1 = np.load('summed_nina_days_ef2_1.npy')
nina_linesig_2 = np.load('summed_nina_days_ef2_2.npy')
nina_linesig_3 = np.load('summed_nina_days_ef2_3.npy')
nina_linesig_4 = np.load('summed_nina_days_ef2_4.npy')
nina_linesig_5 = np.load('summed_nina_days_ef2_5.npy')
nina_linesig_6 = np.load('summed_nina_days_ef2_6.npy')
nina_linesig_7 = np.load('summed_nina_days_ef2_7.npy')
nina_linesig_8 = np.load('summed_nina_days_ef2_8.npy')
nina_linesig_9 = np.load('summed_nina_days_ef2_9.npy')
nina_linesig_10 = np.load('summed_nina_days_ef2_10.npy')

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
###############################################################################   

    
    

