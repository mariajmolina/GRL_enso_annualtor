#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 17:23:53 2018

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
import datetime as dt
import pandas as pd
from datetime import timedelta
from scipy.ndimage import gaussian_filter as gfilt


###############################################################################
###############################################################################
###############################################################################

tor_strength = 2

month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)


Julianday_neutral = []
Julianday_nino = []
Julianday_nina = []
    
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
 
    year_n = alltor1.severe['UTC_yr'].values
    month_n = alltor1.severe['UTC_mo'].values
    day_n = alltor1.severe['UTC_dy'].values

    for n in xrange(0,len(year_n)-1):
        Date_n = dt.datetime(year_n[n],month_n[n],day_n[n])
        Julianday_neutral.append(Date_n.strftime('%j'))

    year_neutral[k] = len(alltor1.neutral_base_yrs)


    print 'loop nina...'    
  
    year_en = alltor2.severe['UTC_yr'].values
    month_en = alltor2.severe['UTC_mo'].values
    day_en = alltor2.severe['UTC_dy'].values

    for n in xrange(0,len(year_en)-1):
        Date_en = dt.datetime(year_en[n],month_en[n],day_en[n])
        Julianday_nino.append(Date_en.strftime('%j'))
    
    year_nino[k] = len(alltor2.el_nino_yrs)


    print 'loop nina...'    

    year_ln = alltor3.severe['UTC_yr'].values
    month_ln = alltor3.severe['UTC_mo'].values
    day_ln = alltor3.severe['UTC_dy'].values

    for n in xrange(0,len(year_ln)-1):
        Date_ln = dt.datetime(year_ln[n],month_ln[n],day_ln[n])
        Julianday_nina.append(Date_ln.strftime('%j'))

    year_nina[k] = len(alltor3.la_nina_yrs)
    
    
###############################################################################
###############################################################################
###############################################################################
    

total_days = np.sum(day_list.astype(np.int))
    
yr_choices = pd.date_range('1953-01-01','2016-12-01',freq='AS').year.values

days_loop = np.hstack([0,day_list.astype(np.int)])

total_days_three = np.hstack([0,np.sum(day_list.astype(np.int)),np.sum(day_list.astype(np.int))*2])

severe = pd.read_csv('/storage/timme1mj/maria_pysplit/1950-2016_actual_tornadoes.csv', error_bad_lines=False,
                     parse_dates=[['mo','dy','yr','time']])

severe = severe.assign(UTC_time = severe['mo_dy_yr_time'] + timedelta(hours=6))       
severe = severe.assign(UTC_yr = severe['UTC_time'].dt.year) 
severe = severe.assign(UTC_dy = severe['UTC_time'].dt.day) 
severe = severe.assign(UTC_mo = severe['UTC_time'].dt.month) 
severe = severe.assign(UTC_hr = severe['UTC_time'].dt.hour)  
        
severe = severe[severe['slat']!=0]
severe = severe[severe['slon']!=0]
severe = severe[severe['slat']>=20]
severe = severe[severe['slat']<=50]
severe = severe[severe['slon']>=-130]
severe = severe[severe['slon']<=-65]
            
severe = severe[severe['mag']>=tor_strength]


sample_neut = np.zeros([1000,total_days])
sample_nino = np.zeros([1000,total_days])
sample_nina = np.zeros([1000,total_days])

for _ in xrange(1000):
    
    jd_neut = []
    jd_nino = []
    jd_nina = []
    
    for w in xrange(12):
        
        extract_N = [yr_choices[np.random.choice(len(yr_choices))] for i in xrange(year_neutral[w])]
        extract_LN = [yr_choices[np.random.choice(len(yr_choices))] for i in xrange(year_nina[w])]
        extract_EN = [yr_choices[np.random.choice(len(yr_choices))] for i in xrange(year_nino[w])]
        
        for i in extract_N:
            
            temp = severe[(severe['UTC_mo'] == w) & (severe['UTC_yr'] == i)]
            year_n = temp['UTC_yr'].values
            month_n = temp['UTC_mo'].values
            day_n = temp['UTC_dy'].values

            for n in xrange(0,len(year_n)-1):
                
                Date_n = dt.datetime(year_n[n],month_n[n],day_n[n])
                jd_neut.append(Date_n.strftime('%j'))
        
        for i in extract_LN:

            temp = severe[(severe['UTC_mo'] == w) & (severe['UTC_yr'] == i)]
            year_n = temp['UTC_yr'].values
            month_n = temp['UTC_mo'].values
            day_n = temp['UTC_dy'].values

            for n in xrange(0,len(year_n)-1):
                
                Date_n = dt.datetime(year_n[n],month_n[n],day_n[n])
                jd_nina.append(Date_n.strftime('%j'))
            
        for i in extract_EN:
            
            temp = severe[(severe['UTC_mo'] == w) & (severe['UTC_yr'] == i)]
            year_n = temp['UTC_yr'].values
            month_n = temp['UTC_mo'].values
            day_n = temp['UTC_dy'].values

            for n in xrange(0,len(year_n)-1):
                
                Date_n = dt.datetime(year_n[n],month_n[n],day_n[n])
                jd_nino.append(Date_n.strftime('%j'))
                

    Julianday_neutral = [int(i) for i in jd_neut]
    Histo = np.histogram(Julianday_neutral, bins = total_days)
    HistoN = np.zeros(total_days * 3)
    HistoN[0:total_days] = Histo[0] 
    HistoN[total_days:total_days*2] = Histo[0]  
    HistoN[total_days*2:total_days*3] = Histo[0]

    Julianday_nino = [int(i) for i in jd_nino]
    Histo2 = np.histogram(Julianday_nino, bins = total_days)
    HistoEN = np.zeros(total_days * 3)
    HistoEN[0:total_days] = Histo2[0] 
    HistoEN[total_days:total_days*2] = Histo2[0]  
    HistoEN[total_days*2:total_days*3] = Histo2[0]

    Julianday_nina = [int(i) for i in jd_nina]
    Histo3 = np.histogram(Julianday_nina, bins = total_days)
    HistoLN = np.zeros(total_days * 3)
    HistoLN[0:total_days] = Histo3[0] 
    HistoLN[total_days:total_days*2] = Histo3[0]  
    HistoLN[total_days*2:total_days*3] = Histo3[0]    
            
    Gauss_SmoothTN = []
    Gauss_SmoothTELN = []
    Gauss_SmoothTLN = []

    for q in total_days_three:
        
        for j in xrange(len(year_nina)):
            
            Gauss_SmoothTN.append(HistoN[(np.sum(days_loop[:j+1])+q):(np.sum(days_loop[:j+2])+(q))]/(year_neutral[j]))  
            Gauss_SmoothTELN.append(HistoEN[(np.sum(days_loop[:j+1])+q):(np.sum(days_loop[:j+2])+(q))]/(year_nino[j]))
            Gauss_SmoothTLN.append(HistoLN[(np.sum(days_loop[:j+1])+q):(np.sum(days_loop[:j+2])+(q))]/(year_nina[j]))
        
    Gauss_SmoothTN = np.concatenate(Gauss_SmoothTN).ravel()
    Gauss_SmoothTELN = np.concatenate(Gauss_SmoothTELN).ravel()
    Gauss_SmoothTLN = np.concatenate(Gauss_SmoothTLN).ravel()

    Gauss_SmoothTN = gfilt(Gauss_SmoothTN*1.0,sigma=15.0)
    Gauss_SmoothTELN = gfilt(Gauss_SmoothTELN*1.0,sigma=15.0)
    Gauss_SmoothTLN = gfilt(Gauss_SmoothTLN*1.0,sigma=15.0)

    Gauss_SmoothTN1 = Gauss_SmoothTN[total_days_three[1]:total_days_three[2]]
    Gauss_SmoothTELN1 = Gauss_SmoothTELN[total_days_three[1]:total_days_three[2]]
    Gauss_SmoothTLN1 = Gauss_SmoothTLN[total_days_three[1]:total_days_three[2]]        
        
    sample_neut[_,:] = Gauss_SmoothTN1
    sample_nino[_,:] = Gauss_SmoothTELN1
    sample_nina[_,:] = Gauss_SmoothTLN1
        
    print str(_)+' completed...'
    
    

np.save('ef2_neut_linesig_'+str(i), sample_neut)
np.save('ef2_nino_linesig_'+str(i), sample_nino)
np.save('ef2_nina_linesig_'+str(i), sample_nina)


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


neut_linesig_1 = np.load('ef1_neut_linesig_1.npy')
neut_linesig_2 = np.load('ef1_neut_linesig_2.npy')
neut_linesig_3 = np.load('ef1_neut_linesig_3.npy')
neut_linesig_4 = np.load('ef1_neut_linesig_4.npy')
neut_linesig_5 = np.load('ef1_neut_linesig_5.npy')
neut_linesig_6 = np.load('ef1_neut_linesig_6.npy')
neut_linesig_7 = np.load('ef1_neut_linesig_7.npy')
neut_linesig_8 = np.load('ef1_neut_linesig_8.npy')
neut_linesig_9 = np.load('ef1_neut_linesig_9.npy')
neut_linesig_10 = np.load('ef1_neut_linesig_10.npy')

neut_linesig = np.vstack([neut_linesig_1,neut_linesig_2,neut_linesig_3,
                          neut_linesig_4,neut_linesig_5,neut_linesig_6,
                          neut_linesig_7,neut_linesig_8,neut_linesig_9,
                          neut_linesig_10])


nino_linesig_1 = np.load('ef1_nino_linesig_1.npy')
nino_linesig_2 = np.load('ef1_nino_linesig_2.npy')
nino_linesig_3 = np.load('ef1_nino_linesig_3.npy')
nino_linesig_4 = np.load('ef1_nino_linesig_4.npy')
nino_linesig_5 = np.load('ef1_nino_linesig_5.npy')
nino_linesig_6 = np.load('ef1_nino_linesig_6.npy')
nino_linesig_7 = np.load('ef1_nino_linesig_7.npy')
nino_linesig_8 = np.load('ef1_nino_linesig_8.npy')
nino_linesig_9 = np.load('ef1_nino_linesig_9.npy')
nino_linesig_10 = np.load('ef1_nino_linesig_10.npy')

nino_linesig = np.vstack([nino_linesig_1,nino_linesig_2,nino_linesig_3,
                          nino_linesig_4,nino_linesig_5,nino_linesig_6,
                          nino_linesig_7,nino_linesig_8,nino_linesig_9,
                          nino_linesig_10])
    
    
nina_linesig_1 = np.load('ef1_nina_linesig_1.npy')
nina_linesig_2 = np.load('ef1_nina_linesig_2.npy')
nina_linesig_3 = np.load('ef1_nina_linesig_3.npy')
nina_linesig_4 = np.load('ef1_nina_linesig_4.npy')
nina_linesig_5 = np.load('ef1_nina_linesig_5.npy')
nina_linesig_6 = np.load('ef1_nina_linesig_6.npy')
nina_linesig_7 = np.load('ef1_nina_linesig_7.npy')
nina_linesig_8 = np.load('ef1_nina_linesig_8.npy')
nina_linesig_9 = np.load('ef1_nina_linesig_9.npy')
nina_linesig_10 = np.load('ef1_nina_linesig_10.npy')

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


import numpy as np


neut_linesig_1 = np.load('ef2_neut_linesig_1.npy')
neut_linesig_2 = np.load('ef2_neut_linesig_2.npy')
neut_linesig_3 = np.load('ef2_neut_linesig_3.npy')
neut_linesig_4 = np.load('ef2_neut_linesig_4.npy')
neut_linesig_5 = np.load('ef2_neut_linesig_5.npy')
neut_linesig_6 = np.load('ef2_neut_linesig_6.npy')
neut_linesig_7 = np.load('ef2_neut_linesig_7.npy')
neut_linesig_8 = np.load('ef2_neut_linesig_8.npy')
neut_linesig_9 = np.load('ef2_neut_linesig_9.npy')
neut_linesig_10 = np.load('ef2_neut_linesig_10.npy')

neut_linesig = np.vstack([neut_linesig_1,neut_linesig_2,neut_linesig_3,
                          neut_linesig_4,neut_linesig_5,neut_linesig_6,
                          neut_linesig_7,neut_linesig_8,neut_linesig_9,
                          neut_linesig_10])


nino_linesig_1 = np.load('ef2_nino_linesig_1.npy')
nino_linesig_2 = np.load('ef2_nino_linesig_2.npy')
nino_linesig_3 = np.load('ef2_nino_linesig_3.npy')
nino_linesig_4 = np.load('ef2_nino_linesig_4.npy')
nino_linesig_5 = np.load('ef2_nino_linesig_5.npy')
nino_linesig_6 = np.load('ef2_nino_linesig_6.npy')
nino_linesig_7 = np.load('ef2_nino_linesig_7.npy')
nino_linesig_8 = np.load('ef2_nino_linesig_8.npy')
nino_linesig_9 = np.load('ef2_nino_linesig_9.npy')
nino_linesig_10 = np.load('ef2_nino_linesig_10.npy')

nino_linesig = np.vstack([nino_linesig_1,nino_linesig_2,nino_linesig_3,
                          nino_linesig_4,nino_linesig_5,nino_linesig_6,
                          nino_linesig_7,nino_linesig_8,nino_linesig_9,
                          nino_linesig_10])
    
    
nina_linesig_1 = np.load('ef2_nina_linesig_1.npy')
nina_linesig_2 = np.load('ef2_nina_linesig_2.npy')
nina_linesig_3 = np.load('ef2_nina_linesig_3.npy')
nina_linesig_4 = np.load('ef2_nina_linesig_4.npy')
nina_linesig_5 = np.load('ef2_nina_linesig_5.npy')
nina_linesig_6 = np.load('ef2_nina_linesig_6.npy')
nina_linesig_7 = np.load('ef2_nina_linesig_7.npy')
nina_linesig_8 = np.load('ef2_nina_linesig_8.npy')
nina_linesig_9 = np.load('ef2_nina_linesig_9.npy')
nina_linesig_10 = np.load('ef2_nina_linesig_10.npy')

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


    print str(_)
    
    
    
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
