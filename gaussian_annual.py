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
import datetime as dt
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter as gfilt
import matplotlib.patches as mpatches
#from itertools import product


###############################################################################
###############################################################################
###############################################################################


month_list = np.array(['01','02','03','04','05','06',
                       '07','08','09','10','11','12'], dtype=str)

day_list = np.array(['31','28','31','30','31','30',
                     '31','31','30','31','30','31'], dtype=str)

Julianday_all = []
Julianday_neutral = []
Julianday_nino = []
Julianday_nina = []

year_all = {}
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

    alltor0 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor0.severe_load() 
    
    
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor1 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=True, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor1.severe_load()    


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor2 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=True, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor2.severe_load()


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'
    
    alltor3 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=True,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor3.severe_load()


    print 'loop all...'

    year_a = alltor0.severe['UTC_yr'].values
    month_a = alltor0.severe['UTC_mo'].values
    day_a = alltor0.severe['UTC_dy'].values

    for n in xrange(0,len(year_a)-1):
        Date_a = dt.datetime(year_a[n],month_a[n],day_a[n])
        Julianday_all.append(Date_a.strftime('%j'))

    year_all[k] = len(alltor0.all_base_yrs)
    

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

Julianday_all = [int(i) for i in Julianday_all]
Histo0 = np.histogram(Julianday_all, bins = total_days)
HistoA = np.zeros(total_days * 3)
HistoA[0:total_days] = Histo0[0] 
HistoA[total_days:total_days*2] = Histo0[0]  
HistoA[total_days*2:total_days*3] = Histo0[0]
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
Histo = np.histogram(Julianday_neutral, bins = total_days)
HistoN = np.zeros(total_days * 3)
HistoN[0:total_days] = Histo[0] 
HistoN[total_days:total_days*2] = Histo[0]  
HistoN[total_days*2:total_days*3] = Histo[0]

Julianday_nino = [int(i) for i in Julianday_nino]
Histo2 = np.histogram(Julianday_nino, bins = total_days)
HistoEN = np.zeros(total_days * 3)
HistoEN[0:total_days] = Histo2[0] 
HistoEN[total_days:total_days*2] = Histo2[0]  
HistoEN[total_days*2:total_days*3] = Histo2[0]

Julianday_nina = [int(i) for i in Julianday_nina]
Histo3 = np.histogram(Julianday_nina, bins = total_days)
HistoLN = np.zeros(total_days * 3)
HistoLN[0:total_days] = Histo3[0] 
HistoLN[total_days:total_days*2] = Histo3[0]  
HistoLN[total_days*2:total_days*3] = Histo3[0]


###############################################################################
###############################################################################
###############################################################################


days_loop = np.hstack([0,day_list.astype(np.int)])

total_days_three = np.hstack([0,np.sum(day_list.astype(np.int)),np.sum(day_list.astype(np.int))*2])

Gauss_SmoothAN = []
Gauss_SmoothTN = []
Gauss_SmoothTELN = []
Gauss_SmoothTLN = []

for q in total_days_three:
    for w in xrange(len(year_nina)):
        Gauss_SmoothAN.append(HistoA[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_all[w]))
        Gauss_SmoothTN.append(HistoN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_neutral[w]))  
        Gauss_SmoothTELN.append(HistoEN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nino[w]))
        Gauss_SmoothTLN.append(HistoLN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nina[w]))

Gauss_SmoothAN = np.concatenate(Gauss_SmoothAN).ravel()
Gauss_SmoothTN = np.concatenate(Gauss_SmoothTN).ravel()
Gauss_SmoothTELN = np.concatenate(Gauss_SmoothTELN).ravel()
Gauss_SmoothTLN = np.concatenate(Gauss_SmoothTLN).ravel()

Gauss_SmoothAN = gfilt(Gauss_SmoothAN*1.0,sigma=15.0)
Gauss_SmoothTN = gfilt(Gauss_SmoothTN*1.0,sigma=15.0)
Gauss_SmoothTELN = gfilt(Gauss_SmoothTELN*1.0,sigma=15.0)
Gauss_SmoothTLN = gfilt(Gauss_SmoothTLN*1.0,sigma=15.0)

Gauss_SmoothAN1 = Gauss_SmoothAN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTN1 = Gauss_SmoothTN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTELN1 = Gauss_SmoothTELN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTLN1 = Gauss_SmoothTLN[total_days_three[1]:total_days_three[2]]


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

Julianday_all = []
Julianday_neutral = []
Julianday_nino = []
Julianday_nina = []

year_all = {}
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

    alltor0 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor0.severe_load() 
    
    
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor1 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=True, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor1.severe_load()    


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor2 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=True, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor2.severe_load()


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'
    
    alltor3 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=True,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='CONUS')
                       
    alltor3.severe_load()


    print 'loop all...'

    year_a = alltor0.severe['UTC_yr'].values
    month_a = alltor0.severe['UTC_mo'].values
    day_a = alltor0.severe['UTC_dy'].values

    for n in xrange(0,len(year_a)-1):
        Date_a = dt.datetime(year_a[n],month_a[n],day_a[n])
        Julianday_all.append(Date_a.strftime('%j'))

    year_all[k] = len(alltor0.all_base_yrs)
    

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

Julianday_all = [int(i) for i in Julianday_all]
Histo0 = np.histogram(Julianday_all, bins = total_days)
HistoA = np.zeros(total_days * 3)
HistoA[0:total_days] = Histo0[0] 
HistoA[total_days:total_days*2] = Histo0[0]  
HistoA[total_days*2:total_days*3] = Histo0[0]
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
Histo = np.histogram(Julianday_neutral, bins = total_days)
HistoN = np.zeros(total_days * 3)
HistoN[0:total_days] = Histo[0] 
HistoN[total_days:total_days*2] = Histo[0]  
HistoN[total_days*2:total_days*3] = Histo[0]

Julianday_nino = [int(i) for i in Julianday_nino]
Histo2 = np.histogram(Julianday_nino, bins = total_days)
HistoEN = np.zeros(total_days * 3)
HistoEN[0:total_days] = Histo2[0] 
HistoEN[total_days:total_days*2] = Histo2[0]  
HistoEN[total_days*2:total_days*3] = Histo2[0]

Julianday_nina = [int(i) for i in Julianday_nina]
Histo3 = np.histogram(Julianday_nina, bins = total_days)
HistoLN = np.zeros(total_days * 3)
HistoLN[0:total_days] = Histo3[0] 
HistoLN[total_days:total_days*2] = Histo3[0]  
HistoLN[total_days*2:total_days*3] = Histo3[0]


###############################################################################
###############################################################################
###############################################################################


days_loop = np.hstack([0,day_list.astype(np.int)])

total_days_three = np.hstack([0,np.sum(day_list.astype(np.int)),np.sum(day_list.astype(np.int))*2])

Gauss_SmoothAN = []
Gauss_SmoothTN = []
Gauss_SmoothTELN = []
Gauss_SmoothTLN = []

for q in total_days_three:
    for w in xrange(len(year_nina)):
        Gauss_SmoothAN.append(HistoA[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_all[w]))  
        Gauss_SmoothTN.append(HistoN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_neutral[w]))   
        Gauss_SmoothTELN.append(HistoEN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nino[w]))
        Gauss_SmoothTLN.append(HistoLN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nina[w]))

Gauss_SmoothAN = np.concatenate(Gauss_SmoothAN).ravel()
Gauss_SmoothTN = np.concatenate(Gauss_SmoothTN).ravel()
Gauss_SmoothTELN = np.concatenate(Gauss_SmoothTELN).ravel()
Gauss_SmoothTLN = np.concatenate(Gauss_SmoothTLN).ravel()

Gauss_SmoothAN = gfilt(Gauss_SmoothAN*1.0,sigma=15.0)
Gauss_SmoothTN = gfilt(Gauss_SmoothTN*1.0,sigma=15.0)
Gauss_SmoothTELN = gfilt(Gauss_SmoothTELN*1.0,sigma=15.0)
Gauss_SmoothTLN = gfilt(Gauss_SmoothTLN*1.0,sigma=15.0)

Gauss_SmoothAN2 = Gauss_SmoothAN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTN2 = Gauss_SmoothTN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTELN2 = Gauss_SmoothTELN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTLN2 = Gauss_SmoothTLN[total_days_three[1]:total_days_three[2]]


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

Julianday_all = []
Julianday_neutral = []
Julianday_nino = []
Julianday_nina = []

year_all = {}
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

    alltor0 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor0.severe_load() 
    
    
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor1 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=True, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor1.severe_load()    


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor2 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=True, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor2.severe_load()


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'
    
    alltor3 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=True,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor3.severe_load()


    print 'loop all...'

    year_a = alltor0.severe['UTC_yr'].values
    month_a = alltor0.severe['UTC_mo'].values
    day_a = alltor0.severe['UTC_dy'].values

    for n in xrange(0,len(year_a)-1):
        Date_a = dt.datetime(year_a[n],month_a[n],day_a[n])
        Julianday_all.append(Date_a.strftime('%j'))

    year_all[k] = len(alltor0.all_base_yrs)
    

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

Julianday_all = [int(i) for i in Julianday_all]
Histo0 = np.histogram(Julianday_all, bins = total_days)
HistoA = np.zeros(total_days * 3)
HistoA[0:total_days] = Histo0[0] 
HistoA[total_days:total_days*2] = Histo0[0]  
HistoA[total_days*2:total_days*3] = Histo0[0]
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
Histo = np.histogram(Julianday_neutral, bins = total_days)
HistoN = np.zeros(total_days * 3)
HistoN[0:total_days] = Histo[0] 
HistoN[total_days:total_days*2] = Histo[0]  
HistoN[total_days*2:total_days*3] = Histo[0]

Julianday_nino = [int(i) for i in Julianday_nino]
Histo2 = np.histogram(Julianday_nino, bins = total_days)
HistoEN = np.zeros(total_days * 3)
HistoEN[0:total_days] = Histo2[0] 
HistoEN[total_days:total_days*2] = Histo2[0]  
HistoEN[total_days*2:total_days*3] = Histo2[0]

Julianday_nina = [int(i) for i in Julianday_nina]
Histo3 = np.histogram(Julianday_nina, bins = total_days)
HistoLN = np.zeros(total_days * 3)
HistoLN[0:total_days] = Histo3[0] 
HistoLN[total_days:total_days*2] = Histo3[0]  
HistoLN[total_days*2:total_days*3] = Histo3[0]


###############################################################################
###############################################################################
###############################################################################


days_loop = np.hstack([0,day_list.astype(np.int)])

total_days_three = np.hstack([0,np.sum(day_list.astype(np.int)),np.sum(day_list.astype(np.int))*2])

Gauss_SmoothAN = []
Gauss_SmoothTN = []
Gauss_SmoothTELN = []
Gauss_SmoothTLN = []

for q in total_days_three:
    for w in xrange(len(year_nina)):
        Gauss_SmoothAN.append(HistoA[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_all[w]))  
        Gauss_SmoothTN.append(HistoN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_neutral[w]))   
        Gauss_SmoothTELN.append(HistoEN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nino[w]))
        Gauss_SmoothTLN.append(HistoLN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nina[w]))

Gauss_SmoothAN = np.concatenate(Gauss_SmoothAN).ravel()
Gauss_SmoothTN = np.concatenate(Gauss_SmoothTN).ravel()
Gauss_SmoothTELN = np.concatenate(Gauss_SmoothTELN).ravel()
Gauss_SmoothTLN = np.concatenate(Gauss_SmoothTLN).ravel()

Gauss_SmoothAN = gfilt(Gauss_SmoothAN*1.0,sigma=15.0)
Gauss_SmoothTN = gfilt(Gauss_SmoothTN*1.0,sigma=15.0)
Gauss_SmoothTELN = gfilt(Gauss_SmoothTELN*1.0,sigma=15.0)
Gauss_SmoothTLN = gfilt(Gauss_SmoothTLN*1.0,sigma=15.0)

Gauss_SmoothAN3 = Gauss_SmoothAN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTN3 = Gauss_SmoothTN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTELN3 = Gauss_SmoothTELN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTLN3 = Gauss_SmoothTLN[total_days_three[1]:total_days_three[2]]


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

Julianday_all = []
Julianday_neutral = []
Julianday_nino = []
Julianday_nina = []

year_all = {}
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

    alltor0 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor0.severe_load() 
    
    
    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor1 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=True, enso_only=False, nino_only=False, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor1.severe_load()    


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'

    alltor2 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=True, nina_only=False,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor2.severe_load()


    time_1 = '1953-'+i+'-01 00:00:00'
    time_2 = '2016-'+i+'-'+j+' 21:00:00'
    
    alltor3 = pk.whats_up_doc(time_1, time_2,
                              frequency = 'subseason', stat_choice = 'composite', tor_min=2,
                              enso_neutral=False, enso_only=False, nino_only=False, nina_only=True,
                              oni_demarcator=0.5, number_of_composite_years=0,
                              auto_run=False, save_name='full_comp8', region='southeast')
                       
    alltor3.severe_load()


    print 'loop all...'

    year_a = alltor0.severe['UTC_yr'].values
    month_a = alltor0.severe['UTC_mo'].values
    day_a = alltor0.severe['UTC_dy'].values

    for n in xrange(0,len(year_a)-1):
        Date_a = dt.datetime(year_a[n],month_a[n],day_a[n])
        Julianday_all.append(Date_a.strftime('%j'))

    year_all[k] = len(alltor0.all_base_yrs)
    

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

Julianday_all = [int(i) for i in Julianday_all]
Histo0 = np.histogram(Julianday_all, bins = total_days)
HistoA = np.zeros(total_days * 3)
HistoA[0:total_days] = Histo0[0] 
HistoA[total_days:total_days*2] = Histo0[0]  
HistoA[total_days*2:total_days*3] = Histo0[0]
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
Histo = np.histogram(Julianday_neutral, bins = total_days)
HistoN = np.zeros(total_days * 3)
HistoN[0:total_days] = Histo[0] 
HistoN[total_days:total_days*2] = Histo[0]  
HistoN[total_days*2:total_days*3] = Histo[0]

Julianday_nino = [int(i) for i in Julianday_nino]
Histo2 = np.histogram(Julianday_nino, bins = total_days)
HistoEN = np.zeros(total_days * 3)
HistoEN[0:total_days] = Histo2[0] 
HistoEN[total_days:total_days*2] = Histo2[0]  
HistoEN[total_days*2:total_days*3] = Histo2[0]

Julianday_nina = [int(i) for i in Julianday_nina]
Histo3 = np.histogram(Julianday_nina, bins = total_days)
HistoLN = np.zeros(total_days * 3)
HistoLN[0:total_days] = Histo3[0] 
HistoLN[total_days:total_days*2] = Histo3[0]  
HistoLN[total_days*2:total_days*3] = Histo3[0]


###############################################################################
###############################################################################
###############################################################################


days_loop = np.hstack([0,day_list.astype(np.int)])

total_days_three = np.hstack([0,np.sum(day_list.astype(np.int)),np.sum(day_list.astype(np.int))*2])

Gauss_SmoothAN = []
Gauss_SmoothTN = []
Gauss_SmoothTELN = []
Gauss_SmoothTLN = []

for q in total_days_three:
    for w in xrange(len(year_nina)):
        Gauss_SmoothAN.append(HistoA[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_all[w]))  
        Gauss_SmoothTN.append(HistoN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_neutral[w]))   
        Gauss_SmoothTELN.append(HistoEN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nino[w]))
        Gauss_SmoothTLN.append(HistoLN[(np.sum(days_loop[:w+1])+q):(np.sum(days_loop[:w+2])+(q))]/(year_nina[w]))

Gauss_SmoothAN = np.concatenate(Gauss_SmoothAN).ravel()
Gauss_SmoothTN = np.concatenate(Gauss_SmoothTN).ravel()
Gauss_SmoothTELN = np.concatenate(Gauss_SmoothTELN).ravel()
Gauss_SmoothTLN = np.concatenate(Gauss_SmoothTLN).ravel()

Gauss_SmoothAN = gfilt(Gauss_SmoothAN*1.0,sigma=15.0)
Gauss_SmoothTN = gfilt(Gauss_SmoothTN*1.0,sigma=15.0)
Gauss_SmoothTELN = gfilt(Gauss_SmoothTELN*1.0,sigma=15.0)
Gauss_SmoothTLN = gfilt(Gauss_SmoothTLN*1.0,sigma=15.0)

Gauss_SmoothAN4 = Gauss_SmoothAN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTN4 = Gauss_SmoothTN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTELN4 = Gauss_SmoothTELN[total_days_three[1]:total_days_three[2]]
Gauss_SmoothTLN4 = Gauss_SmoothTLN[total_days_three[1]:total_days_three[2]]


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


US_tors_ef1 = xr.open_dataset('tors_5316_ef1_obs', decode_cf=True)
US_tors_ef2 = xr.open_dataset('tors_5316_ef2_obs', decode_cf=True)
SE_tors_ef1 = xr.open_dataset('tors_5316_ef1_SE_obs', decode_cf=True)
SE_tors_ef2 = xr.open_dataset('tors_5316_ef2_SE_obs', decode_cf=True)

US_nuyr_ef1 = len(US_tors_ef1.grid.groupby('time.year').sum('time').year.values)
US_nuyr_ef2 = len(US_tors_ef2.grid.groupby('time.year').sum('time').year.values)
SE_nuyr_ef1 = len(SE_tors_ef1.grid.groupby('time.year').sum('time').year.values)
SE_nuyr_ef2 = len(SE_tors_ef2.grid.groupby('time.year').sum('time').year.values)

US_sumtors_ef1 = US_tors_ef1.grid.sum(['x','y'])
US_sumtors_ef2 = US_tors_ef2.grid.sum(['x','y'])
SE_sumtors_ef1 = SE_tors_ef1.grid.sum(['x','y'])
SE_sumtors_ef2 = SE_tors_ef2.grid.sum(['x','y'])

years_count = US_tors_ef1.grid.groupby('time.year').sum('time').year.values


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
    
'''
   
for i in xrange(len(file1[:,0])):
    
    hi_thresh1[i] = np.divide(np.nanpercentile(file1[i,:], 97.5), np.sum(Gauss_SmoothAN1))
    hi_thresh2[i] = np.divide(np.nanpercentile(file2[i,:], 97.5), np.sum(Gauss_SmoothAN2))
    hi_thresh3[i] = np.divide(np.nanpercentile(file3[i,:], 97.5), np.sum(Gauss_SmoothAN3))
    hi_thresh4[i] = np.divide(np.nanpercentile(file4[i,:], 97.5), np.sum(Gauss_SmoothAN4))

    lo_thresh1[i] = np.divide(np.nanpercentile(file1[i,:], 2.5), np.sum(Gauss_SmoothAN1))
    lo_thresh2[i] = np.divide(np.nanpercentile(file2[i,:], 2.5), np.sum(Gauss_SmoothAN2))
    lo_thresh3[i] = np.divide(np.nanpercentile(file3[i,:], 2.5), np.sum(Gauss_SmoothAN3))
    lo_thresh4[i] = np.divide(np.nanpercentile(file4[i,:], 2.5), np.sum(Gauss_SmoothAN4))
''' 

###############################################################################
###############################################################################
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

p1, = ax1.plot(range(0,total_days),Gauss_SmoothTELN1/np.sum(Gauss_SmoothTELN1),'r-',linewidth=2.0)
p2, = ax1.plot(range(0,total_days),Gauss_SmoothTLN1/np.sum(Gauss_SmoothTLN1),'b-',linewidth=2.0)
p3, = ax1.plot(range(0,total_days),Gauss_SmoothTN1/np.sum(Gauss_SmoothTN1),'k-',linewidth=2.0)    
#p4, = ax1.plot(range(0,total_days),Gauss_SmoothAN1/np.sum(Gauss_SmoothAN1),'--',color='grey',linewidth=2.0)  

p5 = ax1.fill_between(range(0,total_days),lo_thresh1,hi_thresh1,color='grey',linewidth=1.0,alpha=0.4)

ax1.set_ylabel('Fraction of Tornado Reports (EF1+)', fontsize=10)

ax1.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
plt.setp(ax1.get_yticklabels(), fontsize=10, rotation=35)
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_title('a) Annual Cycle of CONUS EF1+ Tornado Reports')

ax1.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3,grey_patch],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "IQR"],
                    loc="upper right",
                    fancybox=True, fontsize=12)



ax2 = fig.add_axes([0.0, 0.5, 0.95, 0.225]) 

p1, = ax2.plot(range(0,total_days),Gauss_SmoothTELN2/np.sum(Gauss_SmoothTELN2),'r-',linewidth=2.0)
p2, = ax2.plot(range(0,total_days),Gauss_SmoothTLN2/np.sum(Gauss_SmoothTLN2),'b-',linewidth=2.0)
p3, = ax2.plot(range(0,total_days),Gauss_SmoothTN2/np.sum(Gauss_SmoothTN2),'k-',linewidth=2.0)    
#p4, = ax2.plot(range(0,total_days),Gauss_SmoothAN2/np.sum(Gauss_SmoothAN2),'--',color='grey',linewidth=2.0)  

p5 = ax2.fill_between(range(0,total_days),lo_thresh2,hi_thresh2,color='grey',linewidth=1.0,alpha=0.4)

ax2.set_ylabel('Fraction of Tornado Reports (EF2+)', fontsize=10)

#ax2.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
ax2.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010])
plt.setp(ax2.get_yticklabels(), fontsize=10, rotation=35)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax2.set_title('b) Annual Cycle of CONUS EF2+ Tornado Reports')

ax2.grid(True, linestyle='--', alpha=0.5)




ax3 = fig.add_axes([0.0, 0.25, 0.95, 0.225]) 

p1, = ax3.plot(range(0,total_days),Gauss_SmoothTELN3/np.sum(Gauss_SmoothTELN3),'r-',linewidth=2.0)
p2, = ax3.plot(range(0,total_days),Gauss_SmoothTLN3/np.sum(Gauss_SmoothTLN3),'b-',linewidth=2.0)
p3, = ax3.plot(range(0,total_days),Gauss_SmoothTN3/np.sum(Gauss_SmoothTN3),'k-',linewidth=2.0)    
#p4, = ax3.plot(range(0,total_days),Gauss_SmoothAN3/np.sum(Gauss_SmoothAN3),'--',color='grey',linewidth=2.0)  

p5 = ax3.fill_between(range(0,total_days),lo_thresh3,hi_thresh3,color='grey',linewidth=1.0,alpha=0.4)

ax3.set_ylabel('Fraction of Tornado Reports (EF1+)', fontsize=10)

ax3.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
plt.setp(ax3.get_yticklabels(), fontsize=10, rotation=35)
#ax3.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax3.set_title('c) Annual Cycle of Southeast EF1+ Tornado Reports')

ax3.grid(True, linestyle='--', alpha=0.5)





ax4 = fig.add_axes([0.0, 0.0, 0.95, 0.225]) 

p1, = ax4.plot(range(0,total_days),Gauss_SmoothTELN4/np.sum(Gauss_SmoothTELN4),'r-',linewidth=2.0)
p2, = ax4.plot(range(0,total_days),Gauss_SmoothTLN4/np.sum(Gauss_SmoothTLN4),'b-',linewidth=2.0)
p3, = ax4.plot(range(0,total_days),Gauss_SmoothTN4/np.sum(Gauss_SmoothTN4),'k-',linewidth=2.0)    
#p4, = ax4.plot(range(0,total_days),Gauss_SmoothAN4/np.sum(Gauss_SmoothAN4),'--',color='grey',linewidth=2.0)  

p5 = ax4.fill_between(range(0,total_days),lo_thresh4,hi_thresh4,color='grey',linewidth=1.0,alpha=0.4)

ax4.set_ylabel('Fraction of Tornado Reports (EF2+)', fontsize=10)
ax4.set_xlabel('Day of Year', fontsize=10)

#ax4.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008])
ax4.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009])
plt.setp(ax4.get_yticklabels(), fontsize=10, rotation=35)
#ax4.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax4.set_xticks(tick_locs) 
ax4.set_xticklabels(tick_lbls)
ax4.set_title('d) Annual Cycle of Southeast EF2+ Tornado Reports')

ax4.grid(True, linestyle='--', alpha=0.5)



plt.savefig('wut_5.png', bbox_inches='tight', dpi=200)

#plt.show()


###############################################################################
###############################################################################
###############################################################################

'''
US_tors_ef1 = xr.open_dataset('tors_5316_ef1_obs', decode_cf=True)
US_tors_ef2 = xr.open_dataset('tors_5316_ef2_obs', decode_cf=True)
SE_tors_ef1 = xr.open_dataset('tors_5316_ef1_SE_obs', decode_cf=True)
SE_tors_ef2 = xr.open_dataset('tors_5316_ef2_SE_obs', decode_cf=True)

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
