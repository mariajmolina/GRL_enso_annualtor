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
#import pandas as pd
#from scipy.ndimage import gaussian_filter as gfilt
import matplotlib.patches as mpatches
import pickle
from itertools import product
import shapefile
from mpl_toolkits.basemap import Basemap
from shapely.geometry import MultiPoint, Point, Polygon
from matplotlib import ticker


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
HistoA_US1, _ = np.histogram(Julianday_all, bins = total_days)
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
HistoN_US1, _ = np.histogram(Julianday_neutral, bins = total_days)

Julianday_nino = [int(i) for i in Julianday_nino]
HistoEN_US1, _ = np.histogram(Julianday_nino, bins = total_days)

Julianday_nina = [int(i) for i in Julianday_nina]
HistoLN_US1, _ = np.histogram(Julianday_nina, bins = total_days)


###############################################################################
###############################################################################
###############################################################################

 
days_to_loop = np.cumsum(day_list.astype(int))

days_to_loop_1 = np.hstack([0,days_to_loop[:-1]])


HistoA_US_1 = [HistoA_US1[i:j]/year_all[0] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoN_US_1 = [HistoN_US1[i:j]/year_neutral[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoEN_US_1 = [HistoEN_US1[i:j]/year_nino[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoLN_US_1 = [HistoLN_US1[i:j]/year_nina[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]

HistoA_US_1 = np.array([item for sublist in HistoA_US_1 for item in sublist])
HistoN_US_1 = np.array([item for sublist in HistoN_US_1 for item in sublist])
HistoEN_US_1 = np.array([item for sublist in HistoEN_US_1 for item in sublist])
HistoLN_US_1 = np.array([item for sublist in HistoLN_US_1 for item in sublist])  


###############################################################################
###############################################################################
###############################################################################


with open('HistoA_US1', 'wb') as output:
    pickle.dump(HistoA_US_1, output, pickle.HIGHEST_PROTOCOL)

with open('HistoN_US1', 'wb') as output:
    pickle.dump(HistoN_US_1, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoEN_US1', 'wb') as output:
    pickle.dump(HistoEN_US_1, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoLN_US1', 'wb') as output:
    pickle.dump(HistoLN_US_1, output, pickle.HIGHEST_PROTOCOL)
    

###############################################################################
###############################################################################
###############################################################################
    

with open('HistoA_US1', 'rb') as f:
    HistoA_US1 = pickle.load(f)    
    
with open('HistoN_US1', 'rb') as f:
    HistoN_US1 = pickle.load(f)

with open('HistoEN_US1', 'rb') as f:
    HistoEN_US1 = pickle.load(f)
    
with open('HistoLN_US1', 'rb') as f:
    HistoLN_US1 = pickle.load(f)
    

###############################################################################
###############################################################################
###############################################################################


HistoA_US1_25 = np.divide(HistoA_US1, 4.)
HistoA_US1_75 = np.multiply(HistoA_US1, 3.)    

HistoN_US1_cs = np.cumsum(HistoN_US1)
HistoEN_US1_cs = np.cumsum(HistoEN_US1)
HistoLN_US1_cs = np.cumsum(HistoLN_US1)

HistoA_US1_25_cs = np.cumsum(HistoA_US1_25)
HistoA_US1_75_cs = np.cumsum(HistoA_US1_75)


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
HistoA_US2, _ = np.histogram(Julianday_all, bins = total_days)
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
HistoN_US2, _ = np.histogram(Julianday_neutral, bins = total_days)

Julianday_nino = [int(i) for i in Julianday_nino]
HistoEN_US2, _ = np.histogram(Julianday_nino, bins = total_days)

Julianday_nina = [int(i) for i in Julianday_nina]
HistoLN_US2, _ = np.histogram(Julianday_nina, bins = total_days)


###############################################################################
###############################################################################
###############################################################################

 
days_to_loop = np.cumsum(day_list.astype(int))

days_to_loop_1 = np.hstack([0,days_to_loop[:-1]])


HistoA_US_2 = [HistoA_US2[i:j]/year_all[0] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoN_US_2 = [HistoN_US2[i:j]/year_neutral[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoEN_US_2 = [HistoEN_US2[i:j]/year_nino[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoLN_US_2 = [HistoLN_US2[i:j]/year_nina[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]

HistoA_US_2 = np.array([item for sublist in HistoA_US_2 for item in sublist])
HistoN_US_2 = np.array([item for sublist in HistoN_US_2 for item in sublist])
HistoEN_US_2 = np.array([item for sublist in HistoEN_US_2 for item in sublist])
HistoLN_US_2 = np.array([item for sublist in HistoLN_US_2 for item in sublist])


###############################################################################
###############################################################################
###############################################################################


with open('HistoA_US2', 'wb') as output:
    pickle.dump(HistoA_US_2, output, pickle.HIGHEST_PROTOCOL)

with open('HistoN_US2', 'wb') as output:
    pickle.dump(HistoN_US_2, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoEN_US2', 'wb') as output:
    pickle.dump(HistoEN_US_2, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoLN_US2', 'wb') as output:
    pickle.dump(HistoLN_US_2, output, pickle.HIGHEST_PROTOCOL)
    

###############################################################################
###############################################################################
###############################################################################
    

with open('HistoA_US2', 'rb') as f:
    HistoA_US2 = pickle.load(f)    
    
with open('HistoN_US2', 'rb') as f:
    HistoN_US2 = pickle.load(f)

with open('HistoEN_US2', 'rb') as f:
    HistoEN_US2 = pickle.load(f)
    
with open('HistoLN_US2', 'rb') as f:
    HistoLN_US2 = pickle.load(f)
    

###############################################################################
###############################################################################
###############################################################################


HistoA_US2_25 = np.divide(HistoA_US2, 4.)
HistoA_US2_75 = np.multiply(HistoA_US2_25, 3.)

HistoN_US2_cs = np.cumsum(HistoN_US2)
HistoEN_US2_cs = np.cumsum(HistoEN_US2)
HistoLN_US2_cs = np.cumsum(HistoLN_US2)

HistoA_US2_25_cs = np.cumsum(HistoA_US2_25)
HistoA_US2_75_cs = np.cumsum(HistoA_US2_75)


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
HistoA_SE1, _ = np.histogram(Julianday_all, bins = total_days)
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
HistoN_SE1, _ = np.histogram(Julianday_neutral, bins = total_days)

Julianday_nino = [int(i) for i in Julianday_nino]
HistoEN_SE1, _ = np.histogram(Julianday_nino, bins = total_days)

Julianday_nina = [int(i) for i in Julianday_nina]
HistoLN_SE1, _ = np.histogram(Julianday_nina, bins = total_days)


###############################################################################
###############################################################################
###############################################################################

 
days_to_loop = np.cumsum(day_list.astype(int))

days_to_loop_1 = np.hstack([0,days_to_loop[:-1]])


HistoA_SE_1 = [HistoA_SE1[i:j]/year_all[0] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoN_SE_1 = [HistoN_SE1[i:j]/year_neutral[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoEN_SE_1 = [HistoEN_SE1[i:j]/year_nino[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoLN_SE_1 = [HistoLN_SE1[i:j]/year_nina[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]

HistoA_SE_1 = np.array([item for sublist in HistoA_SE_1 for item in sublist])
HistoN_SE_1 = np.array([item for sublist in HistoN_SE_1 for item in sublist])
HistoEN_SE_1 = np.array([item for sublist in HistoEN_SE_1 for item in sublist])
HistoLN_SE_1 = np.array([item for sublist in HistoLN_SE_1 for item in sublist])


###############################################################################
###############################################################################
###############################################################################


with open('HistoA_SE1', 'wb') as output:
    pickle.dump(HistoA_SE_1, output, pickle.HIGHEST_PROTOCOL)

with open('HistoN_SE1', 'wb') as output:
    pickle.dump(HistoN_SE_1, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoEN_SE1', 'wb') as output:
    pickle.dump(HistoEN_SE_1, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoLN_SE1', 'wb') as output:
    pickle.dump(HistoLN_SE_1, output, pickle.HIGHEST_PROTOCOL)
    

###############################################################################
###############################################################################
###############################################################################
    

with open('HistoA_SE1', 'rb') as f:
    HistoA_SE1 = pickle.load(f)    
    
with open('HistoN_SE1', 'rb') as f:
    HistoN_SE1 = pickle.load(f)

with open('HistoEN_SE1', 'rb') as f:
    HistoEN_SE1 = pickle.load(f)
    
with open('HistoLN_SE1', 'rb') as f:
    HistoLN_SE1 = pickle.load(f)
    

###############################################################################
###############################################################################
###############################################################################


HistoA_SE1_25 = np.divide(HistoA_SE1, 4.)
HistoA_SE1_75 = np.multiply(HistoA_SE1, 3.)    
    
HistoN_SE1_cs = np.cumsum(HistoN_SE1)
HistoEN_SE1_cs = np.cumsum(HistoEN_SE1)
HistoLN_SE1_cs = np.cumsum(HistoLN_SE1)

HistoA_SE1_25_cs = np.cumsum(HistoA_SE1_25)
HistoA_SE1_75_cs = np.cumsum(HistoA_SE1_75)


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
HistoA_SE2, _ = np.histogram(Julianday_all, bins = total_days)
    
Julianday_neutral = [int(i) for i in Julianday_neutral]
HistoN_SE2, _ = np.histogram(Julianday_neutral, bins = total_days)

Julianday_nino = [int(i) for i in Julianday_nino]
HistoEN_SE2, _ = np.histogram(Julianday_nino, bins = total_days)

Julianday_nina = [int(i) for i in Julianday_nina]
HistoLN_SE2, _ = np.histogram(Julianday_nina, bins = total_days)


###############################################################################
###############################################################################

for i, j, k in zip(year_neutral.values(),year_nina.values(),year_nino.values()):
    
    print i+j+k


###############################################################################


days_to_loop = np.cumsum(day_list.astype(int))

days_to_loop_1 = np.hstack([0,days_to_loop[:-1]])


HistoA_SE_2 = [HistoA_SE2[i:j]/year_all[0] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoN_SE_2 = [HistoN_SE2[i:j]/year_neutral[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoEN_SE_2 = [HistoEN_SE2[i:j]/year_nino[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]
HistoLN_SE_2 = [HistoLN_SE2[i:j]/year_nina[k] for k, (i, j) in enumerate(zip(days_to_loop_1, days_to_loop))]

HistoA_SE_2 = np.array([item for sublist in HistoA_SE_2 for item in sublist])
HistoN_SE_2 = np.array([item for sublist in HistoN_SE_2 for item in sublist])
HistoEN_SE_2 = np.array([item for sublist in HistoEN_SE_2 for item in sublist])
HistoLN_SE_2 = np.array([item for sublist in HistoLN_SE_2 for item in sublist])


###############################################################################
###############################################################################
###############################################################################


with open('HistoA_SE2', 'wb') as output:
    pickle.dump(HistoA_SE_2, output, pickle.HIGHEST_PROTOCOL)

with open('HistoN_SE2', 'wb') as output:
    pickle.dump(HistoN_SE_2, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoEN_SE2', 'wb') as output:
    pickle.dump(HistoEN_SE_2, output, pickle.HIGHEST_PROTOCOL)
    
with open('HistoLN_SE2', 'wb') as output:
    pickle.dump(HistoLN_SE_2, output, pickle.HIGHEST_PROTOCOL)
    

###############################################################################
###############################################################################
###############################################################################
    

with open('HistoA_SE2', 'rb') as f:
    HistoA_SE2 = pickle.load(f)    
    
with open('HistoN_SE2', 'rb') as f:
    HistoN_SE2 = pickle.load(f)

with open('HistoEN_SE2', 'rb') as f:
    HistoEN_SE2 = pickle.load(f)
    
with open('HistoLN_SE2', 'rb') as f:
    HistoLN_SE2 = pickle.load(f)
    

###############################################################################
###############################################################################
###############################################################################


HistoA_SE2_25 = np.divide(HistoA_SE2, 4.)
HistoA_SE2_75 = np.multiply(HistoA_SE2, 3.)    

HistoN_SE2_cs = np.cumsum(HistoN_SE2)
HistoEN_SE2_cs = np.cumsum(HistoEN_SE2)
HistoLN_SE2_cs = np.cumsum(HistoLN_SE2)

HistoA_SE2_25_cs = np.cumsum(HistoA_SE2_25)
HistoA_SE2_75_cs = np.cumsum(HistoA_SE2_75)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


data_neut = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_neutral', decode_cf=True)
data_nino = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_nino', decode_cf=True)
data_nina = xr.open_dataset('/storage/timme1mj/maria_pysplit/stphrs_nina', decode_cf=True)

stpmask_neut = data_neut.grid.sum('dayofyear')
stpmask_nino = data_nino.grid.sum('dayofyear')
stpmask_nina = data_nina.grid.sum('dayofyear')

data_neut = data_neut.grid
data_nino = data_nino.grid
data_nina = data_nina.grid


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



for i, j in product(xrange(len(data_neut[0,:,0])),xrange(len(data_neut[0,0,:]))):
    
    y = y1[i,j]
    x = x1[i,j]
    
    if not m.is_land(x,y):
        
        data_neut[:,i,j] = None
        data_nino[:,i,j] = None
        data_nina[:,i,j] = None
        
    xpt, ypt = m(x,y,inverse=True)
    
    if not in_us(ypt, xpt):

        data_neut[:,i,j] = None
        data_nino[:,i,j] = None
        data_nina[:,i,j] = None
        
    if np.all(np.isfinite(data_neut[:,i,j])) and ypt > 50.0:
        
        data_neut[:,i,j] = None
        
    if np.all(np.isfinite(data_nino[:,i,j])) and ypt > 50.0:
        
        data_nino[:,i,j] = None
        
    if np.all(np.isfinite(data_nina[:,i,j])) and ypt > 50.0:
        
        data_nina[:,i,j] = None
    
    if stpmask_neut[i,j] <= 2.5:
        
        data_neut[:,i,j] = None        

    if stpmask_nino[i,j] <= 2.5:
        
        data_nino[:,i,j] = None   
        
    if stpmask_nina[i,j] <= 2.5:
        
        data_nina[:,i,j] = None  


np.count_nonzero(~np.isnan(data_neut[0,:,:]))

data_neut2 = data_neut.sum(['x','y'], skipna=True)
data_nino2 = data_nino.sum(['x','y'], skipna=True)
data_nina2 = data_nina.sum(['x','y'], skipna=True)

cumstp_neut = data_neut2.cumsum('dayofyear', skipna=True)
cumstp_nino = data_nino2.cumsum('dayofyear', skipna=True)
cumstp_nina = data_nina2.cumsum('dayofyear', skipna=True)

cumstp_neut2 = np.divide(cumstp_neut,8)
cumstp_nino2 = np.divide(cumstp_nino,8)
cumstp_nina2 = np.divide(cumstp_nina,8)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



def myticks(x,pos):

    if x == 0: return "$0$"

    exponent = int(np.log10(x))
    coeff = x/10**exponent

    return r"${:2.0f} \times 10^{{ {:2d} }}$".format(coeff,exponent)



fig = plt.figure(figsize=(8,12))


ax1 = fig.add_axes([0.0, 0.8, 0.95, 0.175]) 

p1, = ax1.plot(range(0,total_days),HistoEN_US1_cs,'r-',linewidth=2.0)
p2, = ax1.plot(range(0,total_days),HistoLN_US1_cs,'b-',linewidth=2.0)
p3, = ax1.plot(range(0,total_days),HistoN_US1_cs,'k-',linewidth=2.0)    
#p4, = ax1.plot(range(0,total_days),Gauss_SmoothAN1/np.sum(Gauss_SmoothAN1),'--',color='grey',linewidth=2.0)  

#p5 = ax1.fill_between(range(0,total_days),HistoA_US1_25_cs,HistoA_US1_75_cs,color='grey',linewidth=1.0,alpha=0.4)

ax1.set_ylabel('Tornado Reports (EF1+)', fontsize=10)

ax1.set_yticks([0,100,200,300,400,500,600])
plt.setp(ax1.get_yticklabels(), fontsize=10)
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_title('a) Mean Cumulative CONUS EF1+ Tornado Reports')

ax1.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral"],
                    loc="upper left",
                    fancybox=True, fontsize=12)

#plt.show()


ax2 = fig.add_axes([0.0, 0.6, 0.95, 0.175]) 

p1, = ax2.plot(range(0,total_days),HistoEN_US2_cs,'r-',linewidth=2.0)
p2, = ax2.plot(range(0,total_days),HistoLN_US2_cs,'b-',linewidth=2.0)
p3, = ax2.plot(range(0,total_days),HistoN_US2_cs,'k-',linewidth=2.0)    
#p4, = ax2.plot(range(0,total_days),Gauss_SmoothAN2/np.sum(Gauss_SmoothAN2),'--',color='grey',linewidth=2.0)  

#p5 = ax2.fill_between(range(0,total_days),lo_thresh2,hi_thresh2,color='grey',linewidth=1.0,alpha=0.4)

ax2.set_ylabel('Tornado Reports (EF2+)', fontsize=10)

ax2.set_yticks([0,25,50,75,100,125,150,175,200,225])
#ax2.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010])
plt.setp(ax2.get_yticklabels(), fontsize=10)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax2.set_title('b) Mean Cumulative CONUS EF2+ Tornado Reports')

ax2.grid(True, linestyle='--', alpha=0.5)




ax3 = fig.add_axes([0.0, 0.4, 0.95, 0.175]) 

p1, = ax3.plot(range(0,total_days),HistoEN_SE1_cs,'r-',linewidth=2.0)
p2, = ax3.plot(range(0,total_days),HistoLN_SE1_cs,'b-',linewidth=2.0)
p3, = ax3.plot(range(0,total_days),HistoN_SE1_cs,'k-',linewidth=2.0)    
#p4, = ax3.plot(range(0,total_days),Gauss_SmoothAN3/np.sum(Gauss_SmoothAN3),'--',color='grey',linewidth=2.0)  

#p5 = ax3.fill_between(range(0,total_days),lo_thresh3,hi_thresh3,color='grey',linewidth=1.0,alpha=0.4)

ax3.set_ylabel('Tornado Reports (EF1+)', fontsize=10)

ax3.set_yticks([0,25,50,75,100,125,150,175,200])
plt.setp(ax3.get_yticklabels(), fontsize=10)
#ax3.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax3.set_title('c) Mean Cumulative Southeast EF1+ Tornado Reports')

ax3.grid(True, linestyle='--', alpha=0.5)





ax4 = fig.add_axes([0.0, 0.2, 0.95, 0.175]) 

p1, = ax4.plot(range(0,total_days),HistoEN_SE2_cs,'r-',linewidth=2.0)
p2, = ax4.plot(range(0,total_days),HistoLN_SE2_cs,'b-',linewidth=2.0)
p3, = ax4.plot(range(0,total_days),HistoN_SE2_cs,'k-',linewidth=2.0)    
#p4, = ax4.plot(range(0,total_days),Gauss_SmoothAN4/np.sum(Gauss_SmoothAN4),'--',color='grey',linewidth=2.0)  

#p5 = ax4.fill_between(range(0,total_days),lo_thresh4,hi_thresh4,color='grey',linewidth=1.0,alpha=0.4)

ax4.set_ylabel('Tornado Reports (EF2+)', fontsize=10)
#ax4.set_xlabel('Day of Year', fontsize=10)

ax4.set_yticks([0,10,20,30,40,50,60,70,80])
#ax4.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009])
plt.setp(ax4.get_yticklabels(), fontsize=10)
#ax4.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))


ax4.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax4.set_title('d) Mean Cumulative Southeast EF2+ Tornado Reports')

ax4.grid(True, linestyle='--', alpha=0.5)




ax5 = fig.add_axes([0.0, 0.0, 0.95, 0.175]) 

p1, = ax5.plot(range(0,total_days),cumstp_nino2[:-1],'r-',linewidth=2.0)
p2, = ax5.plot(range(0,total_days),cumstp_nina2[:-1],'b-',linewidth=2.0)
p3, = ax5.plot(range(0,total_days),cumstp_neut2[:-1],'k-',linewidth=2.0)    
#p4, = ax4.plot(range(0,total_days),Gauss_SmoothAN4/np.sum(Gauss_SmoothAN4),'--',color='grey',linewidth=2.0)  

#p5 = ax4.fill_between(range(0,total_days),lo_thresh4,hi_thresh4,color='grey',linewidth=1.0,alpha=0.4)

ax5.set_ylabel('Mean STP 3-Hours', fontsize=10)
ax5.set_xlabel('Day of Year', fontsize=10)

ax5.set_yticks([0,2500,5000,7500,10000,12500,15000,17500,20000,22500])
#ax4.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009])
plt.setp(ax5.get_yticklabels(), fontsize=10)
#ax5.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax5.set_xticks(tick_locs) 
ax5.set_xticklabels(tick_lbls)
ax5.set_title('e) Mean Cumulative STP 3-Hours')

ax5.grid(True, linestyle='--', alpha=0.5)


plt.savefig('wut_5.png', bbox_inches='tight', dpi=200)

#plt.show()


###############################################################################
###############################################################################
###############################################################################

