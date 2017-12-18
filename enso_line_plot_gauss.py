#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 11:14:25 2017

Maria J. Molina
Ph.D. Student
Central Michigan University

"""

###############################################################################
###############################################################################
###############################################################################


from __future__ import division

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from scipy.ndimage import gaussian_filter as gfilt
from datetime import timedelta


###############################################################################
###############################################################################
###############################################################################


severe = pd.read_csv('/storage/timme1mj/maria_pysplit/1950-2016_actual_tornadoes.csv', error_bad_lines=False,
                     parse_dates=[['mo','dy','yr','time']])

severe = severe.assign(UTC_time = severe['mo_dy_yr_time'] + timedelta(hours=6))       
severe = severe.assign(UTC_yr = severe['UTC_time'].dt.year) 
severe = severe.assign(UTC_dy = severe['UTC_time'].dt.day) 
severe = severe.assign(UTC_mo = severe['UTC_time'].dt.month) 
severe = severe.assign(UTC_hr = severe['UTC_time'].dt.hour)   
        

###############################################################################
############EF1################################################################
###############################################################################


severe = severe[severe['mag']>=1]

severe_1953 = severe[((severe['UTC_mo'] >= 6) & (severe['UTC_dy'] >= 1) & (severe['UTC_yr'] == 1953))]

severe_2016 = severe[((severe['UTC_mo'] <= 5) & (severe['UTC_dy'] <= 31) & (severe['UTC_yr'] == 2016))]

severe_5415 = severe[(severe['UTC_yr'] >= 1954) & (severe['UTC_yr'] <= 2015)]

severe_cut = pd.concat([severe_1953, severe_5415, severe_2016])


###############################################################################
###############################################################################
###############################################################################


sev_us = severe_cut[severe_cut['slat']!=0]
sev_us = sev_us[sev_us['slon']!=0]
sev_us = sev_us[sev_us['slat']>=20]
sev_us = sev_us[sev_us['slat']<=50]
sev_us = sev_us[sev_us['slon']>=-130]
sev_us = sev_us[sev_us['slon']<=-65]
            
            
sev_se = severe_cut[severe_cut['slat']!=0]
sev_se = sev_se[sev_se['slon']!=0]
sev_se = sev_se[sev_se['slat']>=25]
sev_se = sev_se[sev_se['slat']<=37]
sev_se = sev_se[sev_se['slon']>=-93]
sev_se = sev_se[sev_se['slon']<=-70]  
            

###############################################################################
###############################################################################
###############################################################################


N = [1956,1959,1960,1961,1962,1966,1967,1978,1980,1981,1985,1989,1990,1992,1993,1996,2001,2003,2012,2013,1979]

EN = [1953,1957,1958,1963,1965,1968,1969,1972,1976,1977,1982,1986,1987,1991,1994,1997,2002,2004,2006,2009,2014,2015]

LN = [1954,1955,1964,1970,1971,1973,1974,1975,1983,1984,1988,1995,1998,1999,2000,2005,2007,2008,2010,2011]


N_two = [x+1 for x in N]

EN_two = [x+1 for x in EN]

LN_two = [x+1 for x in LN]


###############################################################################
###############################################################################
###############################################################################


sev_us_one = sev_us[sev_us['UTC_mo'] >= 6]

sev_us_two = sev_us[sev_us['UTC_mo'] <= 5]


sev_se_one = sev_se[sev_se['UTC_mo'] >= 6]

sev_se_two = sev_se[sev_se['UTC_mo'] <= 5]


###############################################################################
###############################################################################
###############################################################################


J_US_EN = {}
J_SE_EN = {}
    
for z, (i, j) in enumerate(zip(EN, EN_two)):
    
    To_us = sev_us_one[sev_us_one['UTC_yr']==i]    
    yr_us = np.array(To_us['UTC_yr'])
    mo_us = np.array(To_us['UTC_mo'])
    dy_us = np.array(To_us['UTC_dy'])

    To_us_two = sev_us_two[sev_us_two['UTC_yr']==j]    
    yr_us_two = np.array(To_us_two['UTC_yr'])
    mo_us_two = np.array(To_us_two['UTC_mo'])
    dy_us_two = np.array(To_us_two['UTC_dy'])

    To_se = sev_se_one[sev_se_one['UTC_yr']==i]    
    yr_se = np.array(To_se['UTC_yr'])
    mo_se = np.array(To_se['UTC_mo'])
    dy_se = np.array(To_se['UTC_dy'])

    To_se_two = sev_se_two[sev_se_two['UTC_yr']==j]    
    yr_se_two = np.array(To_se_two['UTC_yr'])
    mo_se_two = np.array(To_se_two['UTC_mo'])
    dy_se_two = np.array(To_se_two['UTC_dy'])

    j_us = []
    j_us_two = []
    j_se = []
    j_se_two = []

    for n in range(0,len(yr_us)-1):
        
        Date = dt.datetime(yr_us[n],mo_us[n],dy_us[n])
        j_us.append(Date.strftime('%j'))

    for n in range(0,len(yr_us_two)-1):
        
        Date = dt.datetime(yr_us_two[n],mo_us_two[n],dy_us_two[n])
        j_us_two.append(Date.strftime('%j'))
       
    J_us = j_us + j_us_two

    J_us = [int(i) for i in J_us]
    Histo = np.histogram(J_us, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_US = Gauss_Smooth0[366:732]/len(EN)
        
    J_US_EN[z] = GS_US

    for n in range(0,len(yr_se)-1):

        Date = dt.datetime(yr_se[n],mo_se[n],dy_se[n])
        j_se.append(Date.strftime('%j'))

    for n in range(0,len(yr_se_two)-1):
        
        Date = dt.datetime(yr_se_two[n],mo_se_two[n],dy_se_two[n])
        j_se_two.append(Date.strftime('%j'))
        
    J_se = j_se + j_se_two

    J_se = [int(i) for i in J_se]
    Histo = np.histogram(J_se, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_SE = Gauss_Smooth0[366:732]/len(EN_two)

    J_SE_EN[z] = GS_SE
        

###############################################################################
    
        
J_US_LN = {}
J_SE_LN = {}
    
for z, (i, j) in enumerate(zip(LN, LN_two)):
    
    To_us = sev_us_one[sev_us_one['UTC_yr']==i]    
    yr_us = np.array(To_us['UTC_yr'])
    mo_us = np.array(To_us['UTC_mo'])
    dy_us = np.array(To_us['UTC_dy'])

    To_us_two = sev_us_two[sev_us_two['UTC_yr']==j]    
    yr_us_two = np.array(To_us_two['UTC_yr'])
    mo_us_two = np.array(To_us_two['UTC_mo'])
    dy_us_two = np.array(To_us_two['UTC_dy'])

    To_se = sev_se_one[sev_se_one['UTC_yr']==i]    
    yr_se = np.array(To_se['UTC_yr'])
    mo_se = np.array(To_se['UTC_mo'])
    dy_se = np.array(To_se['UTC_dy'])

    To_se_two = sev_se_two[sev_se_two['UTC_yr']==j]    
    yr_se_two = np.array(To_se_two['UTC_yr'])
    mo_se_two = np.array(To_se_two['UTC_mo'])
    dy_se_two = np.array(To_se_two['UTC_dy'])

    j_us = []
    j_us_two = []
    j_se = []
    j_se_two = []

    for n in range(0,len(yr_us)-1):
        
        Date = dt.datetime(yr_us[n],mo_us[n],dy_us[n])
        j_us.append(Date.strftime('%j'))

    for n in range(0,len(yr_us_two)-1):
        
        Date = dt.datetime(yr_us_two[n],mo_us_two[n],dy_us_two[n])
        j_us_two.append(Date.strftime('%j'))
       
    J_us = j_us + j_us_two

    J_us = [int(i) for i in J_us]
    Histo = np.histogram(J_us, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_US = Gauss_Smooth0[366:732]/len(LN)
        
    J_US_LN[z] = GS_US

    for n in range(0,len(yr_se)-1):

        Date = dt.datetime(yr_se[n],mo_se[n],dy_se[n])
        j_se.append(Date.strftime('%j'))

    for n in range(0,len(yr_se_two)-1):
        
        Date = dt.datetime(yr_se_two[n],mo_se_two[n],dy_se_two[n])
        j_se_two.append(Date.strftime('%j'))
        
    J_se = j_se + j_se_two

    J_se = [int(i) for i in J_se]
    Histo = np.histogram(J_se, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_SE = Gauss_Smooth0[366:732]/len(LN_two)

    J_SE_LN[z] = GS_SE


###############################################################################
###############################################################################
###############################################################################


Torn_us_N = sev_us_one[sev_us_one.UTC_yr.isin(N)]    
year_us_N = np.array(Torn_us_N['UTC_yr'])
month_us_N = np.array(Torn_us_N['UTC_mo'])
day_us_N = np.array(Torn_us_N['UTC_dy'])

Torn_us_EN = sev_us_one[sev_us_one.UTC_yr.isin(EN)]    
year_us_EN = np.array(Torn_us_EN['UTC_yr'])
month_us_EN = np.array(Torn_us_EN['UTC_mo'])
day_us_EN = np.array(Torn_us_EN['UTC_dy'])

Torn_us_LN = sev_us_one[sev_us_one.UTC_yr.isin(LN)]    
year_us_LN = np.array(Torn_us_LN['UTC_yr'])
month_us_LN = np.array(Torn_us_LN['UTC_mo'])
day_us_LN = np.array(Torn_us_LN['UTC_dy'])


Torn_us_N_two = sev_us_two[sev_us_two.UTC_yr.isin(N_two)]    
year_us_N_two = np.array(Torn_us_N_two['UTC_yr'])
month_us_N_two = np.array(Torn_us_N_two['UTC_mo'])
day_us_N_two = np.array(Torn_us_N_two['UTC_dy'])

Torn_us_EN_two = sev_us_two[sev_us_two.UTC_yr.isin(EN_two)]    
year_us_EN_two = np.array(Torn_us_EN_two['UTC_yr'])
month_us_EN_two = np.array(Torn_us_EN_two['UTC_mo'])
day_us_EN_two = np.array(Torn_us_EN_two['UTC_dy'])

Torn_us_LN_two = sev_us_two[sev_us_two.UTC_yr.isin(LN_two)]    
year_us_LN_two = np.array(Torn_us_LN_two['UTC_yr'])
month_us_LN_two = np.array(Torn_us_LN_two['UTC_mo'])
day_us_LN_two = np.array(Torn_us_LN_two['UTC_dy'])


###############################################################################


Torn_se_N = sev_se_one[sev_se_one.UTC_yr.isin(N)]    
year_se_N = np.array(Torn_se_N['UTC_yr'])
month_se_N = np.array(Torn_se_N['UTC_mo'])
day_se_N = np.array(Torn_se_N['UTC_dy'])

Torn_se_EN = sev_se_one[sev_se_one.UTC_yr.isin(EN)]    
year_se_EN = np.array(Torn_se_EN['UTC_yr'])
month_se_EN = np.array(Torn_se_EN['UTC_mo'])
day_se_EN = np.array(Torn_se_EN['UTC_dy'])

Torn_se_LN = sev_se_one[sev_se_one.UTC_yr.isin(LN)]    
year_se_LN = np.array(Torn_se_LN['UTC_yr'])
month_se_LN = np.array(Torn_se_LN['UTC_mo'])
day_se_LN = np.array(Torn_se_LN['UTC_dy'])


Torn_se_N_two = sev_se_two[sev_se_two.UTC_yr.isin(N_two)]    
year_se_N_two = np.array(Torn_se_N_two['UTC_yr'])
month_se_N_two = np.array(Torn_se_N_two['UTC_mo'])
day_se_N_two = np.array(Torn_se_N_two['UTC_dy'])

Torn_se_EN_two = sev_se_two[sev_se_two.UTC_yr.isin(EN_two)]    
year_se_EN_two = np.array(Torn_se_EN_two['UTC_yr'])
month_se_EN_two = np.array(Torn_se_EN_two['UTC_mo'])
day_se_EN_two = np.array(Torn_se_EN_two['UTC_dy'])

Torn_se_LN_two = sev_se_two[sev_se_two.UTC_yr.isin(LN_two)]    
year_se_LN_two = np.array(Torn_se_LN_two['UTC_yr'])
month_se_LN_two = np.array(Torn_se_LN_two['UTC_mo'])
day_se_LN_two = np.array(Torn_se_LN_two['UTC_dy'])


###############################################################################
###############################################################################
###############################################################################


years_all = pd.date_range('1953-06-01','2016-05-31',freq='AS-JUN').year


year_us_all = np.array(sev_us['UTC_yr'])
month_us_all = np.array(sev_us['UTC_mo'])
day_us_all = np.array(sev_us['UTC_dy'])


year_se_all = np.array(sev_se['UTC_yr'])
month_se_all = np.array(sev_se['UTC_mo'])
day_se_all = np.array(sev_se['UTC_dy'])


###############################################################################
###############################################################################
###############################################################################


julian_us_A = []
for n in range(0,len(year_us_all)-1):
   Date = dt.datetime(year_us_all[n],month_us_all[n],day_us_all[n])
   julian_us_A.append(Date.strftime('%j'))
   
julian_se_A = []
for n in range(0,len(year_se_all)-1):
   Date = dt.datetime(year_se_all[n],month_se_all[n],day_se_all[n])
   julian_se_A.append(Date.strftime('%j'))
   

###############################################################################

   
julian_us_N = []
for n in range(0,len(year_us_N)-1):
   Date = dt.datetime(year_us_N[n],month_us_N[n],day_us_N[n])
   julian_us_N.append(Date.strftime('%j'))

julian_us_EN = []
for n in range(0,len(year_us_EN)-1):
   Date = dt.datetime(year_us_EN[n],month_us_EN[n],day_us_EN[n])
   julian_us_EN.append(Date.strftime('%j'))

julian_us_LN = []
for n in range(0,len(year_us_LN)-1):
   Date = dt.datetime(year_us_LN[n],month_us_LN[n],day_us_LN[n])
   julian_us_LN.append(Date.strftime('%j'))
   

julian_us_N_two = []
for n in range(0,len(year_us_N_two)-1):
   Date = dt.datetime(year_us_N_two[n],month_us_N_two[n],day_us_N_two[n])
   julian_us_N_two.append(Date.strftime('%j'))

julian_us_EN_two = []
for n in range(0,len(year_us_EN_two)-1):
   Date = dt.datetime(year_us_EN_two[n],month_us_EN_two[n],day_us_EN_two[n])
   julian_us_EN_two.append(Date.strftime('%j'))

julian_us_LN_two = []
for n in range(0,len(year_us_LN_two)-1):
   Date = dt.datetime(year_us_LN_two[n],month_us_LN_two[n],day_us_LN_two[n])
   julian_us_LN_two.append(Date.strftime('%j'))
   
   
###############################################################################
   

julian_se_N = []
for n in range(0,len(year_se_N)-1):
   Date = dt.datetime(year_se_N[n],month_se_N[n],day_se_N[n])
   julian_se_N.append(Date.strftime('%j'))

julian_se_EN = []
for n in range(0,len(year_se_EN)-1):
   Date = dt.datetime(year_se_EN[n],month_se_EN[n],day_se_EN[n])
   julian_se_EN.append(Date.strftime('%j'))

julian_se_LN = []
for n in range(0,len(year_se_LN)-1):
   Date = dt.datetime(year_se_LN[n],month_se_LN[n],day_se_LN[n])
   julian_se_LN.append(Date.strftime('%j'))
   

julian_se_N_two = []
for n in range(0,len(year_se_N_two)-1):
   Date = dt.datetime(year_se_N_two[n],month_se_N_two[n],day_se_N_two[n])
   julian_se_N_two.append(Date.strftime('%j'))

julian_se_EN_two = []
for n in range(0,len(year_se_EN_two)-1):
   Date = dt.datetime(year_se_EN_two[n],month_se_EN_two[n],day_se_EN_two[n])
   julian_se_EN_two.append(Date.strftime('%j'))

julian_se_LN_two = []
for n in range(0,len(year_se_LN_two)-1):
   Date = dt.datetime(year_se_LN_two[n],month_se_LN_two[n],day_se_LN_two[n])
   julian_se_LN_two.append(Date.strftime('%j'))
   
   
###############################################################################
###############################################################################
###############################################################################
   
 
Julian_US_N = julian_us_N + julian_us_N_two
Julian_US_EN = julian_us_EN + julian_us_EN_two   
Julian_US_LN = julian_us_LN + julian_us_LN_two
   
Julian_SE_N = julian_se_N + julian_se_N_two
Julian_SE_EN = julian_se_EN + julian_se_EN_two   
Julian_SE_LN = julian_se_LN + julian_se_LN_two
   

###############################################################################
###############################################################################
###############################################################################


Julian_US_A = [int(i) for i in julian_us_A]
Histo = np.histogram(Julian_US_A, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_A = Gauss_Smooth0[366:732]/len(years_all)

Julian_SE_A = [int(i) for i in julian_se_A]
Histo = np.histogram(Julian_SE_A, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_A = Gauss_Smooth0[366:732]/len(years_all)


###############################################################################


Julian_US_N = [int(i) for i in Julian_US_N]
Histo = np.histogram(Julian_US_N, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_N = Gauss_Smooth0[366:732]/len(N)

Julian_US_EN = [int(i) for i in Julian_US_EN]
Histo = np.histogram(Julian_US_EN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_EN = Gauss_Smooth0[366:732]/len(EN)

Julian_US_LN = [int(i) for i in Julian_US_LN]
Histo = np.histogram(Julian_US_LN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_LN = Gauss_Smooth0[366:732]/len(LN)


###############################################################################


Julian_SE_N = [int(i) for i in Julian_SE_N]
Histo = np.histogram(Julian_SE_N, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_N = Gauss_Smooth0[366:732]/len(N)

Julian_SE_EN = [int(i) for i in Julian_SE_EN]
Histo = np.histogram(Julian_SE_EN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_EN = Gauss_Smooth0[366:732]/len(EN)

Julian_SE_LN = [int(i) for i in Julian_SE_LN]
Histo = np.histogram(Julian_SE_LN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_LN = Gauss_Smooth0[366:732]/len(LN)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
########################################EF2####################################
###############################################################################
###############################################################################
###############################################################################


severe = pd.read_csv('/storage/timme1mj/maria_pysplit/1950-2016_actual_tornadoes.csv', error_bad_lines=False,
                     parse_dates=[['mo','dy','yr','time']])

severe = severe.assign(UTC_time = severe['mo_dy_yr_time'] + timedelta(hours=6))       
severe = severe.assign(UTC_yr = severe['UTC_time'].dt.year) 
severe = severe.assign(UTC_dy = severe['UTC_time'].dt.day) 
severe = severe.assign(UTC_mo = severe['UTC_time'].dt.month) 
severe = severe.assign(UTC_hr = severe['UTC_time'].dt.hour)   
        

###############################################################################
###############################################################################
###############################################################################


severe = severe[severe['mag']>=2]

severe_1953 = severe[((severe['UTC_mo'] >= 6) & (severe['UTC_dy'] >= 1) & (severe['UTC_yr'] == 1953))]

severe_2016 = severe[((severe['UTC_mo'] <= 5) & (severe['UTC_dy'] <= 31) & (severe['UTC_yr'] == 2016))]

severe_5415 = severe[(severe['UTC_yr'] >= 1954) & (severe['UTC_yr'] <= 2015)]

severe_cut = pd.concat([severe_1953, severe_5415, severe_2016])


###############################################################################
###############################################################################
###############################################################################


sev_us = severe_cut[severe_cut['slat']!=0]
sev_us = sev_us[sev_us['slon']!=0]
sev_us = sev_us[sev_us['slat']>=20]
sev_us = sev_us[sev_us['slat']<=50]
sev_us = sev_us[sev_us['slon']>=-130]
sev_us = sev_us[sev_us['slon']<=-65]
            
            
sev_se = severe_cut[severe_cut['slat']!=0]
sev_se = sev_se[sev_se['slon']!=0]
sev_se = sev_se[sev_se['slat']>=25]
sev_se = sev_se[sev_se['slat']<=37]
sev_se = sev_se[sev_se['slon']>=-93]
sev_se = sev_se[sev_se['slon']<=-70]  
            

###############################################################################
###############################################################################
###############################################################################


N = [1956,1959,1960,1961,1962,1966,1967,1978,1980,1981,1985,1989,1990,1992,1993,1996,2001,2003,2012,2013,1979]

EN = [1953,1957,1958,1963,1965,1968,1969,1972,1976,1977,1982,1986,1987,1991,1994,1997,2002,2004,2006,2009,2014,2015]

LN = [1954,1955,1964,1970,1971,1973,1974,1975,1983,1984,1988,1995,1998,1999,2000,2005,2007,2008,2010,2011]


N_two = [x+1 for x in N]

EN_two = [x+1 for x in EN]

LN_two = [x+1 for x in LN]


###############################################################################
###############################################################################
###############################################################################


sev_us_one = sev_us[sev_us['UTC_mo'] >= 6]

sev_us_two = sev_us[sev_us['UTC_mo'] <= 5]


sev_se_one = sev_se[sev_se['UTC_mo'] >= 6]

sev_se_two = sev_se[sev_se['UTC_mo'] <= 5]


###############################################################################
###############################################################################
###############################################################################


J_US_EN_EF2 = {}
J_SE_EN_EF2 = {}
    
for z, (i, j) in enumerate(zip(EN, EN_two)):
    
    To_us = sev_us_one[sev_us_one['UTC_yr']==i]    
    yr_us = np.array(To_us['UTC_yr'])
    mo_us = np.array(To_us['UTC_mo'])
    dy_us = np.array(To_us['UTC_dy'])

    To_us_two = sev_us_two[sev_us_two['UTC_yr']==j]    
    yr_us_two = np.array(To_us_two['UTC_yr'])
    mo_us_two = np.array(To_us_two['UTC_mo'])
    dy_us_two = np.array(To_us_two['UTC_dy'])

    To_se = sev_se_one[sev_se_one['UTC_yr']==i]    
    yr_se = np.array(To_se['UTC_yr'])
    mo_se = np.array(To_se['UTC_mo'])
    dy_se = np.array(To_se['UTC_dy'])

    To_se_two = sev_se_two[sev_se_two['UTC_yr']==j]    
    yr_se_two = np.array(To_se_two['UTC_yr'])
    mo_se_two = np.array(To_se_two['UTC_mo'])
    dy_se_two = np.array(To_se_two['UTC_dy'])

    j_us = []
    j_us_two = []
    j_se = []
    j_se_two = []

    for n in range(0,len(yr_us)-1):
        
        Date = dt.datetime(yr_us[n],mo_us[n],dy_us[n])
        j_us.append(Date.strftime('%j'))

    for n in range(0,len(yr_us_two)-1):
        
        Date = dt.datetime(yr_us_two[n],mo_us_two[n],dy_us_two[n])
        j_us_two.append(Date.strftime('%j'))
       
    J_us = j_us + j_us_two

    J_us = [int(i) for i in J_us]
    Histo = np.histogram(J_us, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_US = Gauss_Smooth0[366:732]/len(EN)
        
    J_US_EN_EF2[z] = GS_US

    for n in range(0,len(yr_se)-1):

        Date = dt.datetime(yr_se[n],mo_se[n],dy_se[n])
        j_se.append(Date.strftime('%j'))

    for n in range(0,len(yr_se_two)-1):
        
        Date = dt.datetime(yr_se_two[n],mo_se_two[n],dy_se_two[n])
        j_se_two.append(Date.strftime('%j'))
        
    J_se = j_se + j_se_two

    J_se = [int(i) for i in J_se]
    Histo = np.histogram(J_se, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_SE = Gauss_Smooth0[366:732]/len(EN_two)

    J_SE_EN_EF2[z] = GS_SE
        

###############################################################################
    
        
J_US_LN_EF2 = {}
J_SE_LN_EF2 = {}
    
for z, (i, j) in enumerate(zip(LN, LN_two)):
    
    To_us = sev_us_one[sev_us_one['UTC_yr']==i]    
    yr_us = np.array(To_us['UTC_yr'])
    mo_us = np.array(To_us['UTC_mo'])
    dy_us = np.array(To_us['UTC_dy'])

    To_us_two = sev_us_two[sev_us_two['UTC_yr']==j]    
    yr_us_two = np.array(To_us_two['UTC_yr'])
    mo_us_two = np.array(To_us_two['UTC_mo'])
    dy_us_two = np.array(To_us_two['UTC_dy'])

    To_se = sev_se_one[sev_se_one['UTC_yr']==i]    
    yr_se = np.array(To_se['UTC_yr'])
    mo_se = np.array(To_se['UTC_mo'])
    dy_se = np.array(To_se['UTC_dy'])

    To_se_two = sev_se_two[sev_se_two['UTC_yr']==j]    
    yr_se_two = np.array(To_se_two['UTC_yr'])
    mo_se_two = np.array(To_se_two['UTC_mo'])
    dy_se_two = np.array(To_se_two['UTC_dy'])

    j_us = []
    j_us_two = []
    j_se = []
    j_se_two = []

    for n in range(0,len(yr_us)-1):
        
        Date = dt.datetime(yr_us[n],mo_us[n],dy_us[n])
        j_us.append(Date.strftime('%j'))

    for n in range(0,len(yr_us_two)-1):
        
        Date = dt.datetime(yr_us_two[n],mo_us_two[n],dy_us_two[n])
        j_us_two.append(Date.strftime('%j'))
       
    J_us = j_us + j_us_two

    J_us = [int(i) for i in J_us]
    Histo = np.histogram(J_us, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_US = Gauss_Smooth0[366:732]/len(LN)
        
    J_US_LN_EF2[z] = GS_US

    for n in range(0,len(yr_se)-1):

        Date = dt.datetime(yr_se[n],mo_se[n],dy_se[n])
        j_se.append(Date.strftime('%j'))

    for n in range(0,len(yr_se_two)-1):
        
        Date = dt.datetime(yr_se_two[n],mo_se_two[n],dy_se_two[n])
        j_se_two.append(Date.strftime('%j'))
        
    J_se = j_se + j_se_two

    J_se = [int(i) for i in J_se]
    Histo = np.histogram(J_se, bins=366)
    HistoN = np.zeros(366*3)
    HistoN[0:366]=Histo[0] 
    HistoN[366:732]=Histo[0]  
    HistoN[732:1101]=Histo[0]
    Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
    GS_SE = Gauss_Smooth0[366:732]/len(LN_two)

    J_SE_LN_EF2[z] = GS_SE


###############################################################################
###############################################################################
###############################################################################
    

Torn_us_N = sev_us_one[sev_us_one.UTC_yr.isin(N)]    
year_us_N = np.array(Torn_us_N['UTC_yr'])
month_us_N = np.array(Torn_us_N['UTC_mo'])
day_us_N = np.array(Torn_us_N['UTC_dy'])

Torn_us_EN = sev_us_one[sev_us_one.UTC_yr.isin(EN)]    
year_us_EN = np.array(Torn_us_EN['UTC_yr'])
month_us_EN = np.array(Torn_us_EN['UTC_mo'])
day_us_EN = np.array(Torn_us_EN['UTC_dy'])

Torn_us_LN = sev_us_one[sev_us_one.UTC_yr.isin(LN)]    
year_us_LN = np.array(Torn_us_LN['UTC_yr'])
month_us_LN = np.array(Torn_us_LN['UTC_mo'])
day_us_LN = np.array(Torn_us_LN['UTC_dy'])


Torn_us_N_two = sev_us_two[sev_us_two.UTC_yr.isin(N_two)]    
year_us_N_two = np.array(Torn_us_N_two['UTC_yr'])
month_us_N_two = np.array(Torn_us_N_two['UTC_mo'])
day_us_N_two = np.array(Torn_us_N_two['UTC_dy'])

Torn_us_EN_two = sev_us_two[sev_us_two.UTC_yr.isin(EN_two)]    
year_us_EN_two = np.array(Torn_us_EN_two['UTC_yr'])
month_us_EN_two = np.array(Torn_us_EN_two['UTC_mo'])
day_us_EN_two = np.array(Torn_us_EN_two['UTC_dy'])

Torn_us_LN_two = sev_us_two[sev_us_two.UTC_yr.isin(LN_two)]    
year_us_LN_two = np.array(Torn_us_LN_two['UTC_yr'])
month_us_LN_two = np.array(Torn_us_LN_two['UTC_mo'])
day_us_LN_two = np.array(Torn_us_LN_two['UTC_dy'])


###############################################################################


Torn_se_N = sev_se_one[sev_se_one.UTC_yr.isin(N)]    
year_se_N = np.array(Torn_se_N['UTC_yr'])
month_se_N = np.array(Torn_se_N['UTC_mo'])
day_se_N = np.array(Torn_se_N['UTC_dy'])

Torn_se_EN = sev_se_one[sev_se_one.UTC_yr.isin(EN)]    
year_se_EN = np.array(Torn_se_EN['UTC_yr'])
month_se_EN = np.array(Torn_se_EN['UTC_mo'])
day_se_EN = np.array(Torn_se_EN['UTC_dy'])

Torn_se_LN = sev_se_one[sev_se_one.UTC_yr.isin(LN)]    
year_se_LN = np.array(Torn_se_LN['UTC_yr'])
month_se_LN = np.array(Torn_se_LN['UTC_mo'])
day_se_LN = np.array(Torn_se_LN['UTC_dy'])


Torn_se_N_two = sev_se_two[sev_se_two.UTC_yr.isin(N_two)]    
year_se_N_two = np.array(Torn_se_N_two['UTC_yr'])
month_se_N_two = np.array(Torn_se_N_two['UTC_mo'])
day_se_N_two = np.array(Torn_se_N_two['UTC_dy'])

Torn_se_EN_two = sev_se_two[sev_se_two.UTC_yr.isin(EN_two)]    
year_se_EN_two = np.array(Torn_se_EN_two['UTC_yr'])
month_se_EN_two = np.array(Torn_se_EN_two['UTC_mo'])
day_se_EN_two = np.array(Torn_se_EN_two['UTC_dy'])

Torn_se_LN_two = sev_se_two[sev_se_two.UTC_yr.isin(LN_two)]    
year_se_LN_two = np.array(Torn_se_LN_two['UTC_yr'])
month_se_LN_two = np.array(Torn_se_LN_two['UTC_mo'])
day_se_LN_two = np.array(Torn_se_LN_two['UTC_dy'])


###############################################################################
###############################################################################
###############################################################################


years_all = pd.date_range('1953-06-01','2016-05-31',freq='AS-JUN').year


year_us_all = np.array(sev_us['UTC_yr'])
month_us_all = np.array(sev_us['UTC_mo'])
day_us_all = np.array(sev_us['UTC_dy'])


year_se_all = np.array(sev_se['UTC_yr'])
month_se_all = np.array(sev_se['UTC_mo'])
day_se_all = np.array(sev_se['UTC_dy'])


###############################################################################
###############################################################################
###############################################################################


julian_us_A = []
for n in range(0,len(year_us_all)-1):
   Date = dt.datetime(year_us_all[n],month_us_all[n],day_us_all[n])
   julian_us_A.append(Date.strftime('%j'))
   
julian_se_A = []
for n in range(0,len(year_se_all)-1):
   Date = dt.datetime(year_se_all[n],month_se_all[n],day_se_all[n])
   julian_se_A.append(Date.strftime('%j'))
   

###############################################################################

   
julian_us_N = []
for n in range(0,len(year_us_N)-1):
   Date = dt.datetime(year_us_N[n],month_us_N[n],day_us_N[n])
   julian_us_N.append(Date.strftime('%j'))

julian_us_EN = []
for n in range(0,len(year_us_EN)-1):
   Date = dt.datetime(year_us_EN[n],month_us_EN[n],day_us_EN[n])
   julian_us_EN.append(Date.strftime('%j'))

julian_us_LN = []
for n in range(0,len(year_us_LN)-1):
   Date = dt.datetime(year_us_LN[n],month_us_LN[n],day_us_LN[n])
   julian_us_LN.append(Date.strftime('%j'))
   

julian_us_N_two = []
for n in range(0,len(year_us_N_two)-1):
   Date = dt.datetime(year_us_N_two[n],month_us_N_two[n],day_us_N_two[n])
   julian_us_N_two.append(Date.strftime('%j'))

julian_us_EN_two = []
for n in range(0,len(year_us_EN_two)-1):
   Date = dt.datetime(year_us_EN_two[n],month_us_EN_two[n],day_us_EN_two[n])
   julian_us_EN_two.append(Date.strftime('%j'))

julian_us_LN_two = []
for n in range(0,len(year_us_LN_two)-1):
   Date = dt.datetime(year_us_LN_two[n],month_us_LN_two[n],day_us_LN_two[n])
   julian_us_LN_two.append(Date.strftime('%j'))
   
   
###############################################################################
   

julian_se_N = []
for n in range(0,len(year_se_N)-1):
   Date = dt.datetime(year_se_N[n],month_se_N[n],day_se_N[n])
   julian_se_N.append(Date.strftime('%j'))

julian_se_EN = []
for n in range(0,len(year_se_EN)-1):
   Date = dt.datetime(year_se_EN[n],month_se_EN[n],day_se_EN[n])
   julian_se_EN.append(Date.strftime('%j'))

julian_se_LN = []
for n in range(0,len(year_se_LN)-1):
   Date = dt.datetime(year_se_LN[n],month_se_LN[n],day_se_LN[n])
   julian_se_LN.append(Date.strftime('%j'))
   

julian_se_N_two = []
for n in range(0,len(year_se_N_two)-1):
   Date = dt.datetime(year_se_N_two[n],month_se_N_two[n],day_se_N_two[n])
   julian_se_N_two.append(Date.strftime('%j'))

julian_se_EN_two = []
for n in range(0,len(year_se_EN_two)-1):
   Date = dt.datetime(year_se_EN_two[n],month_se_EN_two[n],day_se_EN_two[n])
   julian_se_EN_two.append(Date.strftime('%j'))

julian_se_LN_two = []
for n in range(0,len(year_se_LN_two)-1):
   Date = dt.datetime(year_se_LN_two[n],month_se_LN_two[n],day_se_LN_two[n])
   julian_se_LN_two.append(Date.strftime('%j'))
   
   
###############################################################################
###############################################################################
###############################################################################
   
 
Julian_US_N = julian_us_N + julian_us_N_two
Julian_US_EN = julian_us_EN + julian_us_EN_two   
Julian_US_LN = julian_us_LN + julian_us_LN_two
   
Julian_SE_N = julian_se_N + julian_se_N_two
Julian_SE_EN = julian_se_EN + julian_se_EN_two   
Julian_SE_LN = julian_se_LN + julian_se_LN_two
   

###############################################################################
###############################################################################
###############################################################################


Julian_US_A = [int(i) for i in julian_us_A]
Histo = np.histogram(Julian_US_A, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_A_EF2 = Gauss_Smooth0[366:732]/len(years_all)

Julian_SE_A = [int(i) for i in julian_se_A]
Histo = np.histogram(Julian_SE_A, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_A_EF2 = Gauss_Smooth0[366:732]/len(years_all)


###############################################################################


Julian_US_N = [int(i) for i in Julian_US_N]
Histo = np.histogram(Julian_US_N, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_N_EF2 = Gauss_Smooth0[366:732]/len(N)

Julian_US_EN = [int(i) for i in Julian_US_EN]
Histo = np.histogram(Julian_US_EN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_EN_EF2 = Gauss_Smooth0[366:732]/len(EN)

Julian_US_LN = [int(i) for i in Julian_US_LN]
Histo = np.histogram(Julian_US_LN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_US_LN_EF2 = Gauss_Smooth0[366:732]/len(LN)


###############################################################################


Julian_SE_N = [int(i) for i in Julian_SE_N]
Histo = np.histogram(Julian_SE_N, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_N_EF2 = Gauss_Smooth0[366:732]/len(N)

Julian_SE_EN = [int(i) for i in Julian_SE_EN]
Histo = np.histogram(Julian_SE_EN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_EN_EF2 = Gauss_Smooth0[366:732]/len(EN)

Julian_SE_LN = [int(i) for i in Julian_SE_LN]
Histo = np.histogram(Julian_SE_LN, bins=366)
HistoN = np.zeros(366*3)
HistoN[0:366]=Histo[0] 
HistoN[366:732]=Histo[0]  
HistoN[732:1101]=Histo[0]
Gauss_Smooth0 = gfilt(HistoN*1.0,sigma=15.0)
Gauss_Smooth_SE_LN_EF2 = Gauss_Smooth0[366:732]/len(LN)


###############################################################################
###############################################################################
###############################################################################




fig = plt.figure(figsize=(11.5,11))


ax1 = fig.add_axes([0.0, 0.5, 0.485, 0.475]) 

for f in xrange(len(J_US_EN)):
    
    ax1.plot(range(0,366),J_US_EN[f]/np.sum(J_US_EN[f]),'r-',linewidth=0.8,alpha=0.35)
    
for g in xrange(len(J_US_LN)):    
    
    ax1.plot(range(0,366),J_US_LN[g]/np.sum(J_US_LN[g]),'b-',linewidth=0.8,alpha=0.35)
    

p1, = ax1.plot(range(0,366),Gauss_Smooth_US_EN/np.sum(Gauss_Smooth_US_EN),'r-',linewidth=2.0)
p2, = ax1.plot(range(0,366),Gauss_Smooth_US_LN/np.sum(Gauss_Smooth_US_LN),'b-',linewidth=2.0)
p3, = ax1.plot(range(0,366),Gauss_Smooth_US_N/np.sum(Gauss_Smooth_US_N),'k-',linewidth=2.0)    
p4, = ax1.plot(range(0,366),Gauss_Smooth_US_A/np.sum(Gauss_Smooth_US_A),'--',color='grey',linewidth=2.0)  

ax1.set_ylabel('Fraction of Tornado Reports', fontsize=10)

ax1.set_yticks([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.020])
plt.setp(ax1.get_yticklabels(), fontsize=10, rotation=35)
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_title('a) ENSO Years Annual Cycle of CONUS EF1+ Tornado Reports')

ax1.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3,p4],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "All"],
                    loc="upper right",
                    fancybox=True, fontsize=12)



ax2 = fig.add_axes([0.5, 0.5, 0.485, 0.475]) 

for f in xrange(len(J_US_EN_EF2)):
    
    ax2.plot(range(0,366),J_US_EN_EF2[f]/np.sum(J_US_EN_EF2[f]),'r-',linewidth=0.8,alpha=0.35)
    
for g in xrange(len(J_US_LN_EF2)):    
    
    ax2.plot(range(0,366),J_US_LN_EF2[g]/np.sum(J_US_LN_EF2[g]),'b-',linewidth=0.8,alpha=0.35)

p1, = ax2.plot(range(0,366),Gauss_Smooth_US_EN_EF2/np.sum(Gauss_Smooth_US_EN_EF2),'r-',linewidth=2.0)
p2, = ax2.plot(range(0,366),Gauss_Smooth_US_LN_EF2/np.sum(Gauss_Smooth_US_LN_EF2),'b-',linewidth=2.0)
p3, = ax2.plot(range(0,366),Gauss_Smooth_US_N_EF2/np.sum(Gauss_Smooth_US_N_EF2),'k-',linewidth=2.0)    
p4, = ax2.plot(range(0,366),Gauss_Smooth_US_A_EF2/np.sum(Gauss_Smooth_US_A_EF2),'--',color='grey',linewidth=2.0)  

#ax2.set_ylabel('Fraction of Tornado Reports (EF2+)', fontsize=10)

ax2.set_yticks([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.020])
plt.setp(ax2.get_yticklabels(), fontsize=10, rotation=35, alpha=0)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax2.set_title('b) ENSO Years Annual Cycle of CONUS EF2+ Tornado Reports')

ax2.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3,p4],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "All"],
                    loc="upper right",
                    fancybox=True, fontsize=12)


ax3 = fig.add_axes([0.0, 0.0, 0.485, 0.475]) 

for f in xrange(len(J_SE_EN)):
    
    ax3.plot(range(0,366),J_SE_EN[f]/np.sum(J_SE_EN[f]),'r-',linewidth=0.8,alpha=0.35)
    
for g in xrange(len(J_SE_LN)):    
    
    ax3.plot(range(0,366),J_SE_LN[g]/np.sum(J_SE_LN[g]),'b-',linewidth=0.8,alpha=0.35)

p1, = ax3.plot(range(0,366),Gauss_Smooth_SE_EN/np.sum(Gauss_Smooth_SE_EN),'r-',linewidth=2.0)
p2, = ax3.plot(range(0,366),Gauss_Smooth_SE_LN/np.sum(Gauss_Smooth_SE_LN),'b-',linewidth=2.0)
p3, = ax3.plot(range(0,366),Gauss_Smooth_SE_N/np.sum(Gauss_Smooth_SE_N),'k-',linewidth=2.0)    
p4, = ax3.plot(range(0,366),Gauss_Smooth_SE_A/np.sum(Gauss_Smooth_SE_A),'--',color='grey',linewidth=2.0)  

ax3.set_ylabel('Fraction of Tornado Reports', fontsize=10)
ax3.set_xlabel('Day of Year', fontsize=10)

ax3.set_yticks([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.020])
plt.setp(ax3.get_yticklabels(), fontsize=10, rotation=35)

tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax3.set_xticks(tick_locs) 
ax3.set_xticklabels(tick_lbls)
#ax3.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

#ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax3.set_title('c) ENSO Years Annual Cycle of Southeast EF1+ Tornado Reports')

ax3.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3,p4],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "All"],
                    loc="upper right",
                    fancybox=True, fontsize=12)



ax4 = fig.add_axes([0.5, 0.0, 0.485, 0.475]) 

for f in xrange(len(J_SE_EN_EF2)):
    
    ax4.plot(range(0,366),J_SE_EN_EF2[f]/np.sum(J_SE_EN_EF2[f]),'r-',linewidth=0.8,alpha=0.35)
    
for g in xrange(len(J_SE_LN_EF2)):    
    
    ax4.plot(range(0,366),J_SE_LN_EF2[g]/np.sum(J_SE_LN_EF2[g]),'b-',linewidth=0.8,alpha=0.35)

p1, = ax4.plot(range(0,366),Gauss_Smooth_SE_EN_EF2/np.sum(Gauss_Smooth_SE_EN_EF2),'r-',linewidth=2.0)
p2, = ax4.plot(range(0,366),Gauss_Smooth_SE_LN_EF2/np.sum(Gauss_Smooth_SE_LN_EF2),'b-',linewidth=2.0)
p3, = ax4.plot(range(0,366),Gauss_Smooth_SE_N_EF2/np.sum(Gauss_Smooth_SE_N_EF2),'k-',linewidth=2.0)    
p4, = ax4.plot(range(0,366),Gauss_Smooth_SE_A_EF2/np.sum(Gauss_Smooth_SE_A_EF2),'--',color='grey',linewidth=2.0)  

#ax4.set_ylabel('Fraction of Tornado Reports (EF2+)', fontsize=10)
ax4.set_xlabel('Day of Year', fontsize=10)

ax4.set_yticks([0,0.002,0.004,0.006,0.008,0.01,0.012,0.014,0.016,0.018,0.020])
plt.setp(ax4.get_yticklabels(), fontsize=10, rotation=35, alpha=0)
#ax4.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax4.set_xticks(tick_locs) 
ax4.set_xticklabels(tick_lbls)

ax4.set_title('d) ENSO Years Annual Cycle of Southeast EF2+ Tornado Reports')

ax4.grid(True, linestyle='--', alpha=0.5)

legend = plt.legend([p1,p2,p3,p4],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "All"],
                    loc="upper right",
                    fancybox=True, fontsize=12)

plt.savefig('wut.png', bbox_inches='tight', dpi=200)

#plt.show()


###############################################################################
###############################################################################
###############################################################################
