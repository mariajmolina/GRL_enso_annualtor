#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:07:23 2017

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
import matplotlib.pyplot as plt
#import scipy.stats
import xarray as xr
import matplotlib as mpl
#import matplotlib.colors as coloring


###############################################################################
###############################################################################
###############################################################################


def make_colormap(colors):
    
    from matplotlib.colors import LinearSegmentedColormap, ColorConverter

    z  = np.array(sorted(colors.keys()))
    n  = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / (zn - z1)

    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        Ci = colors[z[i]]      
        if type(Ci) == str:
            RGB = CC.to_rgb(Ci)
        else:
            RGB = Ci
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])

    cmap_dict = {}
    cmap_dict['red']   = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue']  = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)
    
    return mymap


###############################################################################
###############################################################################
###############################################################################
    

maria_color = ({0.:'blue',0.2:'steelblue',0.39:'cornflowerblue',0.4:'w',0.6:'w',0.61:'salmon',0.8:'red',1:'darkred'})
cmap = make_colormap(maria_color)

norm = mpl.colors.Normalize(vmin=-2.5,vmax=2.5)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-12-01 00:00:00'
time_2 = '2016-12-31 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor1 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor1.severe_load() 

yrs1 = pd.date_range('1953','2016',freq='AS').year
idx1 = pd.date_range('1953-12-01','2016-12-31', freq='AS-DEC')
t1 = alltor1.severe.set_index('UTC_time')['fat'].resample('AS-DEC').count()
t1.index = pd.DatetimeIndex(t1.index)
t1 = t1.reindex(idx1, fill_value=0)

data1 = xr.Dataset({'oni': (['time'], alltor1.oni_values),
                   'tor': (['time'], t1)},
                    coords={'time': t1.index.year}) 

g1 = np.nanmean(t1)

label1 = np.array(data1.sortby(data1['oni'],ascending=True).oni.values, dtype=str)
xlabels1 = np.array(data1.sortby(data1['oni'],ascending=True).time.values, dtype=str)
numtors1 = np.array(data1.sortby(data1['oni'],ascending=True).tor.values, dtype=int)
onis1 = np.arange(0,len(numtors1),1)

c_map1 = np.array(data1.sortby(data1['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-01-01 00:00:00'
time_2 = '2016-01-31 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor2 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor2.severe_load() 


yrs2 = pd.date_range('1953','2016',freq='AS').year
idx2 = pd.date_range('1953-01-01','2016-01-31', freq='AS-JAN')
t2 = alltor2.severe.set_index('UTC_time')['fat'].resample('AS-JAN').count()
t2.index = pd.DatetimeIndex(t2.index)
t2 = t2.reindex(idx2, fill_value=0)

data2 = xr.Dataset({'oni': (['time'], alltor2.oni_values),
                   'tor': (['time'], t2)},
                    coords={'time': t2.index.year}) 

g2 = np.nanmean(t2)

label2 = np.array(data2.sortby(data2['oni'],ascending=True).oni.values, dtype=str)
xlabels2 = np.array(data2.sortby(data2['oni'],ascending=True).time.values, dtype=str)
numtors2 = np.array(data2.sortby(data2['oni'],ascending=True).tor.values, dtype=int)
onis2 = np.arange(0,len(numtors2),1)

c_map2 = np.array(data2.sortby(data2['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-02-01 00:00:00'
time_2 = '2016-02-28 21:00:00'

lat_1 = 18.
lat_2 = 30.5
lon_1 = -98
lon_2 = -82


alltor3 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor3.severe_load() 

yrs3 = pd.date_range('1953','2016',freq='AS').year
idx3 = pd.date_range('1953-02-01','2016-02-28', freq='AS-FEB')
t3 = alltor3.severe.set_index('UTC_time')['fat'].resample('AS-FEB').count()
t3.index = pd.DatetimeIndex(t3.index)
t3 = t3.reindex(idx3, fill_value=0)

data3 = xr.Dataset({'oni': (['time'], alltor3.oni_values),
                   'tor': (['time'], t3)},
                    coords={'time': t3.index.year}) 

g3 = np.nanmean(t3)

label3 = np.array(data3.sortby(data3['oni'],ascending=True).oni.values, dtype=str)
xlabels3 = np.array(data3.sortby(data3['oni'],ascending=True).time.values, dtype=str)
numtors3 = np.array(data3.sortby(data3['oni'],ascending=True).tor.values, dtype=int)
onis3 = np.arange(0,len(numtors3),1)

c_map3 = np.array(data3.sortby(data3['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-03-01 00:00:00'
time_2 = '2016-03-31 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor4 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor4.severe_load() 

yrs4 = pd.date_range('1953','2016',freq='AS').year
idx4 = pd.date_range('1953-03-01','2016-03-31', freq='AS-MAR')
t4 = alltor4.severe.set_index('UTC_time')['fat'].resample('AS-MAR').count()
t4.index = pd.DatetimeIndex(t4.index)
t4 = t4.reindex(idx4, fill_value=0)

data4 = xr.Dataset({'oni': (['time'], alltor4.oni_values),
                   'tor': (['time'], t4)},
                    coords={'time': t4.index.year}) 

g4 = np.nanmean(t4)

label4 = np.array(data4.sortby(data4['oni'],ascending=True).oni.values, dtype=str)
xlabels4 = np.array(data4.sortby(data4['oni'],ascending=True).time.values, dtype=str)
numtors4 = np.array(data4.sortby(data4['oni'],ascending=True).tor.values, dtype=int)
onis4 = np.arange(0,len(numtors4),1)

c_map4 = np.array(data4.sortby(data4['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-04-01 00:00:00'
time_2 = '2016-04-30 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor5 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor5.severe_load() 

yrs5 = pd.date_range('1953','2016',freq='AS').year
idx5 = pd.date_range('1953-04-01','2016-04-30', freq='AS-APR')
t5 = alltor5.severe.set_index('UTC_time')['fat'].resample('AS-APR').count()
t5.index = pd.DatetimeIndex(t5.index)
t5 = t5.reindex(idx5, fill_value=0)

data5 = xr.Dataset({'oni': (['time'], alltor5.oni_values),
                   'tor': (['time'], t5)},
                    coords={'time': t5.index.year}) 

g5 = np.nanmean(t5)

label5 = np.array(data5.sortby(data5['oni'],ascending=True).oni.values, dtype=str)
xlabels5 = np.array(data5.sortby(data5['oni'],ascending=True).time.values, dtype=str)
numtors5 = np.array(data5.sortby(data5['oni'],ascending=True).tor.values, dtype=int)
onis5 = np.arange(0,len(numtors5),1)

c_map5 = np.array(data5.sortby(data5['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-05-01 00:00:00'
time_2 = '2016-05-31 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor6 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor6.severe_load() 

yrs6 = pd.date_range('1953','2016',freq='AS').year
idx6 = pd.date_range('1953-05-01','2016-05-31', freq='AS-MAY')
t6 = alltor6.severe.set_index('UTC_time')['fat'].resample('AS-MAY').count()
t6.index = pd.DatetimeIndex(t6.index)
t6 = t6.reindex(idx6, fill_value=0)

data6 = xr.Dataset({'oni': (['time'], alltor6.oni_values),
                   'tor': (['time'], t6)},
                    coords={'time': t6.index.year}) 

g6 = np.nanmean(t6)

label6 = np.array(data6.sortby(data6['oni'],ascending=True).oni.values, dtype=str)
xlabels6 = np.array(data6.sortby(data6['oni'],ascending=True).time.values, dtype=str)
numtors6 = np.array(data6.sortby(data6['oni'],ascending=True).tor.values, dtype=int)
onis6 = np.arange(0,len(numtors6),1)

c_map6 = np.array(data6.sortby(data6['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-06-01 00:00:00'
time_2 = '2016-06-30 21:00:00'

lat_1 = 18.
lat_2 = 30.5
lon_1 = -98
lon_2 = -82


alltor7 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor7.severe_load() 

yrs7 = pd.date_range('1953','2016',freq='AS').year
idx7 = pd.date_range('1953-06-01','2016-06-30', freq='AS-JUN')
t7 = alltor7.severe.set_index('UTC_time')['fat'].resample('AS-JUN').count()
t7.index = pd.DatetimeIndex(t7.index)
t7 = t7.reindex(idx7, fill_value=0)

data7 = xr.Dataset({'oni': (['time'], alltor7.oni_values),
                   'tor': (['time'], t7)},
                    coords={'time': t7.index.year}) 

g7 = np.nanmean(t7)

label7 = np.array(data7.sortby(data7['oni'],ascending=True).oni.values, dtype=str)
xlabels7 = np.array(data7.sortby(data7['oni'],ascending=True).time.values, dtype=str)
numtors7 = np.array(data7.sortby(data7['oni'],ascending=True).tor.values, dtype=int)
onis7 = np.arange(0,len(numtors7),1)

c_map7 = np.array(data7.sortby(data7['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-07-01 00:00:00'
time_2 = '2016-07-31 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor8 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor8.severe_load() 

yrs8 = pd.date_range('1953','2016',freq='AS').year
idx8 = pd.date_range('1953-07-01','2016-07-31', freq='AS-JUL')
t8 = alltor8.severe.set_index('UTC_time')['fat'].resample('AS-JUL').count()
t8.index = pd.DatetimeIndex(t8.index)
t8 = t8.reindex(idx8, fill_value=0)

data8 = xr.Dataset({'oni': (['time'], alltor8.oni_values),
                   'tor': (['time'], t8)},
                    coords={'time': t8.index.year}) 

g8 = np.nanmean(t8)

label8 = np.array(data8.sortby(data8['oni'],ascending=True).oni.values, dtype=str)
xlabels8 = np.array(data8.sortby(data8['oni'],ascending=True).time.values, dtype=str)
numtors8 = np.array(data8.sortby(data8['oni'],ascending=True).tor.values, dtype=int)
onis8 = np.arange(0,len(numtors8),1)

c_map8 = np.array(data8.sortby(data8['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-08-01 00:00:00'
time_2 = '2016-08-31 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor9 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor9.severe_load() 

yrs9 = pd.date_range('1953','2016',freq='AS').year
idx9 = pd.date_range('1953-08-01','2016-08-31', freq='AS-AUG')
t9 = alltor9.severe.set_index('UTC_time')['fat'].resample('AS-AUG').count()
t9.index = pd.DatetimeIndex(t9.index)
t9 = t9.reindex(idx9, fill_value=0)

data9 = xr.Dataset({'oni': (['time'], alltor9.oni_values),
                   'tor': (['time'], t9)},
                    coords={'time': t9.index.year}) 

g9 = np.nanmean(t9)

label9 = np.array(data9.sortby(data9['oni'],ascending=True).oni.values, dtype=str)
xlabels9 = np.array(data9.sortby(data9['oni'],ascending=True).time.values, dtype=str)
numtors9 = np.array(data9.sortby(data9['oni'],ascending=True).tor.values, dtype=int)
onis9 = np.arange(0,len(numtors9),1)

c_map9 = np.array(data9.sortby(data9['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-09-01 00:00:00'
time_2 = '2016-09-30 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor10 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor10.severe_load() 

yrs10 = pd.date_range('1953','2016',freq='AS').year
idx10 = pd.date_range('1953-09-01','2016-09-30', freq='AS-SEP')
t10 = alltor10.severe.set_index('UTC_time')['fat'].resample('AS-SEP').count()
t10.index = pd.DatetimeIndex(t10.index)
t10 = t10.reindex(idx10, fill_value=0)

data10 = xr.Dataset({'oni': (['time'], alltor10.oni_values),
                   'tor': (['time'], t10)},
                    coords={'time': t10.index.year}) 

g10 = np.nanmean(t10)

label10 = np.array(data10.sortby(data10['oni'],ascending=True).oni.values, dtype=str)
xlabels10 = np.array(data10.sortby(data10['oni'],ascending=True).time.values, dtype=str)
numtors10 = np.array(data10.sortby(data10['oni'],ascending=True).tor.values, dtype=int)
onis10 = np.arange(0,len(numtors10),1)

c_map10 = np.array(data10.sortby(data10['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-10-01 00:00:00'
time_2 = '2016-10-31 21:00:00'

lat_1 = 18.
lat_2 = 30.5
lon_1 = -98
lon_2 = -82


alltor11 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor11.severe_load() 

yrs11 = pd.date_range('1953','2016',freq='AS').year
idx11 = pd.date_range('1953-10-01','2016-10-31', freq='AS-OCT')
t11 = alltor11.severe.set_index('UTC_time')['fat'].resample('AS-OCT').count()
t11.index = pd.DatetimeIndex(t11.index)
t11 = t11.reindex(idx11, fill_value=0)

data11 = xr.Dataset({'oni': (['time'], alltor11.oni_values),
                   'tor': (['time'], t11)},
                    coords={'time': t11.index.year}) 

g11 = np.nanmean(t11)

label11 = np.array(data11.sortby(data11['oni'],ascending=True).oni.values, dtype=str)
xlabels11 = np.array(data11.sortby(data11['oni'],ascending=True).time.values, dtype=str)
numtors11 = np.array(data11.sortby(data11['oni'],ascending=True).tor.values, dtype=int)
onis11 = np.arange(0,len(numtors11),1)

c_map11 = np.array(data11.sortby(data11['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


time_1 = '1953-11-01 00:00:00'
time_2 = '2017-11-30 21:00:00'

lat_1 = 18.5
lat_2 = 30.5
lon_1 = -98
lon_2 = -82

alltor12 = pk.whats_up_doc(time_1, time_2,
                       frequency = 'subseason', stat_choice = 'composite', tor_min=1,
                       enso_neutral=False, enso_only=False, nino_only=False, nina_only=False, gom_warm=False, gom_cold=False,
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='southeast',
                       lat_1 = lat_1, lat_2 = lat_2, lon_1 = lon_1, lon_2 = lon_2)

alltor12.severe_load() 

yrs12 = pd.date_range('1953','2016',freq='AS').year
idx12 = pd.date_range('1953-11-01','2016-11-30', freq='AS-NOV')
t12 = alltor12.severe.set_index('UTC_time')['fat'].resample('AS-NOV').count()
t12.index = pd.DatetimeIndex(t12.index)
t12 = t12.reindex(idx12, fill_value=0)


data12 = xr.Dataset({'oni': (['time'], alltor12.oni_values),
                   'tor': (['time'], t12)},
                    coords={'time': t12.index.year}) 


g12 = np.nanmean(t12)

label12 = np.array(data12.sortby(data12['oni'],ascending=True).oni.values, dtype=str)
xlabels12 = np.array(data12.sortby(data12['oni'],ascending=True).time.values, dtype=str)
numtors12 = np.array(data12.sortby(data12['oni'],ascending=True).tor.values, dtype=int)
onis12 = np.arange(0,len(numtors12),1)

c_map12 = np.array(data12.sortby(data12['oni'],ascending=True).oni.values, dtype=float)


###############################################################################
###############################################################################
###############################################################################


p1 = numtors1/np.nanmean(numtors1)
p2 = numtors2/np.nanmean(numtors2)
p3 = numtors3/np.nanmean(numtors3)
p4 = numtors4/np.nanmean(numtors4)
p5 = numtors5/np.nanmean(numtors5)
p6 = numtors6/np.nanmean(numtors6)
p7 = numtors7/np.nanmean(numtors7)
p8 = numtors8/np.nanmean(numtors8)
p9 = numtors9/np.nanmean(numtors9)
p10 = numtors10/np.nanmean(numtors10)
p11 = numtors11/np.nanmean(numtors11)
p12 = numtors12/np.nanmean(numtors12)



'''

p1 = numtors1/np.sum(numtors1)
p2 = numtors2/np.sum(numtors2)
p3 = numtors3/np.sum(numtors3)
p4 = numtors4/np.sum(numtors4)
p5 = numtors5/np.sum(numtors5)
p6 = numtors6/np.sum(numtors6)
p7 = numtors7/np.sum(numtors7)
p8 = numtors8/np.sum(numtors8)
p9 = numtors9/np.sum(numtors9)
p10 = numtors10/np.sum(numtors10)
p11 = numtors11/np.sum(numtors11)
p12 = numtors12/np.sum(numtors12)

'''

'''

p1 = []
p2 = []
p3 = []
p4 = []
p5 = []
p6 = []
p7 = []
p8 = []
p9 = []
p10 = []
p11 = []
p12 = []


for i in xrange(0,len(numtors1)):
    p1.append(numtors1[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))
    
for i in xrange(0,len(numtors1)):
    p2.append(numtors2[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p3.append(numtors3[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p4.append(numtors4[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p5.append(numtors5[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p6.append(numtors6[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p7.append(numtors7[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p8.append(numtors8[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p9.append(numtors9[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p10.append(numtors10[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p11.append(numtors11[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))

for i in xrange(0,len(numtors1)):
    p12.append(numtors12[i]/(numtors1[i]+numtors2[i]+numtors3[i]+numtors4[i]+numtors5[i]+numtors6[i]+numtors7[i]+numtors8[i]+numtors9[i]+numtors10[i]+numtors11[i]+numtors12[i]))
'''


###############################################################################
###############################################################################
###############################################################################



'''
class MidpointNormalize(coloring.Normalize):
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        coloring.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):

        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
'''


###############################################################################
###############################################################################
###############################################################################


fig = plt.figure(figsize=(12,16))
#fig = plt.figure(figsize=(24,32))

#ax1 = fig.add_axes([0.0, 0.75, 0.32, 0.21]) 
ax1 = fig.add_axes([0.0, 0.75, 0.295, 0.225]) 


#ax1 = fig.add_axes([0.0, 0.0, 1, 1]) 

t1 = ax1.barh(onis1, p1, height=1.0, align='center', color=cmap(norm(c_map1)), edgecolor='dimgrey', alpha=1, zorder=2)
t_1, = ax1.plot([np.nanmean(p1),np.nanmean(p1)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax1.set_xticks(ticks1)
plt.setp(ax1.get_xticklabels(), fontsize=9, alpha=0)

ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_yticks(onis1[::3])
ax1.set_yticklabels(label1[::3], fontsize=9, rotation=0)

ax1.set_title('a) Dec Tornadoes; NDJ ONI',fontsize=16)
ax1.set_ylabel('Years Ranked by ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels1):
    plt.annotate(txt, (p1[i], onis1[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''
#plt.show()

###############################################################################
###############################################################################
###############################################################################


#ax2 = fig.add_axes([0.37, 0.75, 0.32, 0.21])
ax2 = fig.add_axes([0.34, 0.75, 0.295, 0.225])

t2 = ax2.barh(onis2, p2, height=1.0, align='center', color=cmap(norm(c_map2)), edgecolor='dimgrey', alpha=1, zorder=2)
t_2, = ax2.plot([np.nanmean(p2),np.nanmean(p2)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax2.set_xticks(ticks1)
plt.setp(ax2.get_xticklabels(), fontsize=9, alpha=0)

ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax2.set_yticks(onis2[::3])
ax2.set_yticklabels(label2[::3], fontsize=10, rotation=0)

ax2.set_title('b) Jan Tornadoes; DJF ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels2):
    plt.annotate(txt, (p2[i], onis2[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


#ax3 = fig.add_axes([0.74, 0.75, 0.32, 0.21])
ax3 = fig.add_axes([0.68, 0.75, 0.295, 0.225])

t3 = ax3.barh(onis3, p3, height=1.0, align='center', color=cmap(norm(c_map3)), edgecolor='dimgrey', alpha=1, zorder=2)
t_3, = ax3.plot([np.nanmean(p3),np.nanmean(p3)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax3.set_xticks(ticks1)
plt.setp(ax3.get_xticklabels(), fontsize=9, alpha=0)

ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax3.set_yticks(onis3[::3])
ax3.set_yticklabels(label3[::3], fontsize=10, rotation=0)

ax3.set_title('c) Feb Tornadoes; JFM ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels3):
    plt.annotate(txt, (p3[i], onis3[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


#ax4 = fig.add_axes([0.0, 0.5, 0.32, 0.21])
ax4 = fig.add_axes([0.0, 0.505, 0.295, 0.225])

t4 = ax4.barh(onis4, p4, height=1.0, align='center', color=cmap(norm(c_map4)), edgecolor='dimgrey', alpha=1, zorder=2)
t_4, = ax4.plot([np.nanmean(p4),np.nanmean(p4)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax4.set_xticks(ticks1)
plt.setp(ax4.get_xticklabels(), fontsize=9, alpha=0)

ax4.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax4.set_yticks(onis4[::3])
ax4.set_yticklabels(label4[::3], fontsize=10, rotation=0)

ax4.set_title('d) Mar Tornadoes; FMA ONI',fontsize=16)
ax4.set_ylabel('Years Ranked by ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels4):
    plt.annotate(txt, (p4[i], onis4[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


#ax5 = fig.add_axes([0.37, 0.5, 0.32, 0.21])
ax5 = fig.add_axes([0.34, 0.505, 0.295, 0.225])

t5 = ax5.barh(onis5, p5, height=1.0, align='center', color=cmap(norm(c_map5)), edgecolor='dimgrey', alpha=1, zorder=2)
t_5, = ax5.plot([np.nanmean(p5),np.nanmean(p5)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax5.set_xticks(ticks1)
plt.setp(ax5.get_xticklabels(), fontsize=9, alpha=0)

ax5.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax5.set_yticks(onis5[::3])
ax5.set_yticklabels(label5[::3], fontsize=10, rotation=0)

ax5.set_title('e) Apr Tornadoes; MAM ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels5):
    plt.annotate(txt, (p5[i], onis5[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


#ax6 = fig.add_axes([0.74, 0.5, 0.32, 0.21])
ax6 = fig.add_axes([0.68, 0.505, 0.295, 0.225])

t6 = ax6.barh(onis6, p6, height=1.0, align='center', color=cmap(norm(c_map6)), edgecolor='dimgrey', alpha=1, zorder=2)
t_6, = ax6.plot([np.nanmean(p6),np.nanmean(p6)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax6.set_xticks(ticks1)
plt.setp(ax6.get_xticklabels(), fontsize=9, alpha=0)

ax6.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax6.set_yticks(onis6[::3])
ax6.set_yticklabels(label6[::3], fontsize=10, rotation=0)

ax6.set_title('f) May Tornadoes; AMJ ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels6):
    plt.annotate(txt, (p6[i], onis6[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


#ax7 = fig.add_axes([0.0, 0.25, 0.32, 0.21])
ax7 = fig.add_axes([0.0, 0.26, 0.295, 0.225])

t7 = ax7.barh(onis7, p7, height=1.0, align='center', color=cmap(norm(c_map7)), edgecolor='dimgrey', alpha=1, zorder=2)
t_7, = ax7.plot([np.nanmean(p7),np.nanmean(p7)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax7.set_xticks(ticks1)
plt.setp(ax7.get_xticklabels(), fontsize=9, alpha=0)

ax7.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax7.set_yticks(onis7[::3])
ax7.set_yticklabels(label7[::3], fontsize=10, rotation=0)

ax7.set_title('g) Jun Tornadoes; MJJ ONI',fontsize=16)
ax7.set_ylabel('Years Ranked by ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels7):
    plt.annotate(txt, (p7[i], onis7[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


#ax8 = fig.add_axes([0.37, 0.25, 0.32, 0.21])
ax8 = fig.add_axes([0.34, 0.26, 0.295, 0.225])

t8 = ax8.barh(onis8, p8, height=1.0, align='center', color=cmap(norm(c_map8)), edgecolor='dimgrey', alpha=1, zorder=2)
t_8, = ax8.plot([np.nanmean(p8),np.nanmean(p8)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax8.set_xticks(ticks1)
plt.setp(ax8.get_xticklabels(), fontsize=9, alpha=0)

ax8.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax8.set_yticks(onis8[::3])
ax8.set_yticklabels(label8[::3], fontsize=10, rotation=0)

ax8.set_title('h) Jul Tornadoes; JJA ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels8):
    plt.annotate(txt, (p8[i], onis8[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


#ax9 = fig.add_axes([0.74, 0.25, 0.32, 0.21])
ax9 = fig.add_axes([0.68, 0.26, 0.295, 0.225])

t9 = ax9.barh(onis9, p9, height=1.0, align='center', color=cmap(norm(c_map9)), edgecolor='dimgrey', alpha=1, zorder=2)
t_9, = ax9.plot([np.nanmean(p9),np.nanmean(p9)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax9.set_xticks(ticks1)
plt.setp(ax9.get_xticklabels(), fontsize=9, alpha=0)

ax9.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax9.set_yticks(onis9[::3])
ax9.set_yticklabels(label9[::3], fontsize=10, rotation=0)

ax9.set_title('i) Aug Tornadoes; JAS ONI',fontsize=16)
'''
for i, txt in enumerate(xlabels9):
    plt.annotate(txt, (p9[i], onis9[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################
    

#ax10 = fig.add_axes([0.0, 0.0, 0.32, 0.21])
ax10 = fig.add_axes([0.0, 0.015, 0.295, 0.225])

t10 = ax10.barh(onis10, p10, height=1.0, align='center', color=cmap(norm(c_map10)), edgecolor='dimgrey', alpha=1, zorder=2)
t_10, = ax10.plot([np.nanmean(p10),np.nanmean(p10)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax10.set_xticks(ticks1)
ticks_cb = ['0','300%','600%','900%','1200%']
ax10.set_xticklabels(ticks_cb, fontsize=12)
#plt.setp(ax10.get_xticklabels(), fontsize=10)

ax10.set_xticks(ax10.get_xticks()[::3])

#ax10.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax10.set_yticks(onis10[::3])
ax10.set_yticklabels(label10[::3], fontsize=10, rotation=0)

ax10.set_title('j) Sep Tornadoes; ASO ONI',fontsize=16)
ax10.set_ylabel('Years Ranked by ONI',fontsize=16)
ax10.set_xlabel('% EF1+ Climatological Frequency',fontsize=16)
'''
for i, txt in enumerate(xlabels10):
    plt.annotate(txt, (p10[i], onis10[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################
    

#ax11 = fig.add_axes([0.37, 0.0, 0.32, 0.21])
ax11 = fig.add_axes([0.34, 0.015, 0.295, 0.225])

t11 = ax11.barh(onis11, p11, height=1.0, align='center', color=cmap(norm(c_map11)), edgecolor='dimgrey', alpha=1, zorder=2)
t_11, = ax11.plot([np.nanmean(p11),np.nanmean(p11)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax11.set_xticks(ticks1)
plt.setp(ax11.get_xticklabels(), fontsize=9, alpha=0)

ax11.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax11.set_yticks(onis11[::3])
ax11.set_yticklabels(label11[::3], fontsize=10, rotation=0)

ax11.set_title('k) Oct Tornadoes; SON ONI',fontsize=16)
#ax11.set_ylabel('Years Ranked by ONI',fontsize=14)
#ax11.set_xlabel('Number of Tornadoes',fontsize=14)
'''
for i, txt in enumerate(xlabels11):
    plt.annotate(txt, (p11[i], onis11[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''


cbar_ax = fig.add_axes([0.34, 0.005, 0.295, 0.008])


bounds = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
normed = mpl.colors.Normalize(vmin=-2.5, vmax=2.5)

cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                norm=normed,
                                ticks=bounds,
                                orientation='horizontal')




cbar.set_label('ONI Centered by Month', fontsize=16) 
cbar.ax.tick_params(labelsize=12)


###############################################################################
###############################################################################
###############################################################################


#ax12 = fig.add_axes([0.74, 0.0, 0.32, 0.21])
ax12 = fig.add_axes([0.68, 0.015, 0.295, 0.225])

t12 = ax12.barh(onis12, p12, height=1.0, align='center', color=cmap(norm(c_map12)), edgecolor='dimgrey', alpha=1, zorder=2)
t_12, = ax12.plot([np.nanmean(p12),np.nanmean(p12)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,13,1)
ax12.set_xticks(ticks1)
ticks_cb = ['0','300%','600%','900%','1200%']
ax12.set_xticklabels(ticks_cb, fontsize=12)
#plt.setp(ax12.get_xticklabels(), fontsize=10)

ax12.set_xticks(ax12.get_xticks()[::3])

#ax12.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax12.set_yticks(onis12[::3])
ax12.set_yticklabels(label12[::3], fontsize=10, rotation=0)

#ax12.set_yticks(ax12.get_yticks()[::2])

ax12.set_title('l) Nov Tornadoes; OND ONI',fontsize=16)
#ax12.set_ylabel('Years Ranked by ONI',fontsize=14)
ax12.set_xlabel('% EF1+ Climatological Frequency',fontsize=16)
'''
for i, txt in enumerate(xlabels12):
    plt.annotate(txt, (p12[i], onis12[i]), horizontalalignment='left', verticalalignment='center', rotation=0, fontsize=6, zorder=2)
'''

###############################################################################
###############################################################################
###############################################################################


plt.savefig('john_fig.png',bbox_inches='tight',dpi=200)


###############################################################################
###############################################################################
###############################################################################
