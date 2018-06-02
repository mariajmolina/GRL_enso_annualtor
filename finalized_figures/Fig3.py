#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 13:03:52 2018

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
import xarray as xr
import matplotlib as mpl


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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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
                       oni_demarcator=0.5, gom_quantile=0,
                       composite_type='anomaly', comp_comp='mean',
                       number_of_composite_years=7, 
                       composite_prompt=True,
                       auto_run=False, save_name='full_comp8', region='CONUS',
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

p10 = numtors10/np.nanmean(numtors10)
p11 = numtors11/np.nanmean(numtors11)
p12 = numtors12/np.nanmean(numtors12)



###############################################################################
###############################################################################
###############################################################################


fig = plt.figure(figsize=(12,10.5))

ax1 = fig.add_axes([0.0, 0.3825, 0.295, 0.33])

t1 = ax1.barh(onis1, p1, height=1.0, align='center', color=cmap(norm(c_map1)), edgecolor='dimgrey', alpha=1, zorder=2)
t_1, = ax1.plot([np.nanmean(p1),np.nanmean(p1)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,10,1)
ax1.set_xticks(ticks1)
plt.setp(ax1.get_xticklabels(), fontsize=9, alpha=0)

ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_yticks(onis1[::3])
ax1.set_yticklabels(label1[::3], fontsize=10, rotation=0)

ax1.set_title('d) Dec Tornadoes; NDJ ONI',fontsize=16)
ax1.set_ylabel('Years Ranked by ONI',fontsize=16)

ax1.set_xticks(ax1.get_xticks()[::2])


###############################################################################
###############################################################################
###############################################################################


ax2 = fig.add_axes([0.34, 0.3825, 0.295, 0.33])


t2 = ax2.barh(onis2, p2, height=1.0, align='center', color=cmap(norm(c_map2)), edgecolor='dimgrey', alpha=1, zorder=2)
t_2, = ax2.plot([np.nanmean(p2),np.nanmean(p2)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,10,1)
ax2.set_xticks(ticks1)
plt.setp(ax2.get_xticklabels(), fontsize=9, alpha=0)

ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax2.set_yticks(onis2[::3])
ax2.set_yticklabels(label2[::3], fontsize=10, rotation=0)

ax2.set_title('e) Jan Tornadoes; DJF ONI',fontsize=16)


###############################################################################
###############################################################################
###############################################################################


ax3 = fig.add_axes([0.68, 0.3825, 0.295, 0.33])


t3 = ax3.barh(onis3, p3, height=1.0, align='center', color=cmap(norm(c_map3)), edgecolor='dimgrey', alpha=1, zorder=2)
t_3, = ax3.plot([np.nanmean(p3),np.nanmean(p3)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,10,1)
ax3.set_xticks(ticks1)
plt.setp(ax3.get_xticklabels(), fontsize=9, alpha=0)

ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax3.set_yticks(onis3[::3])
ax3.set_yticklabels(label3[::3], fontsize=10, rotation=0)
ax3.set_xticks(ax3.get_xticks()[::2])

ax3.set_title('f) Feb Tornadoes; JFM ONI',fontsize=16)


###############################################################################
###############################################################################
###############################################################################


ax4 = fig.add_axes([0.0, 0.015, 0.295, 0.33])


t4 = ax4.barh(onis4, p4, height=1.0, align='center', color=cmap(norm(c_map4)), edgecolor='dimgrey', alpha=1, zorder=2)
t_4, = ax4.plot([np.nanmean(p4),np.nanmean(p4)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,10,1)
ax4.set_xticks(ticks1[::])
ticks_cb = ['0','','200%','','400%','','600%','','800%','']
ax4.set_xticklabels(ticks_cb[::], fontsize=12)[::2]

ax4.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')

ax4.set_yticks(onis4[::3])
ax4.set_yticklabels(label4[::3], fontsize=10, rotation=0)

ax4.set_title('g) Mar Tornadoes; FMA ONI',fontsize=16)
ax4.set_ylabel('Years Ranked by ONI',fontsize=16)
ax4.set_xlabel('% EF1+ Climatological Frequency',fontsize=16)


###############################################################################
###############################################################################
###############################################################################


ax5 = fig.add_axes([0.34, 0.015, 0.295, 0.33])

t5 = ax5.barh(onis5, p5, height=1.0, align='center', color=cmap(norm(c_map5)), edgecolor='dimgrey', alpha=1, zorder=2)
t_5, = ax5.plot([np.nanmean(p5),np.nanmean(p5)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,10,1)
ax5.set_xticks(ticks1)
plt.setp(ax5.get_xticklabels(), fontsize=9, alpha=0)

ax5.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax5.set_yticks(onis5[::3])
ax5.set_yticklabels(label5[::3], fontsize=10, rotation=0)

ax5.set_title('h) Apr Tornadoes; MAM ONI',fontsize=16)


###############################################################################
###############################################################################
###############################################################################


ax6 = fig.add_axes([0.68, 0.015, 0.295, 0.33])

t6 = ax6.barh(onis6, p6, height=1.0, align='center', color=cmap(norm(c_map6)), edgecolor='dimgrey', alpha=1, zorder=2)
t_6, = ax6.plot([np.nanmean(p6),np.nanmean(p6)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,10,1)
ax6.set_xticks(ticks1[::])
ticks_cb = ['0','','200%','','400%','','600%','','800%','']
ax6.set_xticklabels(ticks_cb[::], fontsize=12)[::2]

ax6.tick_params(axis='x', which='both', bottom='on', top='off', labelbottom='on')

ax6.set_yticks(onis6[::3])
ax6.set_yticklabels(label6[::3], fontsize=10, rotation=0)

ax6.set_title('i) May Tornadoes; AMJ ONI',fontsize=16)
    
ax6.set_xlabel('% EF1+ Climatological Frequency',fontsize=16)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
    

ax10 = fig.add_axes([0.0, 0.75, 0.295, 0.33]) 


t10 = ax10.barh(onis10, p10, height=1.0, align='center', color=cmap(norm(c_map10)), edgecolor='dimgrey', alpha=1, zorder=2)
t_10, = ax10.plot([np.nanmean(p10),np.nanmean(p10)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)


ticks1 = np.arange(0,10,1)
ax10.set_xticks(ticks1)
plt.setp(ax10.get_xticklabels(), fontsize=9, alpha=0)

ax10.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax10.set_yticks(onis10[::3])
ax10.set_yticklabels(label10[::3], fontsize=10, rotation=0)

ax10.set_title('a) Sep Tornadoes; ASO ONI',fontsize=16)
ax10.set_ylabel('Years Ranked by ONI',fontsize=16)



###############################################################################
###############################################################################
###############################################################################
    

ax11 = fig.add_axes([0.34, 0.75, 0.295, 0.33])


t11 = ax11.barh(onis11, p11, height=1.0, align='center', color=cmap(norm(c_map11)), edgecolor='dimgrey', alpha=1, zorder=2)
t_11, = ax11.plot([np.nanmean(p11),np.nanmean(p11)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,10,1)
ax11.set_xticks(ticks1)
plt.setp(ax11.get_xticklabels(), fontsize=9, alpha=0)

ax11.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax11.set_yticks(onis11[::3])
ax11.set_yticklabels(label11[::3], fontsize=10, rotation=0)

ax11.set_title('b) Oct Tornadoes; SON ONI',fontsize=16)


cbar_ax = fig.add_axes([0.34, 0.003, 0.295, 0.0095])


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


ax12 = fig.add_axes([0.68, 0.75, 0.295, 0.33])


t12 = ax12.barh(onis12, p12, height=1.0, align='center', color=cmap(norm(c_map12)), edgecolor='dimgrey', alpha=1, zorder=2)
t_12, = ax12.plot([np.nanmean(p12),np.nanmean(p12)],[0,63], linewidth=2.5, linestyle='--', color='k', alpha=0.5)

ticks1 = np.arange(0,9,1)
ax12.set_xticks(ticks1)
ticks_cb = ['0','200%','400%','600%','800%']
ax12.set_xticklabels(ticks_cb, fontsize=12)


ax12.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax12.set_yticks(onis12[::3])
ax12.set_yticklabels(label12[::3], fontsize=10, rotation=0)


ax12.set_title('c) Nov Tornadoes; OND ONI',fontsize=16)


###############################################################################
###############################################################################
###############################################################################


plt.savefig('fig3_ensoannual.eps',bbox_inches='tight',dpi=200)

#plt.savefig('john_fig.png',bbox_inches='tight',dpi=200)


###############################################################################
###############################################################################
###############################################################################

