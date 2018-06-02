#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:39:16 2017

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
#import pickle


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
############################ALL YEARS##########################################
###############################################################################


data = xr.open_dataset('tors_5316_ef1', decode_cf=True)

num_yrs = len(data.grid.groupby('time.year').sum('time').year.values)

sum_tors = data.grid.groupby('time.dayofyear').sum('time')

three_sum_tors = xr.concat([sum_tors,sum_tors,sum_tors], dim='dayofyear')

Gauss_SmoothAN = np.divide(three_sum_tors, num_yrs)

for i, j in product(xrange(len(Gauss_SmoothAN[0,:,0])),xrange(len(Gauss_SmoothAN[0,0,:]))):
    
    Gauss_SmoothAN[:,i,j] = gfilt(Gauss_SmoothAN[:,i,j]*1.0,sigma=15.0)
    
    
sliced_gauss = Gauss_SmoothAN[(549+31):(len(sum_tors[:,0,0])*2)+31,:,:]

sliced_gauss = np.divide(sliced_gauss,np.sum(sliced_gauss, axis=0))

gauss_peak_AN = np.ndarray.argmax(sliced_gauss.values,axis=0)


###############################################################################
###############################################################################
###############################################################################


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
    
    
for k in xrange(12):
    
    num_yrs_neut = len(neut_comp[k].groupby('time.year').sum('time').year.values)
    num_yrs_nino = len(nino_comp[k].groupby('time.year').sum('time').year.values)
    num_yrs_nina = len(nina_comp[k].groupby('time.year').sum('time').year.values)

    if k == 0:
    
        sum_tors_neut = neut_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_neut = np.divide(sum_tors_neut, num_yrs_neut)
        
        sum_tors_nino = nino_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nino = np.divide(sum_tors_nino, num_yrs_nino)
        
        sum_tors_nina = nina_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nina = np.divide(sum_tors_nina, num_yrs_nina)
    
    elif k == 1:
        
        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
        
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)

        sum_group_neut = xr.concat([sum_tors_neut, sum_tors_neut_2], dim=('dayofyear'))        
        sum_group_nino = xr.concat([sum_tors_nino, sum_tors_nino_2], dim=('dayofyear'))   
        sum_group_nina = xr.concat([sum_tors_nina, sum_tors_nina_2], dim=('dayofyear'))   
        
    else:

        sum_tors_neut_2 = neut_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_neut_2 = np.divide(sum_tors_neut_2, num_yrs_neut)
        
        wut1 = sum_tors_neut_2[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
        sum_tors_neut_stack = np.vstack([sum_tors_neut_2[:-2,:,:].values,wut1.values])
        sum_tors_neut_2 = xr.Dataset({'grid': (['dayofyear','y','x'], sum_tors_neut_stack)},
                                      coords={'dayofyear': sum_tors_neut_2.coords['dayofyear'].values[:-1]})     
                
        sum_tors_nino_2 = nino_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nino_2 = np.divide(sum_tors_nino_2, num_yrs_nino)
        
        wut2 = sum_tors_nino_2[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
        sum_tors_nino_stack = np.vstack([sum_tors_nino_2[:-2,:,:].values,wut2.values])
        sum_tors_nino_2 = xr.Dataset({'grid': (['dayofyear','y','x'], sum_tors_nino_stack)},
                                      coords={'dayofyear': sum_tors_nino_2.coords['dayofyear'].values[:-1]})         
        
        sum_tors_nina_2 = nina_comp[k].groupby('time.dayofyear').sum('time')
        sum_tors_nina_2 = np.divide(sum_tors_nina_2, num_yrs_nina)
        
        wut3 = sum_tors_nina_2[-2:,:,:].mean('dayofyear', skipna=True).expand_dims('dayofyear',axis=0)
        sum_tors_nina_stack = np.vstack([sum_tors_nina_2[:-2,:,:].values,wut3.values])
        sum_tors_nina_2 = xr.Dataset({'grid': (['dayofyear','y','x'], sum_tors_nina_stack)},
                                      coords={'dayofyear': sum_tors_nina_2.coords['dayofyear'].values[:-1]}) 

        sum_group_neut = xr.concat([sum_group_neut, sum_tors_neut_2.grid], dim=('dayofyear'))        
        sum_group_nino = xr.concat([sum_group_nino, sum_tors_nino_2.grid], dim=('dayofyear'))   
        sum_group_nina = xr.concat([sum_group_nina, sum_tors_nina_2.grid], dim=('dayofyear'))       


###############################################################################
###############################################################################
###############################################################################

                   
Gauss_SmoothN = xr.concat([sum_group_neut,sum_group_neut,sum_group_neut], dim='dayofyear')
Gauss_SmoothEN = xr.concat([sum_group_nino,sum_group_nino,sum_group_nino], dim='dayofyear')
Gauss_SmoothLN = xr.concat([sum_group_nina,sum_group_nina,sum_group_nina], dim='dayofyear')
    
for i, j in product(xrange(len(Gauss_SmoothN[0,:,0])),xrange(len(Gauss_SmoothN[0,0,:]))):
    
    Gauss_SmoothN[:,i,j] = gfilt(Gauss_SmoothN[:,i,j]*1.0,sigma=15.0)
    Gauss_SmoothEN[:,i,j] = gfilt(Gauss_SmoothEN[:,i,j]*1.0,sigma=15.0)
    Gauss_SmoothLN[:,i,j] = gfilt(Gauss_SmoothLN[:,i,j]*1.0,sigma=15.0)


sliced_gaussN = Gauss_SmoothN[(549+31):(len(sum_tors[:,0,0])*2)+31,:,:]
sliced_gaussEN = Gauss_SmoothEN[(549+31):(len(sum_tors[:,0,0])*2)+31,:,:]
sliced_gaussLN = Gauss_SmoothLN[(549+31):(len(sum_tors[:,0,0])*2)+31,:,:]

sliced_gaussN = np.divide(sliced_gaussN,np.sum(sliced_gaussN, axis=0))
sliced_gaussEN = np.divide(sliced_gaussEN,np.sum(sliced_gaussEN, axis=0))
sliced_gaussLN = np.divide(sliced_gaussLN,np.sum(sliced_gaussLN, axis=0))

gauss_peak_N = np.ndarray.argmax(sliced_gaussN.values,axis=0)
gauss_peak_EN = np.ndarray.argmax(sliced_gaussEN.values,axis=0)
gauss_peak_LN = np.ndarray.argmax(sliced_gaussLN.values,axis=0)
    


###############################################################################
###############################################################################
###############################################################################


latlon = xr.open_dataset('tor_grid_latlon', decode_cf=False)

gauss_smooth = gfilt(gauss_peak_AN*1.0, sigma=1.5)
gauss_smooth_N = gfilt(gauss_peak_N*1.0, sigma=1.5)
gauss_smooth_EN = gfilt(gauss_peak_EN*1.0, sigma=1.5)
gauss_smooth_LN = gfilt(gauss_peak_LN*1.0, sigma=1.5)


gauss_smooth[gauss_smooth<0.05] = None
gauss_smooth_N[gauss_smooth_N<0.05] = None
gauss_smooth_EN[gauss_smooth_EN<0.05] = None
gauss_smooth_LN[gauss_smooth_LN<0.05] = None


llcrnrlon = -120
llcrnrlat = 15
urcrnrlon = -60
urcrnrlat = 50

m = Basemap(projection='lcc', lat_0 = 39, lon_0 = -96, lat_1 = 40,
            llcrnrlon = llcrnrlon, llcrnrlat = llcrnrlat,
            urcrnrlat = urcrnrlat, urcrnrlon = urcrnrlon,
            resolution='l')

for i, j in product(xrange(len(gauss_smooth[:,0])),xrange(len(gauss_smooth[0,:]))):
    
    y = latlon.lats.values[i]
    x = latlon.lons.values[j]
    
    if not m.is_land(x,y):
        
        gauss_smooth[i,j] = None
        gauss_smooth_N[i,j] = None
        gauss_smooth_EN[i,j] = None
        gauss_smooth_LN[i,j] = None        
    

###############################################################################
###############################################################################
###############################################################################


g_mask = np.ma.masked_invalid(gauss_smooth_N)
mask_help = np.sum(np.vstack([np.array_split(sum_group_neut,[31,213])[0].values, np.array_split(sum_group_neut,[31,213])[2].values]), axis=0)
g_mask3 = np.ma.masked_array(g_mask, mask_help==0)

g_mask = np.ma.masked_invalid(gauss_smooth_EN)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nino,[31,213])[0].values, np.array_split(sum_group_nino,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help==0)

g_mask = np.ma.masked_invalid(gauss_smooth_LN)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nina,[31,213])[0].values, np.array_split(sum_group_nina,[31,213])[2].values]), axis=0)
g_mask4 = np.ma.masked_array(g_mask, mask_help==0)


g_nino = g_mask2.data - g_mask3.data
g_nina = g_mask4.data - g_mask3.data


###############################################################################
###############################################################################
###############################################################################


neut_concat = np.zeros([10000, 51, 66])
nina_concat = np.zeros([10000, 51, 66])
nino_concat = np.zeros([10000, 51, 66])


for _ in xrange(10000):


    neut_file = np.load('/storage/timme1mj/maria_pysplit/torobs_sig/sliced_neut_'+str(_)+'.npy')
    nina_file = np.load('/storage/timme1mj/maria_pysplit/torobs_sig/sliced_nina_'+str(_)+'.npy')
    nino_file = np.load('/storage/timme1mj/maria_pysplit/torobs_sig/sliced_nino_'+str(_)+'.npy')
    
    neut_file = np.vstack([neut_file[-152:,:,:],neut_file[:31,:,:]])
    nina_file = np.vstack([nina_file[-152:,:,:],nina_file[:31,:,:]])
    nino_file = np.vstack([nino_file[-152:,:,:],nino_file[:31,:,:]])

    neut_concat[_,:,:] = np.ndarray.argmax(neut_file, axis=0)
    nina_concat[_,:,:] = np.ndarray.argmax(nina_file, axis=0)
    nino_concat[_,:,:] = np.ndarray.argmax(nino_file, axis=0)
   
    print _


for i in xrange(10000):
    
    neut_concat[i,:,:] = gfilt(neut_concat[i,:,:]*1.0, sigma=1.5)
    nina_concat[i,:,:] = gfilt(nina_concat[i,:,:]*1.0, sigma=1.5)
    nino_concat[i,:,:] = gfilt(nino_concat[i,:,:]*1.0, sigma=1.5)
    
    print str(i)+' completed...'


###############################################################################
###############################################################################
###############################################################################


nina_diff = nina_concat - neut_concat
nino_diff = nino_concat - neut_concat


nina_upper = np.nanpercentile(nina_diff,97.5,axis=0)
nina_lower = np.nanpercentile(nina_diff,2.5,axis=0)
nino_upper = np.nanpercentile(nino_diff,97.5,axis=0)
nino_lower = np.nanpercentile(nino_diff,2.5,axis=0)


nina_sign = np.zeros(g_nino.shape)
nino_sign = np.zeros(g_nino.shape)


for i, j in product(xrange(len(g_nino[:,0])),xrange(len(g_nino[0,:]))):
    
    if np.isfinite(g_nino[i,j]):
        
        if g_nino[i,j] >= nino_upper[i,j]:
            
            nino_sign[i,j] = 1
            
        if g_nino[i,j] <= nino_lower[i,j]:

            nino_sign[i,j] = 1

    if np.isfinite(g_nina[i,j]):
        
        if g_nina[i,j] >= nina_upper[i,j]:
            
            nina_sign[i,j] = 1
            
        if g_nina[i,j] <= nina_lower[i,j]:

            nina_sign[i,j] = 1


###############################################################################
###############################################################################
###############################################################################
        
        
maria_color = ({0.:'rebeccapurple',
                0.1:'midnightblue',
                0.2:'mediumblue',
                0.28:'cornflowerblue',
                0.38:'lightsteelblue',
                0.46:'darkgrey',
                0.52:'darkgoldenrod',
                0.57:'goldenrod',
                0.63:'lightsalmon',
                0.7:'salmon',
                0.77:'red',
                0.83:'darkred',
                0.9:'k',
                1.0:'k'})
                

cmap = make_colormap(maria_color)

'''

maria_color = ({0.:'rebeccapurple',
                0.0833:'midnightblue',
                0.1666:'mediumblue',
                0.25:'cornflowerblue',
                0.3333:'lightsteelblue',
                0.4166:'darkgrey',
                0.5:'darkgoldenrod',
                0.5833:'goldenrod',
                0.6666:'lightsalmon',
                0.75:'salmon',
                0.8333:'red',
                0.9166:'darkred',
                1.0:'k'})
                

cmap = make_colormap(maria_color)
'''


###############################################################################
###############################################################################
###############################################################################


x1, y1 = latlon.lons.values, latlon.lats.values

x2 = np.zeros(x1.shape)
y2 = np.zeros(y1.shape)
x3 = np.zeros(x1.shape)
y3 = np.zeros(y1.shape)

for i, j in product(xrange(len(x1)),xrange(len(y1))):
    
    x2[i], y2[j] = m(x1[i], y1[j], inverse=True)
        
    x2[i] = x2[i]-0.4
    y2[j] = y2[j]-0.4
    
    x3[i], y3[j] = m(x2[i], y2[j])


###############################################################################
###############################################################################
###############################################################################
    

fig = plt.figure(figsize=(9.25,12))


ax1 = fig.add_axes([0.0, 0.666, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth)
mask_help = np.sum(np.vstack([np.array_split(sum_tors,[31,213])[0].values, np.array_split(sum_tors,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help==0)

cs = ax1.pcolormesh(x3, y3, g_mask2, vmin=0, vmax=183, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.0, fontsize=11)

ax1.set_title(u'a) All Years', fontsize=13, loc='center')
    

ax2 = fig.add_axes([0.5, 0.666, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_N)
mask_help = np.sum(np.vstack([np.array_split(sum_group_neut,[31,213])[0].values, np.array_split(sum_group_neut,[31,213])[2].values]), axis=0)
g_mask3 = np.ma.masked_array(g_mask, mask_help==0)

cs = ax2.pcolormesh(x3, y3, g_mask3, vmin=0, vmax=183, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax2.set_title(u'b) ENSO-Neutral Years', fontsize=13, loc='center')


ax3 = fig.add_axes([0.0, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nino,[31,213])[0].values, np.array_split(sum_group_nino,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help==0)

cs = ax3.pcolormesh(x3, y3, g_mask2, vmin=0, vmax=183, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.0, fontsize=11)
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.0, fontsize=11)

ax3.set_title(u'c) El Ni\xf1o Years', fontsize=13, loc='center')


ax4 = fig.add_axes([0.5, 0.3425, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_EN)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nino,[31,213])[0].values, np.array_split(sum_group_nino,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help==0)

cs3 = ax4.pcolormesh(x3, y3, np.divide((g_mask2-g_mask3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

x, y = x3, y3
scat_lats = np.zeros(len(y3)-1)
scat_lons = np.zeros(len(x3)-1)
for a in xrange(len(scat_lats)):
    scat_lats[a] = np.nanmean([y3[a],y3[a+1]])
for b in xrange(len(scat_lons)):
    scat_lons[b] = np.nanmean([x3[b],x3[b+1]])
scat_lons, scat_lats = np.meshgrid(scat_lons, scat_lats)

g_mask = np.ma.masked_invalid(nino_sign)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nino,[31,213])[0].values, np.array_split(sum_group_nino,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help<=0.6)

ax4.scatter(scat_lons, scat_lats, g_mask2*8, facecolors='w', edgecolors='k', zorder=10)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawmeridians(np.arange(int(-110),int(-65),10),labels=[0,0,0,1], linewidth=0.0, fontsize=11)

ax4.set_title(u'd) El Ni\xf1o - ENSO-Neutral Years', fontsize=13, loc='center')


ax5 = fig.add_axes([0.0, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nina,[31,213])[0].values, np.array_split(sum_group_nina,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help==0)

cs2 = ax5.pcolormesh(x3, y3, g_mask2, vmin=0, vmax=183, cmap=cmap)

m.drawcoastlines()
m.drawstates()
m.drawcountries()
m.drawparallels(np.arange(int(20),int(51),5),labels=[1,0,0,0], linewidth=0.0, fontsize=11)

ax5.set_title(u'e) La Ni\xf1a Years', fontsize=13, loc='center')

    
ax6 = fig.add_axes([0.5, 0.005, 0.5, 0.33])

g_mask = np.ma.masked_invalid(gauss_smooth_LN)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nina,[31,213])[0].values, np.array_split(sum_group_nina,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help==0)

cs = ax6.pcolormesh(x3, y3, np.divide((g_mask2-g_mask3),7), vmin=-10, vmax=10, cmap=plt.cm.get_cmap('bwr'))

g_mask = np.ma.masked_invalid(nina_sign)
mask_help = np.sum(np.vstack([np.array_split(sum_group_nina,[31,213])[0].values, np.array_split(sum_group_nina,[31,213])[2].values]), axis=0)
g_mask2 = np.ma.masked_array(g_mask, mask_help<=0.6)

ax6.scatter(scat_lons, scat_lats, g_mask2*8, facecolors='w', edgecolors='k', zorder=10)

m.drawcoastlines()
m.drawstates()
m.drawcountries()

ax6.set_title(u'f) La Ni\xf1a - ENSO-Neutral Years', fontsize=13, loc='center')
    

cbar_ax = fig.add_axes([0.01, 0.0, 0.48, 0.015])
ticks_1 = [0,30.5,61,91.5,122,152.5,183]
tick_1 = ['Aug','Sep','Oct','Nov','Dec','Jan']
cbar = fig.colorbar(cs2, cax=cbar_ax, ticks=ticks_1, orientation='horizontal')
cbar.ax.set_xticklabels(tick_1)
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Annual Peak Fraction of Tornado Reports EF1+', fontsize=12) 
#cbar.set_label('Annual Peak Fraction of Tornado Reports EF2+', fontsize=12) 

cbar_ax = fig.add_axes([0.51, 0.0, 0.48, 0.015])
cbar = fig.colorbar(cs3, cax=cbar_ax, orientation='horizontal')
cbar.ax.tick_params(labelsize=11)
cbar.set_label('Tornado Report Peak Fraction Difference (weeks)', fontsize=12) 


#plt.savefig('wut_2.png', bbox_inches='tight', dpi=200)
#plt.savefig('wut_2.ps', bbox_inches='tight', dpi=200)
plt.savefig('wut_2.pdf', bbox_inches='tight', dpi=200)
#plt.savefig('wut_2.eps', bbox_inches='tight', dpi=200)

#plt.savefig('wut_3.png', bbox_inches='tight', dpi=200)


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
