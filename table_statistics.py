#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 22:32:00 2018

@author: timme1mj
"""




data_neut4 = np.cumsum(Gauss_SmoothTN2)
data_nino4 = np.cumsum(Gauss_SmoothTELN2)
data_nina4 = np.cumsum(Gauss_SmoothTLN2)


per_N = np.divide(data_neut4,np.sum(Gauss_SmoothTN2))
 
round_N = np.around(per_N, decimals=3)

ten_N = np.where(round_N==0.099)[0]
tn5_N = np.where(round_N==0.248)[0]
fif_N = np.where(round_N==0.503)[0]
peak_N = np.where(Gauss_SmoothTN2==np.nanmax(Gauss_SmoothTN2))[0]
sv5_N = np.where(round_N==0.75)[0]
nin_N = np.where(round_N==0.9)[0]
 
print ten_N
print tn5_N
print fif_N
print peak_N
print sv5_N
print nin_N
    
    
    
per_EN = np.divide(data_nino4,np.sum(Gauss_SmoothTELN2))
per_LN = np.divide(data_nina4,np.sum(Gauss_SmoothTLN2))

round_EN = np.around(per_EN, decimals=3)
round_LN = np.around(per_LN, decimals=3)

ten_EN = np.where(round_EN==0.101)[0]
tn5_EN = np.where(round_EN==0.252)[0]
fif_EN = np.where(round_EN==0.498)[0]
peak_EN = np.where(Gauss_SmoothTELN2==np.nanmax(Gauss_SmoothTELN2))[0]
sv5_EN = np.where(round_EN==0.749)[0]
nin_EN = np.where(round_EN==0.90)[0]
 
print ten_EN - ten_N
print tn5_EN - tn5_N
print fif_EN - fif_N
print peak_EN - peak_N
print sv5_EN - sv5_N
print nin_EN - nin_N

ten_LN = np.where(round_LN==0.101)[0]
tn5_LN = np.where(round_LN==0.249)[0]
fif_LN = np.where(round_LN==0.50)[0]
peak_LN = np.where(Gauss_SmoothTLN2==np.nanmax(Gauss_SmoothTLN2))[0]
sv5_LN = np.where(round_LN==0.751)[0]
nin_LN = np.where(round_LN==0.9)[0]

print ten_LN - ten_N
print tn5_LN - tn5_N
print fif_LN - fif_N
print peak_LN - peak_N
print sv5_LN - sv5_N
print nin_LN - nin_N
    
    
    
    

sliced_gaussN_5 = sliced_gaussN5/np.sum(sliced_gaussN5)
sliced_gaussEN_5 = sliced_gaussEN5/np.sum(sliced_gaussEN5)
sliced_gaussLN_5 = sliced_gaussLN5/np.sum(sliced_gaussLN5)


data_neut4 = np.cumsum(sliced_gaussN_5)
data_nino4 = np.cumsum(sliced_gaussEN_5)
data_nina4 = np.cumsum(sliced_gaussLN_5)


per_N = np.divide(data_neut4,np.sum(sliced_gaussN_5))
 
round_N = np.around(per_N, decimals=3)

ten_N = np.where(round_N==0.1)[0]
tn5_N = np.where(round_N==0.249)[0]
fif_N = np.where(round_N==0.502)[0]
peak_N = np.where(sliced_gaussN_5==np.nanmax(sliced_gaussN_5))[0]
sv5_N = np.where(round_N==0.749)[0]
nin_N = np.where(round_N==0.899)[0]
 
print ten_N
print tn5_N
print fif_N
print peak_N
print sv5_N
print nin_N
    
    
    
per_EN = np.divide(data_nino4,np.sum(sliced_gaussEN_5))
per_LN = np.divide(data_nina4,np.sum(sliced_gaussLN_5))

round_EN = np.around(per_EN, decimals=3)
round_LN = np.around(per_LN, decimals=3)

ten_EN = np.where(round_EN==0.10)[0]
tn5_EN = np.where(round_EN==0.251)[0]
fif_EN = np.where(round_EN==0.503)[0]
peak_EN = np.where(sliced_gaussEN_5==np.nanmax(sliced_gaussEN_5))[0]
sv5_EN = np.where(round_EN==0.751)[0]
nin_EN = np.where(round_EN==0.899)[0]
 
print ten_EN - ten_N
print tn5_EN - tn5_N
print fif_EN - fif_N
print peak_EN - peak_N
print sv5_EN - sv5_N
print nin_EN - nin_N

ten_LN = np.where(round_LN==0.10)[0]
tn5_LN = np.where(round_LN==0.249)[0]
fif_LN = np.where(round_LN==0.498)[0]
peak_LN = np.where(sliced_gaussLN_5==np.nanmax(sliced_gaussLN_5))[0]
sv5_LN = np.where(round_LN==0.751)[0]
nin_LN = np.where(round_LN==0.9)[0]

print ten_LN - ten_N
print tn5_LN - tn5_N
print fif_LN - fif_N
print peak_LN - peak_N
print sv5_LN - sv5_N
print nin_LN - nin_N













sliced_gaussN_6 = sliced_gaussN6/np.sum(sliced_gaussN6)
sliced_gaussEN_6 = sliced_gaussEN6/np.sum(sliced_gaussEN6)
sliced_gaussLN_6 = sliced_gaussLN6/np.sum(sliced_gaussLN6)


data_neut4 = np.cumsum(sliced_gaussN_6)
data_nino4 = np.cumsum(sliced_gaussEN_6)
data_nina4 = np.cumsum(sliced_gaussLN_6)


per_N = np.divide(data_neut4,np.sum(sliced_gaussN_6))
 
round_N = np.around(per_N, decimals=3)

ten_N = np.where(round_N==0.101)[0]
tn5_N = np.where(round_N==0.25)[0]
fif_N = np.where(round_N==0.498)[0]
peak_N = np.where(sliced_gaussN_6==np.nanmax(sliced_gaussN_6))[0]
sv5_N = np.where(round_N==0.749)[0]
nin_N = np.where(round_N==0.9)[0]
 
print ten_N
print tn5_N
print fif_N
print peak_N
print sv5_N
print nin_N
    
    
    
per_EN = np.divide(data_nino4,np.sum(sliced_gaussEN_6))
per_LN = np.divide(data_nina4,np.sum(sliced_gaussLN_6))

round_EN = np.around(per_EN, decimals=3)
round_LN = np.around(per_LN, decimals=3)

ten_EN = np.where(round_EN==0.099)[0]
tn5_EN = np.where(round_EN==0.253)[0]
fif_EN = np.where(round_EN==0.498)[0]
peak_EN = np.where(sliced_gaussEN_6==np.nanmax(sliced_gaussEN_6))[0]
sv5_EN = np.where(round_EN==0.751)[0]
nin_EN = np.where(round_EN==0.9)[0]
 
print ten_EN - ten_N
print tn5_EN - tn5_N
print fif_EN - fif_N
print peak_EN - peak_N
print sv5_EN - sv5_N
print nin_EN - nin_N

ten_LN = np.where(round_LN==0.101)[0]
tn5_LN = np.where(round_LN==0.253)[0]
fif_LN = np.where(round_LN==0.497)[0]
peak_LN = np.where(sliced_gaussLN_6==np.nanmax(sliced_gaussLN_6))[0]
sv5_LN = np.where(round_LN==0.749)[0]
nin_LN = np.where(round_LN==0.9)[0]

print ten_LN - ten_N
print tn5_LN - tn5_N
print fif_LN - fif_N
print peak_LN - peak_N
print sv5_LN - sv5_N
print nin_LN - nin_N









data_neut4 = np.cumsum(sliced_gaussN)
data_nino4 = np.cumsum(sliced_gaussEN)
data_nina4 = np.cumsum(sliced_gaussLN)


per_N = np.divide(data_neut4,np.sum(sliced_gaussN))
 
round_N = np.around(per_N, decimals=3)

ten_N = np.where(round_N==0.099)[0]
tn5_N = np.where(round_N==0.249)[0]
fif_N = np.where(round_N==0.498)[0]
peak_N = np.where(sliced_gaussN==np.nanmax(sliced_gaussN))[0]
sv5_N = np.where(round_N==0.75)[0]
nin_N = np.where(round_N==0.9)[0]
 
print ten_N
print tn5_N
print fif_N
print peak_N
print sv5_N
print nin_N
    
    
    
per_EN = np.divide(data_nino4,np.sum(sliced_gaussEN))
per_LN = np.divide(data_nina4,np.sum(sliced_gaussLN))

round_EN = np.around(per_EN, decimals=3)
round_LN = np.around(per_LN, decimals=3)

ten_EN = np.where(round_EN==0.099)[0]
tn5_EN = np.where(round_EN==0.247)[0]
fif_EN = np.where(round_EN==0.499)[0]
peak_EN = np.where(sliced_gaussEN==np.nanmax(sliced_gaussEN))[0]
sv5_EN = np.where(round_EN==0.749)[0]
nin_EN = np.where(round_EN==0.9)[0]
 
print ten_EN - ten_N
print tn5_EN - tn5_N
print fif_EN - fif_N
print peak_EN - peak_N
print sv5_EN - sv5_N
print nin_EN - nin_N

ten_LN = np.where(round_LN==0.098)[0]
tn5_LN = np.where(round_LN==0.252)[0]
fif_LN = np.where(round_LN==0.497)[0]
peak_LN = np.where(sliced_gaussLN==np.nanmax(sliced_gaussLN))[0]
sv5_LN = np.where(round_LN==0.75)[0]
nin_LN = np.where(round_LN==0.899)[0]

print ten_LN - ten_N
print tn5_LN - tn5_N
print fif_LN - fif_N
print peak_LN - peak_N
print sv5_LN - sv5_N
print nin_LN - nin_N


    
