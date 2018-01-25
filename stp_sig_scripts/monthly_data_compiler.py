#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 02:41:36 2018

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


###############################################################################
###############################################################################
###############################################################################
 

month_name = np.array(['jan','feb','mar','apr','may','jun',
                       'jul','aug','sep','oct','nov','dec'], dtype=str)
    

###############################################################################
###############################################################################
###############################################################################



for _ in xrange(500):
    
    neut_dict = {}
    nina_dict = {}
    nino_dict = {}    


    for i, m in enumerate(month_name):

        with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+m+'/neutnina_month_'+m+'_'+str(_+9501), 'rb') as f:
            neut_dict[i] = pickle.load(f)
    
        with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+m+'/nina_month_'+m+'_'+str(_+9501), 'rb') as f:
            nina_dict[i] = pickle.load(f)
    
        with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/'+m+'/nino_month_'+m+'_'+str(_+9501), 'rb') as f:
            nino_dict[i] = pickle.load(f)
            
        if i == 0:
            
            neut_0 = neut_dict[i]
            nina_0 = nina_dict[i]
            nino_0 = nino_dict[i]
            
        elif i == 1:
            
            neut_1 = neut_dict[i]
            nina_1 = nina_dict[i]
            nino_1 = nino_dict[i]

            neut_stack = np.concatenate([neut_0,neut_1], axis=1) 
            nina_stack = np.concatenate([nina_0,nina_1], axis=1) 
            nino_stack = np.concatenate([nino_0,nino_1], axis=1) 
            
        else:
            
            neut_more = neut_dict[i]
            nina_more = nina_dict[i]
            nino_more = nino_dict[i]            

            neut_stack = np.concatenate([neut_stack,neut_more], axis=1) 
            nina_stack = np.concatenate([nina_stack,nina_more], axis=1) 
            nino_stack = np.concatenate([nino_stack,nino_more], axis=1)             


    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/annual/neut_'+str(_+9501), 'wb') as output:
        pickle.dump(neut_stack, output, pickle.HIGHEST_PROTOCOL)
        
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/annual/nina_'+str(_+9501), 'wb') as output:
        pickle.dump(nina_stack, output, pickle.HIGHEST_PROTOCOL)
        
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/annual/nino_'+str(_+9501), 'wb') as output:
        pickle.dump(nino_stack, output, pickle.HIGHEST_PROTOCOL)
    
    
    
    print str(_+9501)+' complete...'
    
    
            
###############################################################################
###############################################################################
###############################################################################
            
   
    
    
'''
    sum_files_neut = np.concatenate([neut_stack,neut_stack,neut_stack], axis=1)
    sum_files_nina = np.concatenate([nina_stack,nina_stack,nina_stack], axis=1)
    sum_files_nino = np.concatenate([nino_stack,nino_stack,nino_stack], axis=1)

    print str(_)+' complete...'
    
    for c, v in product(xrange(len(sum_files_neut[0,0,:,0])),xrange(len(sum_files_neut[0,0,0,:]))):
        
        sum_files_neut[:,:,c,v] = gfilt(sum_files_neut[:,:,c,v]*1.0, sigma=15.0)
        sum_files_nina[:,:,c,v] = gfilt(sum_files_nina[:,:,c,v]*1.0, sigma=15.0)
        sum_files_nino[:,:,c,v] = gfilt(sum_files_nino[:,:,c,v]*1.0, sigma=15.0)
        
        
    sliced_neut = sum_files_neut[:,len(neut_stack[0,:,0,0]):len(neut_stack[0,:,0,0])*2,:,:]
    sliced_nina = sum_files_nina[:,len(nina_stack[0,:,0,0]):len(nina_stack[0,:,0,0])*2,:,:]
    sliced_nino = sum_files_nino[:,len(nino_stack[0,:,0,0]):len(nino_stack[0,:,0,0])*2,:,:]
                
    ready_neut = np.divide(sliced_neut,np.sum(sliced_neut, axis=1))
    ready_nina = np.divide(sliced_nina,np.sum(sliced_nina, axis=1))
    ready_nino = np.divide(sliced_nino,np.sum(sliced_nino, axis=1))
        
    gauss_neut = np.ndarray.argmax(ready_neut, axis=1)
    gauss_nina = np.ndarray.argmax(ready_nina, axis=1)
    gauss_nino = np.ndarray.argmax(ready_nino, axis=1)
'''    
    

###############################################################################
###############################################################################
###############################################################################

