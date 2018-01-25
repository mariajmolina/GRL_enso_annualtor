#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:45:27 2018

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
from itertools import product
from scipy.ndimage import gaussian_filter as gfilt


###############################################################################
###############################################################################
###############################################################################



for _ in xrange(1000):
    

    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/annual/neut_'+str(_+1), 'rb') as f:
        neut_stack = pickle.load(f)
        
            
    sum_files_neut = np.concatenate([neut_stack,neut_stack,neut_stack], axis=1)

   
    for c, v in product(xrange(len(sum_files_neut[0,0,:,0])),xrange(len(sum_files_neut[0,0,0,:]))):
        
        sum_files_neut[:,:,c,v] = gfilt(sum_files_neut[:,:,c,v]*1.0, sigma=15.0)
        
        
    sliced_neut = sum_files_neut[:,len(neut_stack[0,:,0,0]):len(neut_stack[0,:,0,0])*2,:,:]
                
    ready_neut = np.divide(sliced_neut,np.sum(sliced_neut, axis=1))
        
    gauss_neut = np.ndarray.argmax(ready_neut, axis=1)
   
    
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/gauss/neut_'+str(_+1), 'wb') as output:
        pickle.dump(gauss_neut, output, pickle.HIGHEST_PROTOCOL)
            

    print str(_)+' complete...'
            
            
            
###############################################################################
###############################################################################
###############################################################################
           
           
    

    
    
    
    
    
