#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:27:12 2018

Maria J. Molina
Central Michigan University
Ph.D. Student

"""


###############################################################################
###############################################################################
###############################################################################



import numpy as np
from scipy.ndimage import gaussian_filter as gfilt


neut_stp_stack = np.load('neut_stp_stack.npy')

for _ in xrange(10000):
    
    neut_stp_stack[_,:,:] = gfilt(neut_stp_stack[_,:,:]*1.0, sigma=1.5)
    print str(_+1)
    
np.save('neut_stp_smooth', neut_stp_stack)


###############################################################################
###############################################################################
###############################################################################


import numpy as np
from scipy.ndimage import gaussian_filter as gfilt


nina_stp_stack = np.load('nina_stp_stack.npy')

for _ in xrange(10000):
    
    nina_stp_stack[_,:,:] = gfilt(nina_stp_stack[_,:,:]*1.0, sigma=1.5)
    print str(_+1)
    
np.save('nina_stp_smooth', nina_stp_stack)


###############################################################################
###############################################################################
###############################################################################


import numpy as np
from scipy.ndimage import gaussian_filter as gfilt


nino_stp_stack = np.load('nino_stp_stack.npy')

for _ in xrange(10000):
    
    nino_stp_stack[_,:,:] = gfilt(nino_stp_stack[_,:,:]*1.0, sigma=1.5)
    print str(_+1)
    
np.save('nino_stp_smooth', nino_stp_stack)


###############################################################################
###############################################################################
###############################################################################







