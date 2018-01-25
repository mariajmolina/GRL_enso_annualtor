#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

Created on Thu Jan 25 11:19:05 2018

Maria J. Molina
Ph.D. Student
Central Michigan University

"""



###############################################################################
###############################################################################
###############################################################################


import numpy as np
import pickle


neut_stack = np.zeros([10000,277,349])

for _ in xrange(10000):
    
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/gauss/neut_'+str(_+1), 'rb') as f:
        neut_stack_temp = pickle.load(f)
    
    neut_stack[_,:,:] = neut_stack_temp[0,:,:]
    
    
with open('neut_stp_stack', 'wb') as output:
    pickle.dump(neut_stack, output, pickle.HIGHEST_PROTOCOL)       
    

###############################################################################
###############################################################################
###############################################################################
        
      
import numpy as np
import pickle


nina_stack = np.zeros([10000,277,349]) 
           
for _ in xrange(10000):
 
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/gauss/nina_'+str(_+1), 'rb') as f:
        nina_stack_temp = pickle.load(f)
    
    nina_stack[_,:,:] = nina_stack_temp[0,:,:] 


with open('nina_stp_stack', 'wb') as output:
    pickle.dump(nina_stack, output, pickle.HIGHEST_PROTOCOL)
    

###############################################################################
###############################################################################
###############################################################################


import numpy as np
import pickle


nino_stack = np.zeros([10000,277,349])

for _ in xrange(10000):

    
    with open('/storage/timme1mj/maria_pysplit/enso_annual_paper/gauss/nino_'+str(_+1), 'rb') as f:
        nino_stack_temp = pickle.load(f)
    
    nino_stack[_,:,:] = nino_stack_temp[0,:,:]
    
    
with open('nino_stp_stack', 'wb') as output:
    pickle.dump(nino_stack, output, pickle.HIGHEST_PROTOCOL)
    
    
###############################################################################
###############################################################################
###############################################################################

