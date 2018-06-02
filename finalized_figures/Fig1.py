#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 02:21:34 2018

Maria J. Molina
Ph.D. Student 
Central Michigan University

"""

###############################################################################
###############################################################################
###############################################################################


from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.patches as mpatches


###############################################################################
###############################################################################
###############################################################################



with open('tors_neut1', 'rb') as f:
    tors_neut1 = pickle.load(f)

with open('tors_nino1', 'rb') as f:
    tors_nino1 = pickle.load(f)
    
with open('tors_nina1', 'rb') as f:
    tors_nina1 = pickle.load(f)
    
with open('tors_low1', 'rb') as f:
    tors_low1 = pickle.load(f)
    
with open('tors_high1', 'rb') as f:
    tors_high1 = pickle.load(f) 
    
    

with open('tors_neut2', 'rb') as f:
    tors_neut2 = pickle.load(f)

with open('tors_nino2', 'rb') as f:
    tors_nino2 = pickle.load(f)
    
with open('tors_nina2', 'rb') as f:
    tors_nina2 = pickle.load(f)
    
with open('tors_low2', 'rb') as f:
    tors_low2 = pickle.load(f)
    
with open('tors_high2', 'rb') as f:
    tors_high2 = pickle.load(f)
    
    
    
with open('tors_neut3', 'rb') as f:
    tors_neut3 = pickle.load(f)

with open('tors_nino3', 'rb') as f:
    tors_nino3 = pickle.load(f)
    
with open('tors_nina3', 'rb') as f:
    tors_nina3 = pickle.load(f)
    
with open('tors_low3', 'rb') as f:
    tors_low3 = pickle.load(f)
    
with open('tors_high3', 'rb') as f:
    tors_high3 = pickle.load(f)
    
    

with open('tors_neut4', 'rb') as f:
    tors_neut4 = pickle.load(f)

with open('tors_nino4', 'rb') as f:
    tors_nino4 = pickle.load(f)
    
with open('tors_nina4', 'rb') as f:
    tors_nina4 = pickle.load(f)
    
with open('tors_low4', 'rb') as f:
    tors_low4 = pickle.load(f)
    
with open('tors_high4', 'rb') as f:
    tors_high4 = pickle.load(f)
    
    
    
    
with open('torday_neut1', 'rb') as f:
     torday_neut1 = pickle.load(f)

with open('torday_nino1', 'rb') as f:
     torday_nino1 = pickle.load(f)
    
with open('torday_nina1', 'rb') as f:
     torday_nina1 = pickle.load(f)
    
with open('torday_low1', 'rb') as f:
     torday_low1 = pickle.load(f)
    
with open('torday_high1', 'rb') as f:
     torday_high1 = pickle.load(f)    
    
    

with open('torday_neut2', 'rb') as f:
    torday_neut2 = pickle.load(f)

with open('torday_nino2', 'rb') as f:
    torday_nino2 = pickle.load(f)
    
with open('torday_nina2', 'rb') as f:
    torday_nina2 = pickle.load(f)
    
with open('torday_low2', 'rb') as f:
    torday_low2 = pickle.load(f)
    
with open('torday_high2', 'rb') as f:
    torday_high2 = pickle.load(f) 
    
    
    
with open('torday_neut3', 'rb') as f:
    torday_neut3 = pickle.load(f)

with open('torday_nino3', 'rb') as f:
    torday_nino3 = pickle.load(f)
    
with open('torday_nina3', 'rb') as f:
    torday_nina3 = pickle.load(f)
    
with open('torday_low3', 'rb') as f:
    torday_low3 = pickle.load(f)
    
with open('torday_high3', 'rb') as f:
    torday_high3 = pickle.load(f)
    
    

with open('torday_neut4', 'rb') as f:
    torday_neut4 = pickle.load(f)

with open('torday_nino4', 'rb') as f:
    torday_nino4 = pickle.load(f)
    
with open('torday_nina4', 'rb') as f:
    torday_nina4 = pickle.load(f)
    
with open('torday_low4', 'rb') as f:
    torday_low4 = pickle.load(f)
    
with open('torday_high4', 'rb') as f:
    torday_high4 = pickle.load(f)





with open('stp_neut4', 'rb') as f:
    stp_neut4 = pickle.load(f)

with open('stp_nino4', 'rb') as f:
    stp_nino4 = pickle.load(f)
    
with open('stp_nina4', 'rb') as f:
    stp_nina4 = pickle.load(f)
    
with open('stp_low4', 'rb') as f:
    stp_low4 = pickle.load(f)
    
with open('stp_high4', 'rb') as f:
    stp_high4 = pickle.load(f)
    
    
    
with open('stpse_neut4', 'rb') as f:
    stpse_neut4 = pickle.load(f)

with open('stpse_nino4', 'rb') as f:
    stpse_nino4 = pickle.load(f)
    
with open('stpse_nina4', 'rb') as f:
    stpse_nina4 = pickle.load(f)
    
with open('stpse_low4', 'rb') as f:
    stpse_low4 = pickle.load(f)
    
with open('stpse_high4', 'rb') as f:
    stpse_high4 = pickle.load(f)


###############################################################################
###############################################################################
###############################################################################
    
    

grey_patch = mpatches.Patch(color='darkgrey', alpha=0.4, label='All')


fig = plt.figure(figsize=(9.5,12))



ax1 = fig.add_axes([0.0, 0.8, 0.48, 0.1775])

p1, = ax1.plot(range(0,len(tors_nino1)),np.divide(tors_nino1,np.sum(tors_nino1)),'r-',linewidth=2.0)
p2, = ax1.plot(range(0,len(tors_nina1)),np.divide(tors_nina1,np.sum(tors_nina1)),'b-',linewidth=2.0)
p3, = ax1.plot(range(0,len(tors_neut1)),np.divide(tors_neut1,np.sum(tors_neut1)),'k-',linewidth=2.0)    
#p4, = ax1.plot(range(0,366),Gauss_SmoothAN1/np.sum(Gauss_SmoothAN1),'--',color='grey',linewidth=2.0)  

p5 = ax1.fill_between(range(0,len(tors_low1)),tors_low1,tors_high1,color='darkgrey',linewidth=1.0,alpha=0.5)

ax1.set_ylabel('Fraction of Tornadoes (EF1+)', fontsize=10)

ax1.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax1.get_yticklabels(), fontsize=10, rotation=35)
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax1.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax1.set_title('a) Annual Cycle of CONUS EF1+ Tornado Reports', fontsize=11)

ax1.grid(True, linestyle='--', alpha=0.5)





ax2 = fig.add_axes([0.0, 0.6, 0.48, 0.1775]) 

p1, = ax2.plot(range(0,len(tors_nino2)),np.divide(tors_nino2,np.sum(tors_nino2)),'r-',linewidth=2.0)
p2, = ax2.plot(range(0,len(tors_nina2)),np.divide(tors_nina2,np.sum(tors_nina2)),'b-',linewidth=2.0)
p3, = ax2.plot(range(0,len(tors_neut2)),np.divide(tors_neut2,np.sum(tors_neut2)),'k-',linewidth=2.0)

p5 = ax2.fill_between(range(0,len(tors_low2)),tors_low2,tors_high2,color='darkgrey',linewidth=1.0,alpha=0.5)

ax2.set_ylabel('Fraction of Tornadoes (EF2+)', fontsize=10)

ax2.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax2.get_yticklabels(), fontsize=10, rotation=35)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax2.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax2.set_title('b) Annual Cycle of CONUS EF2+ Tornado Reports', fontsize=11)

ax2.grid(True, linestyle='--', alpha=0.5)




ax3 = fig.add_axes([0.0, 0.4, 0.48, 0.1775]) 

p1, = ax3.plot(range(0,len(tors_nino3)),np.divide(tors_nino3,np.sum(tors_nino3)),'r-',linewidth=2.0)
p2, = ax3.plot(range(0,len(tors_nina3)),np.divide(tors_nina3,np.sum(tors_nina3)),'b-',linewidth=2.0)
p3, = ax3.plot(range(0,len(tors_neut3)),np.divide(tors_neut3,np.sum(tors_neut3)),'k-',linewidth=2.0)

p5 = ax3.fill_between(range(0,len(tors_low3)),tors_low3,tors_high3,color='darkgrey',linewidth=1.0,alpha=0.5)

ax3.set_ylabel('Fraction of Tornadoes (EF1+)', fontsize=10)

ax3.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax3.get_yticklabels(), fontsize=10, rotation=35)
#ax3.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax3.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax3.set_title('c) Annual Cycle of Southeast EF1+ Tornado Reports', fontsize=11)

ax3.grid(True, linestyle='--', alpha=0.5)



ax4 = fig.add_axes([0.0, 0.2, 0.48, 0.1775]) 

p1, = ax4.plot(range(0,len(tors_nino4)),np.divide(tors_nino4,np.sum(tors_nino4)),'r-',linewidth=2.0)
p2, = ax4.plot(range(0,len(tors_nina4)),np.divide(tors_nina4,np.sum(tors_nina4)),'b-',linewidth=2.0)
p3, = ax4.plot(range(0,len(tors_neut4)),np.divide(tors_neut4,np.sum(tors_neut4)),'k-',linewidth=2.0)

p5 = ax4.fill_between(range(0,len(tors_low4)),tors_low4,tors_high4,color='darkgrey',linewidth=1.0,alpha=0.5) 

ax4.set_ylabel('Fraction of Tornadoes (EF2+)', fontsize=10)

ax4.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax4.get_yticklabels(), fontsize=10, rotation=35)
#ax4.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax4.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')

ax4.set_title('d) Annual Cycle of Southeast EF2+ Tornado Reports', fontsize=11)

ax4.grid(True, linestyle='--', alpha=0.5)




ax9 = fig.add_axes([0.0, 0., 0.48, 0.1775]) 

p1, = ax9.plot(range(0,len(stp_nino4)),stp_nino4,'r-',linewidth=2.0)
p2, = ax9.plot(range(0,len(stp_nina4)),stp_nina4,'b-',linewidth=2.0)
p3, = ax9.plot(range(0,len(stp_neut4)),stp_neut4,'k-',linewidth=2.0)

p5 = ax9.fill_between(range(0,len(stp_low4)),stp_low4,stp_high4,color='darkgrey',linewidth=1.0,alpha=0.5)

#ax7.set_ylabel('Fraction of Tornadoes (EF1+)', fontsize=10)

ax9.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])

plt.setp(ax9.get_yticklabels(), fontsize=10, rotation=35)
#ax3.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

#ax9.tick_params(axis='y', which='both', bottom='off', top='off', labelleft='off')

ax9.set_title('i) Annual Cycle of CONUS STP', fontsize=11)
    
ax9.set_ylabel('Fraction of STP', fontsize=10)

ax9.grid(True, linestyle='--', alpha=0.5)

ax9.set_xlabel('Day of Year', fontsize=10)
tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax9.set_xticks(tick_locs) 
ax9.set_xticklabels(tick_lbls)




ax5 = fig.add_axes([0.5, 0.8, 0.48, 0.1775]) 

p1, = ax5.plot(range(0,len(torday_nino1)),np.divide(torday_nino1,np.sum(torday_nino1)),'r-',linewidth=2.0)
p2, = ax5.plot(range(0,len(torday_nina1)),np.divide(torday_nina1,np.sum(torday_nina1)),'b-',linewidth=2.0)
p3, = ax5.plot(range(0,len(torday_neut1)),np.divide(torday_neut1,np.sum(torday_neut1)),'k-',linewidth=2.0)    

p5 = ax5.fill_between(range(0,len(torday_low1)),torday_low1,torday_high1,color='darkgrey',linewidth=1.0,alpha=0.5)

#ax1.set_ylabel('Fraction of Tornadoes (EF1+)', fontsize=10)

ax5.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax5.get_yticklabels(), fontsize=10, rotation=35)
#ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax5.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
ax5.tick_params(axis='y', which='both', bottom='off', top='off', labelleft='off')

ax5.set_title('e) Annual Cycle of CONUS EF1+ Tornado Days', fontsize=11)

ax5.grid(True, linestyle='--', alpha=0.5)

legend = ax5.legend([p1,p2,p3,grey_patch],
                    [u"El Ni\xf1o",
                    u"La Ni\xf1a",
                    "Neutral",
                    "IQR"],
                    loc="upper right",
                    fancybox=True, fontsize=10)



ax6 = fig.add_axes([0.5, 0.6, 0.48, 0.1775]) 

p1, = ax6.plot(range(0,len(torday_nino2)),np.divide(torday_nino2,np.sum(torday_nino2)),'r-',linewidth=2.0)
p2, = ax6.plot(range(0,len(torday_nina2)),np.divide(torday_nina2,np.sum(torday_nina2)),'b-',linewidth=2.0)
p3, = ax6.plot(range(0,len(torday_neut2)),np.divide(torday_neut2,np.sum(torday_neut2)),'k-',linewidth=2.0) 

p5 = ax6.fill_between(range(0,len(torday_low2)),torday_low2,torday_high2,color='darkgrey',linewidth=1.0,alpha=0.5)

#ax6.set_ylabel('Fraction of Tornadoes (EF2+)', fontsize=10)

ax6.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax6.get_yticklabels(), fontsize=10, rotation=35)
#ax2.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax6.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
ax6.tick_params(axis='y', which='both', bottom='off', top='off', labelleft='off')

ax6.set_title('f) Annual Cycle of CONUS EF2+ Tornado Days', fontsize=11)

ax6.grid(True, linestyle='--', alpha=0.5)



ax7 = fig.add_axes([0.5, 0.4, 0.48, 0.1775]) 

p1, = ax7.plot(range(0,len(torday_nino3)),np.divide(torday_nino3,np.sum(torday_nino3)),'r-',linewidth=2.0)
p2, = ax7.plot(range(0,len(torday_nina3)),np.divide(torday_nina3,np.sum(torday_nina3)),'b-',linewidth=2.0)
p3, = ax7.plot(range(0,len(torday_neut3)),np.divide(torday_neut3,np.sum(torday_neut3)),'k-',linewidth=2.0) 

p5 = ax7.fill_between(range(0,len(torday_low3)),torday_low3,torday_high3,color='darkgrey',linewidth=1.0,alpha=0.5)

#ax7.set_ylabel('Fraction of Tornadoes (EF1+)', fontsize=10)

ax7.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax7.get_yticklabels(), fontsize=10, rotation=35)
#ax3.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax7.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
ax7.tick_params(axis='y', which='both', bottom='off', top='off', labelleft='off')

ax7.set_title('g) Annual Cycle of Southeast EF1+ Tornado Days', fontsize=11)

ax7.grid(True, linestyle='--', alpha=0.5)



ax8 = fig.add_axes([0.5, 0.2, 0.48, 0.1775]) 

p1, = ax8.plot(range(0,len(torday_nino4)),np.divide(torday_nino4,np.sum(torday_nino4)),'r-',linewidth=2.0)
p2, = ax8.plot(range(0,len(torday_nina4)),np.divide(torday_nina4,np.sum(torday_nina4)),'b-',linewidth=2.0)
p3, = ax8.plot(range(0,len(torday_neut4)),np.divide(torday_neut4,np.sum(torday_neut4)),'k-',linewidth=2.0) 

p5 = ax8.fill_between(range(0,len(torday_low4)),torday_low4,torday_high4,color='darkgrey',linewidth=1.0,alpha=0.5) 

#ax4.set_ylabel('Fraction of Tornadoes (EF2+)', fontsize=10)

ax8.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax8.get_yticklabels(), fontsize=10, rotation=35)
#ax4.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax8.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
ax8.tick_params(axis='y', which='both', bottom='off', top='off', labelleft='off')

ax8.set_title('h) Annual Cycle of Southeast EF2+ Tornado Days', fontsize=11)

ax8.grid(True, linestyle='--', alpha=0.5)





ax10 = fig.add_axes([0.5, 0.0, 0.48, 0.1775]) 

p1, = ax10.plot(range(0,len(stpse_nino4)),stpse_nino4,'r-',linewidth=2.0)
p2, = ax10.plot(range(0,len(stpse_nina4)),stpse_nina4,'b-',linewidth=2.0)
p3, = ax10.plot(range(0,len(stpse_neut4)),stpse_neut4,'k-',linewidth=2.0)

p5 = ax10.fill_between(range(0,len(stpse_low4)),stpse_low4,stpse_high4,color='darkgrey',linewidth=1.0,alpha=0.5)

#ax4.set_ylabel('Fraction of Tornadoes (EF2+)', fontsize=10)

ax10.set_yticks([0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011][1::2])
plt.setp(ax10.get_yticklabels(), fontsize=10, rotation=35)
#ax4.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))

ax10.set_title('j) Annual Cycle of Southeast STP', fontsize=11)

ax10.grid(True, linestyle='--', alpha=0.5)

ax10.tick_params(axis='y', which='both', bottom='off', top='off', labelleft='off')


ax10.set_xlabel('Day of Year', fontsize=10)
tick_locs = [1,50,100,150,200,250,300,350]
tick_lbls = ['Jan 1','Feb 19','Apr 10','May 30','Jul 19','Sep 7','Oct 27','Dec 16']
ax10.set_xticks(tick_locs) 
ax10.set_xticklabels(tick_lbls)


plt.savefig('wut.eps', bbox_inches='tight', dpi=200)





