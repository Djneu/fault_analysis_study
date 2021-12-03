#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:47:40 2021

@author: djneuh
"""

# Save fault ages.

import numpy as np 
import networkx as nx
import pickle
import timeit
import os

import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

from sys import stdout
# Import module
import sys
sys.path.append('/home/djneuh/software/fault_analysis/fatbox/')

from fatbox.edits import *
from fatbox.metrics import *
from fatbox.plots import *
from fatbox.utils import *

tstart = timeit.default_timer()

vistimes = get_times('../../statistics')

# User set variables.
endtime = float(sys.argv[1])*1e6
starttime = float(sys.argv[2])*1e6
x_pixels = int(sys.argv[3])
y_pixels = int(sys.argv[4])
xlength = float(sys.argv[5])*1e3
strain_rate_factor = float(sys.argv[6])
minimum_distance = float(sys.argv[7])
R = int(sys.argv[8])
num_proc = int(sys.argv[9])

# Find the start and end points.
endstep = 0
if endtime == 0:
    endstep= len(vistimes) - 1
else:
    while vistimes[endstep] < endtime:
        endstep = endstep + 1
    
startstep = 0
while vistimes[startstep] < starttime:
    startstep = startstep + 1
    
#Sometimes correlation ends early and we don't have values till the end.
#if endstep > len(os.listdir('images/correlated')):
#    endstep = len(os.listdir('images/correlated')) + startstep
  
# We want to make sure the labels never exceed a.
a = [0]*50000 
fault_slip = [0]*50000
active_limit = 1e-4 # 0.1 mm/yr
for time in range(startstep,endstep-1):
    #print(time)
    G_0 = pickle.load(open('./graphs/pickle/g_' + str(time) + '.p', 'rb'))
    G = pickle.load(open('./graphs/pickle/g_' + str(time+1) + '.p', 'rb'))
    
    labels_0           = get_fault_labels(G_0)
    labels           = get_fault_labels(G)
    
    # Assume fault was active entire previous timestep.
    dt = vistimes[time+1]-vistimes[time]
    for label in labels_0:
        fault = get_fault(G_0, label)
        fault_slip[label] = compute_node_values(fault, 'slip_rate','max');
        
        if fault_slip[label] >= active_limit and label in labels:
            a[label] = a[label]+dt
        
                
    for node in G.nodes:
        lab = G.nodes[node]['fault']
        G.nodes[node]['fault_age'] = a[lab]
        
        if fault_slip[lab] >= active_limit:                          
            G.nodes[node]['fault_active'] = 1
        else:
            G.nodes[node]['fault_active'] = 0
        
    pickle.dump(G, open('./graphs/pickle/g_' + str(time) + '.p', "wb" ))
             

#multiprocessing.cpu_count()
stop = timeit.default_timer()

print('Age time: ', stop - tstart) 
