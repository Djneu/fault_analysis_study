import matplotlib.pyplot as plt
plt.close("all")

import numpy as np
import meshio
import networkx as nx
import pickle
import timeit
import os

# Import module
import sys
sys.path.append('/home/djneuh/software/fault_analysis/fatbox/')

from fatbox.edits import *
from fatbox.metrics import *
from fatbox.plots import *
from fatbox.utils import *

start = timeit.default_timer()

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

# We do endtime-1 as the final timestep won't have a next timestep to correlate to.
for file in range(startstep,endstep-1):
    
    
    #print(file)
    
    # Load the two graphs.
    G_0 = pickle.load(open('./graphs/pickle/g_' + str(file) + '.p', 'rb'))        
    G_1 = pickle.load(open('./graphs/pickle/g_' + str(file+1) + '.p', 'rb'))

    # For the second graph, write the slips to heave, throw, and displacement.
    G_1 = write_slip_to_displacement(G_1, dim=2)

    # If it's the first timestep, write G_0 as there is nothing to compare to.
    if file == startstep:
        G_0 = write_slip_to_displacement(G_0, dim=2)
        pickle.dump(G_0, open('./graphs/pickle/g_' + str(file) + '.p', "wb" )) 
    
    def common_faults(G, H):
        C_G = get_fault_labels(G)
        C_H = get_fault_labels(H)
        return list(set(C_G) & set(C_H))
    
    # Get the faults the two graphs have in common.
    cf = common_faults(G_0, G_1)    
        

    for fault in cf:
        
        # Write the heave, throw, displacement, etc. into points
        # for each fault.
        points_0 = get_displacement(get_fault(G_0, fault), dim=2)
        points_1 = get_displacement(get_fault(G_1, fault), dim=2)
    
        # Add the current displacement to the nearest node of the next timestep.
        for n in range(points_1.shape[0]):    
            index = closest_node(points_1[n,1:3], points_0[:,1:3]) 
            
            # Add previous heave
            points_1[n,3] += points_0[index][3]
            # Add previous throw
            points_1[n,4] += points_0[index][4]
            # Add previous displacement
            points_1[n,5] += points_0[index][5]
        
        # Assign updated displacements.
        G_1 = assign_displacement(G_1, points_1, dim=2)


    #fig, ax = plt.subplots(figsize=(16,4))
    #ax.imshow(np.zeros((y_pixels, x_pixels)), 'gray_r')
    
    # Only plot the attribute if it exists.
    #if len(G_1) > 0:
    #  plot_attribute(G_1, 'displacement', ax=ax, node_size= 1)
    

    
    #plt.savefig('./images/displacement/image_' + str(file+1).zfill(5) + '.png', dpi=200)
    #plt.close("all")   




    pickle.dump(G_1, open('./graphs/pickle/g_' + str(file+1) + '.p', "wb" ))



#multiprocessing.cpu_count()
stop = timeit.default_timer()

print('Displacement time: ', stop - start) 



