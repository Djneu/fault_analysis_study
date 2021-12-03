import pickle
import matplotlib.pyplot as plt
plt.close("all")


import math
import numpy as np
import meshio
import networkx as nx
import pickle
import pandas as pandas
import dask.dataframe
import timeit
#from joblib import Parallel, delayed

# Import module
import sys
sys.path.append('/home/djneuh/software/fault_analysis/fatbox/')

from fatbox.edits import *
from fatbox.metrics import *
from fatbox.plots import *
from fatbox.utils import *
import os

### Function definitions ####
def get_fault_labels(G):
    labels=set()
    for node in G:
        labels.add(G.nodes[node]['fault'])
    return sorted(list(labels))

def get_fault(G, n):
    nodes = [node for node in G if G.nodes[node]['fault']==n]
    return G.subgraph(nodes)

def get_polarity(G):
    for node in G:
        polarity = G.nodes[node]['polarity']
        break
    return polarity

def get_fault_polarities(G):
    labels = get_fault_labels(G)
    polarities=[]
    for label in labels:            
        G_fault = get_fault(G, label)
        polarities.append(get_polarity(G_fault))
    return polarities

def G_to_pts(G):
    labels = get_fault_labels(G)
    point_set=[]
    for label in labels:            
        G_fault = get_fault(G, label)
        points = []
        for node in G_fault:
            points.append(G_fault.nodes[node]['pos'])
        point_set.append(points)
    return point_set

def is_A_in_B(set_A, set_B, R):    
      distances = np.zeros((len(set_A), len(set_B)))
      for n, pt_0 in enumerate(set_A):
          for m, pt_1 in enumerate(set_B):
              distances[n,m] = math.sqrt((pt_0[0]-pt_1[0])**2 + (pt_0[1]-pt_1[1])**2)
      if np.mean(np.min(distances, axis=1)) > R:
          return False
      else:
          return True
      
#def connect_faults(n): 
#    temp_connect = set()
#    for m in range(len(faults_1)):
         # We only consider them correlated if they have the same polarity.
#         if polarities_0[n] == polarities_1[m]:
#             if is_A_in_B(pt_set_0[n], pt_set_1[m], R):
#                 temp_connect.add((faults_0[n], faults_1[m]))
#             if is_A_in_B(pt_set_1[m], pt_set_0[n], R):
#                 temp_connect.add((faults_0[n], faults_1[m]))  
    
#    return temp_connect
###################################

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
scale = xlength/x_pixels

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
#if endstep > len(os.listdir('images/extracted')):
#    endstep = len(os.listdir('images/extracted')) + startstep

# We do endtime-1 as the final timestep won't have a next timestep to correlate to.
times = range(startstep,endstep-1)
max_comp = 0
H = nx.Graph()


for count, time in enumerate(times):
        
    ### Load needed data.
    #print(time)
    
    # Load the current timestep graph and next timesteps graph to compare.
    G_0 = pickle.load(open('./graphs/pickle/g_' + str(time) + '.p', 'rb'))          
    G_1 = pickle.load(open('./graphs/pickle/g_' + str(time+1) + '.p', 'rb'))    
                            
    ##########################################################################
    ### Get fault information
    
    # If it is the first timestep, we want to save the fault labels as there is
    # nothing to correlate to.
    if count == 0:
        for node in G_0:
            G_0.nodes[node]['fault'] = G_0.nodes[node]['component']            
        pickle.dump(G_0, open('./graphs/pickle/g_' + str(time) + '.p', "wb" )) 
        
    # Next we set initial fault labels to the next time step.
    for node in G_1:
        G_1.nodes[node]['fault'] = G_1.nodes[node]['component']  
        
    # Get fault labels for each graph.
    faults_0 = get_fault_labels(G_0)
    faults_1 = get_fault_labels(G_1)
    
    # Calculate the polarities
    G_0 = calculate_polarity(G_0)
    G_1 = calculate_polarity(G_1)
    
    # Get polarities for each graph.
    polarities_0 = get_fault_polarities(G_0)
    polarities_1 = get_fault_polarities(G_1)
    
    # Find the corresponding points for the nodes.
    pt_set_0 = G_to_pts(G_0)
    pt_set_1 = G_to_pts(G_1)
          
    # Set a radius where faults are correlated and then determine
    # whether or not the two timestep faults are connected.                   
    connections = set()
    for n in range(len(faults_0)):
        for m in range(len(faults_1)):
            # We only consider them correlated if they have the same polarity.
            if polarities_0[n] == polarities_1[m]:
                if is_A_in_B(pt_set_0[n], pt_set_1[m], R):
                    connections.add((faults_0[n], faults_1[m]))
                if is_A_in_B(pt_set_1[m], pt_set_0[n], R):
                    connections.add((faults_0[n], faults_1[m]))  
                    
    #results = (Parallel(n_jobs=min(len(faults_0), num_proc))(delayed(connect_faults)(n) for n in range(len(faults_0)))) 
    #for i in range(len(results)):
    #    connections.update(results[i])   
    
    
       
    ###### Relabel ################
    
    # Start by assuming they are not correlated.
    for node in G_1:
        G_1.nodes[node]['matched']=False
    

    # Sort by length
    lengths = [total_length(get_fault(G_0, connection[0])) for connection in connections]
    
    # If there are any connections, then sort them
    if len(connections) > 0:
        lengths, connections = zip(*sorted(zip(lengths, connections)))
    
    
    # Go through connections and label the connected faults as the same.
    for node in G_1:
        for connection in connections:
            if G_1.nodes[node]['component'] == connection[1]:
                G_1.nodes[node]['fault'] = connection[0]
                G_1.nodes[node]['matched'] = True
        
    
    
    # Relabel unmatched components, only if G_1 has faults in it.
    if len(get_fault_labels(G_1)) > 0:
        max_comp = max(max_comp, max(get_fault_labels(G_1)))
    
        G_1_sub = nx.subgraph(G_1, [node for node in G_1 if G_1.nodes[node]['matched']==False])  
        for label, cc in enumerate(sorted(nx.connected_components(G_1_sub))): 
            for n in cc:
                G_1.nodes[n]['fault'] = label+max_comp+1
    
        max_comp = max(max_comp, max(get_fault_labels(G_1)))
    
############### Plot  ##########################    
    #if time == startstep:
    #    Gmaxx = np.nanmax(np.array([G_0.nodes[node]['pos'][0] for node in G_0])) + 60
    #    Gmaxy = np.nanmax(np.array([G_0.nodes[node]['pos'][1] for node in G_0])) + 60
    #    Gminx = np.nanmin(np.array([G_0.nodes[node]['pos'][0] for node in G_0])) - 60   
    #elif len(G_0) > 0:      
    #    if np.nanmax(np.array([G_0.nodes[node]['pos'][0] for node in G_0])) > Gmaxx - 10:
    #        Gmaxx = Gmaxx + 60
    #    if np.nanmax(np.array([G_0.nodes[node]['pos'][1] for node in G_0])) > Gmaxy - 10:
    #        Gmaxy = Gmaxy + 60
    #    if np.nanmin(np.array([G_0.nodes[node]['pos'][0] for node in G_0])) < Gminx + 10:
    #        Gminx = Gminx - 60
    
    Gmaxx = x_pixels
    Gminx = 0
    Gmaxy = y_pixels
    Gminy = 0
    
    xsc = 10e3/scale
    ysc = 10e3/scale
    Mxdiff = math.ceil( (np.nanmax(np.array([G_0.nodes[node]['pos'][0] for node in G_0])) - x_pixels/2) /xsc) + 0.1
    Mixdiff =math.ceil( (x_pixels/2 - np.nanmin(np.array([G_0.nodes[node]['pos'][0] for node in G_0]))) /xsc) + 0.1
    Mydiff = math.floor( (y_pixels - np.nanmax(np.array([G_0.nodes[node]['pos'][1] for node in G_0]))) /ysc) - 0.1
    
    Gmaxx = int(x_pixels/2 + Mxdiff*xsc)
    Gminx = int(x_pixels/2 - Mixdiff*xsc)
    Gmaxy = int(y_pixels - Mydiff*ysc)
    
    title_time = round(vistimes[time]/1e6,3)

    
    fig, axs = plt.subplots(1, 1, figsize=(11,8))
    
    lscale = xlength/x_pixels
    node_scale = 0.75*(x_pixels/(Gmaxx-Gminx))
    plot_faults(G_0, axs, node_size = node_scale, label=False)
    axs.set_xlim([Gminx, Gmaxx])
    axs.set_ylim([Gmaxy, 0])
    axs.set_ylabel('Depth (km)')
    axs.set_xlabel('Distance from model center (km)')
    axs.set_title('Correlated plot, Time: ' + str(title_time) + ' Myr, File:' +str(time), fontweight='bold')
    axs.axis('equal')
    
    
    maxy = Gmaxy*lscale/1000
    maxx = Gmaxx*lscale/1000
    minx = Gminx*lscale/1000
    
    xint = 5
    if (maxx - minx) < 40:
        xint = 5
    elif (maxx - minx) < 80:
        xint = 10
    elif (maxx - minx) < 155:
        xint = 25
    else:
        xint = 50
    
    yint = 5
    if (maxy + 10) < 35:
        yint = 5
    elif (maxy + 10) < 80:
        yint = 10

        
    xlab = np.array(np.zeros(math.floor((maxx-minx)/xint)))
    xlab[0] = math.ceil(minx/xint)*xint 
    for i in range(1, len(xlab)):
        xlab[i] = xlab[i-1] + xint
            
    ylab = np.array(np.zeros(math.floor((maxy+10)/yint)))
    ylab[0] = -10 
    for i in range(2, len(ylab)):
        ylab[i] = ylab[i-1] + yint

    
    axs.set_xticks(xlab*1000/lscale)
    xlab = xlab - (xlength/2)/1e3
    axs.set_xticklabels(xlab.astype(int))
    
    axs.set_yticks((ylab+10)*1000/lscale)
    axs.set_yticklabels(ylab)
    
    divider = make_axes_locatable(axs)
    
    
    
    # Save image to check correlation worked.
    plt.savefig('./images/correlated/img' + str(time).zfill(5) + '.png', dpi=200)
    plt.close('all') 
    
    # Save pickle graph with updated information.
    pickle.dump(G_1, open('./graphs/pickle/g_' + str(time+1) + '.p', "wb" )) 
    
#multiprocessing.cpu_count()
stop = timeit.default_timer()


print('Correlation time: ', stop - start) 
