import matplotlib.pyplot as plt
plt.close("all")

import numpy as np
import meshio
import networkx as nx
import pickle
from scipy.spatial import distance_matrix
import pandas as pandas
import dask.dataframe

# Import module
import sys
sys.path.append('/home/djneuh/software/fault_analysis/fatbox/')

from fatbox.preprocessing import *
from fatbox.edits import *
from fatbox.metrics import *
from fatbox.plots import *
from fatbox.utils import *

from joblib import Parallel, delayed
import multiprocessing
import cv2
import os
import timeit

### Function definitions ####
def calculate_dip(G, dof):
    for node in G:
             
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=dof)
                        
        neighbors = sorted(neighbors.items())
            
        first = neighbors[0][0]
        last = neighbors[-1][0]
             
        x1 = G.nodes[first]['pos'][0]
        y1 = G.nodes[first]['pos'][1]
               
        x2 = G.nodes[last]['pos'][0]
        y2 = G.nodes[last]['pos'][1]
                          
        G.nodes[node]['dip'] = dip(x1, y1, x2, y2)
                
    return G
#############################

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
min_fault_length = float(sys.argv[10])*1e3
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

files = range(startstep,endstep)

def extract_faults(file):
    
    
    #print(file)
    
    ### Load timestep data from csv file. Pandas seems to be fastest when using multiple cores.
    data = pandas.read_csv('../csv/data_' + str(file) + '.csv', delimiter=',')

    # Get positions for fields we need.
    nps_pos = data.columns.get_loc("noninitial_plastic_strain")
    
    # Convert pandas to numpy.
    data = data.to_numpy()
    data = np.flip(data, axis=0)
    
    non_strain = data[:,nps_pos].reshape(y_pixels, x_pixels)
    non_strain = np.flip(non_strain, axis=1)
       
    non_strain_threshold = np.nanmax(non_strain)*strain_rate_factor

    # We use a threshold for when values are below 1, and also take anything above our maximum weakening interval.
    max_weak_interval = 1
    threshold = np.where(np.logical_or(non_strain >= non_strain_threshold, non_strain >= max_weak_interval), 1, 0).astype(np.uint8)
    #threshold = np.where(non_strain >= non_strain_threshold, 1, 0).astype(np.uint8) 
    
    ##########################################################################
    
    # SKELETONIZE - reduce the width of faults to a single pixel.
    skeleton = skeleton_guo_hall(threshold)
    #plot_comparison([data, threshold, skeleton])
    
    #// How does this work?
    ret, markers = cv2.connectedComponents(skeleton)
    
    # The G graph holds our positive polarity info
    G = nx.Graph()
    
    # First, we add the points as nodes and set their information.
    node = 0
    for comp in range(1,ret):   
        
        # Why do we transpose
        points = np.transpose(np.vstack((np.where(markers==comp))))    
        
        for point in points:
            # Add a node to correspond to the point.
            G.add_node(node)
            # Set the position data.
            G.nodes[node]['pos'] = (point[1], point[0])
            G.nodes[node]['x'] = G.nodes[node]['pos'][0]
            G.nodes[node]['y'] = G.nodes[node]['pos'][1]
            G.nodes[node]['z'] = 0
            # Set the fault label.
            G.nodes[node]['component'] = comp

            # The next point is set as the next node.
            node += 1
        
    
    # Next we add the edges between these points.
    for comp in range(1,ret): 
        
        # Add point data for nodes that are connected (fault label).
        points = [G.nodes[node]['pos'] for node in G if G.nodes[node]['component']==comp]
        # Add corresponding node numbers.
        nodes  = [node for node in G if G.nodes[node]['component']==comp]
    
        # Find the distance between points to see if they are the same fault.
        dm = distance_matrix(points, points)  
        
        for n in range(len(points)):
            for m in range(len(points)):
                # Determine which nodes are actually connected.
                if dm[n,m]<minimum_distance:
                    G.add_edge(nodes[n],nodes[m])
    
  
    G = compute_edge_length(G)
    
    G = remove_self_edge(G)
    
    G = label_components(G)
    
    G = remove_triangles(G)
    
    G = split_triple_junctions(G, 8, split='minimum', threshold=20, plot=False)
    
    # Remove any faults below a certain size.
    rsc = round(min_fault_length/scale)    # Convert length to pixels.
    G = remove_small_components(G, minimum_size = rsc)
    
    G = label_components(G)
    
    G = calculate_dip(G, 5)
        
    # Threshold when removing.
    dof = 5
    for node in G:
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=dof)
        dips = [G.nodes[node]['dip'] for node in neighbors.keys()]
        G.nodes[node]['max_diff'] = np.max(np.diff(dips))
        G.nodes[node]['cut'] = False  
    
    
    removals = []
    for node in G:
        if G.nodes[node]['max_diff'] > 60:
            removals.append(node)
    
    
    G.remove_nodes_from(removals)
    
    # Remove any faults below a certain size.
    rsc = round(min_fault_length/scale)    # Convert length to pixels.
    G = remove_small_components(G, minimum_size = rsc)
    
    G = label_components(G)    
    
    
    ##########################################################################
    ###  Plot
    # Determine whether to extend our lower graph.
    Gmaxx = x_pixels
    Gminx = 0
    Gmaxy = y_pixels
    Gminy = 0
    
    xsc = 10e3/scale
    ysc = 10e3/scale
    Mxdiff = math.ceil( (np.nanmax(np.array([G.nodes[node]['pos'][0] for node in G])) - x_pixels/2) /xsc) + 0.1
    Mixdiff =math.ceil( (x_pixels/2 - np.nanmin(np.array([G.nodes[node]['pos'][0] for node in G]))) /xsc) + 0.1
    Mydiff = math.floor( (y_pixels - np.nanmax(np.array([G.nodes[node]['pos'][1] for node in G]))) /ysc) - 0.1
    
    Gmaxx = int(x_pixels/2 + Mxdiff*xsc)
    Gminx = int(x_pixels/2 - Mixdiff*xsc)
    Gmaxy = int(y_pixels - Mydiff*ysc)
    
    title_time = round(vistimes[file]/1e6,3)
    fig, axs = plt.subplots(2, 1, figsize=(16,8))
    
    non_strain = non_strain[0:Gmaxy, Gminx:Gmaxx]  # Change the size of our non_strain matrix.
    p = axs[0].matshow(non_strain, cmap='gray_r', extent=[Gminx,Gmaxx,Gmaxy,Gminy],aspect="equal") 
    # Color bar locator
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.15)   
    cb0 = fig.colorbar(p, ax=axs[0], cax=cax)

    node_scale = 0.75*(x_pixels/(Gmaxx-Gminx))
    p = axs[1].matshow(non_strain, cmap='gray_r', extent=[Gminx,Gmaxx,Gmaxy,Gminy],aspect="equal")  
    
    # Plot the extracted faults.
    plot_components(G, axs[1], label=False, node_size = node_scale)
    axs[1].set_xlim([Gminx, Gmaxx])
    axs[1].set_ylim([Gmaxy, 0])
    axs[0].set_title('Non-initial plastic strain, Time: ' + str(title_time) + ' Myr, File:' +str(file), fontweight='bold')
    axs[1].set_title('Non-initial plastic strain with faults ', fontweight='bold')
    axs[0].set_ylabel('Depth (km)')
    axs[1].set_ylabel('Depth (km)')
    axs[1].set_xlabel('Distance from model center (km)')
    
    # Color bar locator
    divider = make_axes_locatable(axs[1])
    #cax = divider.append_axes("right", size="3%", pad=0.15)    
    #cb0 = fig.colorbar(p, ax=axs[1], cax=cax)
    
    maxy = Gmaxy*scale/1000
    maxx = Gmaxx*scale/1000
    minx = Gminx*scale/1000
    
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
    else:
        yint = 10

        
    xlab = np.array(np.zeros(math.floor((maxx-minx)/xint)))
    xlab[0] = math.ceil(minx/xint)*xint 
    for i in range(1, len(xlab)):
        xlab[i] = xlab[i-1] + xint
            
    ylab = np.array(np.zeros(math.floor((maxy+10)/yint)))
    ylab[0] = -10 
    for i in range(2, len(ylab)):
        ylab[i] = ylab[i-1] + yint

    
    axs[0].set_xticks(xlab*1000/scale)    
    axs[1].set_xticks(xlab*1000/scale)
    xlab = xlab - (xlength/2)/1e3
    axs[0].set_xticklabels(xlab.astype(int))
    axs[1].set_xticklabels(xlab.astype(int))
    
    axs[0].set_yticks((ylab+10)*1000/scale)
    axs[0].set_yticklabels(ylab)
    axs[1].set_yticks((ylab+10)*1000/scale)
    axs[1].set_yticklabels(ylab)
          
    # Save figure as image to check that faults are being properly identified.
    plt.savefig(fname='./images/extracted/image_' + str(file).zfill(5) + '.png', dpi=200)
    plt.close("all")
    
    
    # Save the pickle graph, which can be reloaded and contain all information
    # corresponding to the nodes and edges.
    # This data can later be viewed, after loading a graph using the command
    # name.nodes
    # Where name is the variable given to the loaded graph. This will display
    # all nodes associated with the graph. Then by asking for a number:
    # name.nodes[5]
    # you can see all information corresponding to that node.
    pickle.dump(G, open('./graphs/pickle/g_' + str(file) + '.p', "wb" ))
        
        
        
        
#%%


    
       
#num_cores = multiprocessing.cpu_count()-4
results = Parallel(n_jobs=min(len(files), num_proc))(delayed(extract_faults)(file) for file in files)     
       
#multiprocessing.cpu_count()
stop = timeit.default_timer()

print('Extract time: ', stop - start) 
        
      
       




