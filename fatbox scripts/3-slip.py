import matplotlib.pyplot as plt
plt.close("all")

import numpy as np
import meshio
import networkx as nx
import pickle
import pandas as pandas
import dask.dataframe
import timeit
import os

# Import module
# Import module
import sys
sys.path.append('/home/djneuh/software/fault_analysis/fatbox/')

from fatbox.edits import *
from fatbox.metrics import *
from fatbox.plots import *
from fatbox.utils import *

from joblib import Parallel, delayed
import multiprocessing

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
#if endstep > len(os.listdir('images/correlated')):
#    endstep = len(os.listdir('images/correlated')) + startstep

files = range(startstep, endstep)


def instant_displacement(file):

    #print(file)   
    
    ### Load timestep data from csv file. Pandas seems to be fastest when using multiple cores.
    data = pandas.read_csv('../csv/data_' + str(file) + '.csv', delimiter=',')

    # Get positions for fields we need.
    vx_pos = data.columns.get_loc("velocity:0")
    vy_pos = data.columns.get_loc("velocity:1") 
    nps_pos = data.columns.get_loc("noninitial_plastic_strain")
    
    # Convert pandas to numpy.
    data = data.to_numpy()
    
    data = np.flip(data, axis=0)
        
    v_x            = data[:,vx_pos].reshape(y_pixels, x_pixels)
    v_z            = data[:,vy_pos].reshape(y_pixels, x_pixels)
    non_strain     = data[:,nps_pos].reshape(y_pixels, x_pixels)
    
    v_x = np.flip(v_x, axis=1)
    v_z = np.flip(v_z, axis=1)
    non_strain = np.flip(non_strain, axis=1)
    
    # Load the graph.
    G = pickle.load(open('./graphs/pickle/g_' + str(file) + '.p', 'rb'))
    G = nx.Graph(G)
    
    # Calculate slip direction, giving a dx and dy associated with the vector.
    G = calculate_direction(G, 3)
    
    #// Create pickup points perpendicular from the node based on dx and dy.
    H = generate_pickup_points(G, 1)
    
    #// Determine whether pickup points are within reason?
    H = extract_attribute(H, v_x, 'v_x')
    H = extract_attribute(H, v_z, 'v_z')
    
    #// Why do they need filtering?
    H = filter_pickup_points(G, H)
    
    dt = vistimes[file+1]-vistimes[file]
    G = calculate_slip_rate(G, H, dim=2)
    G = calculate_slip(G, H, dim=2, dt=dt)
    
    ############ Plots ####################   
    # Determine whether to extend our lower graph.
    #if len(G) > 0:
    #    Gmaxx = np.nanmax(np.array([G.nodes[node]['pos'][0] for node in G])) + 75
    #    Gmaxy = np.nanmax(np.array([G.nodes[node]['pos'][1] for node in G])) + 75 
    #    Gminx = np.nanmin(np.array([G.nodes[node]['pos'][0] for node in G])) - 75
    #    Gminy = 0
    #else:
    #    Gmaxx = x_pixels
    #    Gminx = 0
    #    Gmaxy = y_pixels
    #    Gminy = 0
        
    Gmaxx = x_pixels
    Gminx = 0
    Gmaxy = y_pixels
    Gminy = 0
    
    xsc = 10e3/scale
    ysc = 10e3/scale
    Mxdiff = math.ceil( (np.nanmax(np.array([G.nodes[node]['pos'][0] for node in G])) - x_pixels/2) /xsc) + 0.1
    Mixdiff =math.ceil( (x_pixels/2 - np.nanmin(np.array([G.nodes[node]['pos'][0] for node in G]))) /xsc) + 0.1
    Mydiff = math.floor( (y_pixels - np.nanmax(np.array([G.nodes[node]['pos'][1] for node in G]))) /ysc) + 0.1
    
    Gmaxx = int(x_pixels/2 + Mxdiff*xsc)
    Gminx = int(x_pixels/2 - Mixdiff*xsc)
    Gmaxy = int(y_pixels - Mydiff*ysc)
    
    xsc = 10e3/scale
    ysc = 10e3/scale
    Mxdiff = math.ceil( (np.nanmax(np.array([G.nodes[node]['pos'][0] for node in G])) - x_pixels/2) /xsc) + 0.1
    Mixdiff =math.ceil( (x_pixels/2 - np.nanmin(np.array([G.nodes[node]['pos'][0] for node in G]))) /xsc) + 0.1
    Mydiff = math.floor( (y_pixels - np.nanmax(np.array([G.nodes[node]['pos'][1] for node in G]))) /ysc) - 0.1
    
    Gmaxx = int(x_pixels/2 + Mxdiff*xsc)
    Gminx = int(x_pixels/2 - Mixdiff*xsc)
    Gmaxy = int(y_pixels - Mydiff*ysc)

    v_x = v_x[0:Gmaxy, Gminx:Gmaxx]  
           
    title_time = round(vistimes[file]/1e6,3)
    fig, axs = plt.subplots(1, 1, figsize=(12,8))
    
    non_strain = non_strain[0:Gmaxy, Gminx:Gmaxx]  # Change the size of our non_strain matrix.
    p = axs.matshow(non_strain, cmap='gray_r', extent=[Gminx,Gmaxx,Gmaxy,0],aspect="equal")  
    axs.set_xlim([Gminx, Gmaxx])
    axs.set_ylim([Gmaxy, 0])
    axs.set_ylabel('Depth (km)')
    axs.set_xlabel('Distance from model center (km)')
    axs.set_title('Slip rate plot, Time: ' + str(title_time) + ' Myr, File:' +str(file), fontweight='bold')
    
    
    # Only plot the attribute if it exists.
    node_scale = (x_pixels/(Gmaxx-Gminx))
    if len(G) > 0:
        plot_attribute(G, 'slip_rate', ax=axs, node_size=node_scale,vmin=0,vmax=0.01)
        
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

    
    axs.set_xticks(xlab*1000/scale)
    xlab = xlab - (xlength/2)/1e3
    axs.set_xticklabels(xlab.astype(int))
    
    axs.set_yticks((ylab+10)*1000/scale)
    axs.set_yticklabels(ylab)
    plt.tight_layout()
    
    plt.savefig('./images/slip/image_' + str(file).zfill(5) + '.png', dpi=200)
    plt.close("all")       
    
       
    pickle.dump(G, open('./graphs/pickle/g_' + str(file) + '.p', "wb" ))






#num_cores = multiprocessing.cpu_count()-4
       
results = Parallel(n_jobs=min(len(files), num_proc))(delayed(instant_displacement)(file) for file in files)     
       

#multiprocessing.cpu_count()
stop = timeit.default_timer()

print('Slip time: ', stop - start) 


