#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 12:20:33 2021

@author: djneuh
"""

#Polarity graphs.

import numpy as np 
import networkx as nx
import pickle
import collections
import timeit
import os

import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.signal import medfilt

from sys import stdout
# Import module
import sys
sys.path.append('/home/djneuh/software/fault_analysis/fatbox/')

from fatbox.edits import *
from fatbox.metrics import *
from fatbox.plots import *
from fatbox.utils import *
import seaborn as sns



#matplotlib inline
from ipywidgets import interactive, widgets, Layout
plt.close("all")

def get_fault_polarities(G):
    labels = get_fault_labels(G)
    polarities=[]
    for label in labels:            
           G_fault = get_fault(G, label)
           polarities.append(get_polarity(G_fault))
    return polarities

 
def get_polarity(G):
    for node in G:
        polarity = G.nodes[node]['polarity']
        break
    return polarity

def plot_stage(p, alpha=0.15):
    c1 = np.array(255/255, 210/255, 193/255)
    ax[p].fill_between(sx1,yy, color=c1, alpha=alpha, linewidth=0)
    ax[p].fill_between(sx2,yy, alpha=alpha, linewidth=0)
    ax[p].fill_between(sx3,yy, alpha=alpha, linewidth=0)
    ax[p].fill_between(sx4,yy, alpha=alpha, linewidth=0)
    ax[p].fill_between(sx5,yy, alpha=alpha, linewidth=0)  

def get_edge_attribute(H, attribute):
    colors = np.zeros(len(H.edges))
    for n, edge in enumerate(H.edges):
        colors[n] = 0.5*H.nodes[edge[0]][attribute]+0.5*H.nodes[edge[1]][attribute]
    return colors

def get_p_colors(G, attribute, color_l, color_r):
    "Colo based solely off 2 polarity values"

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    
    #cl = sns.color_palette() #palette="muted", n_colors=nums
#colors = ([255/255,  163/255,  23/255,   0.85], [35/255,  119/255,  213/255,  0.85])
   # colors = (cl[3], cl[0])
    
    #color_l = [255/255,  177/255,  57/255]
    #color_r =  [67/255,  139/255,  219/255]
   # color_l = 
    edge_color = np.zeros((len(G.edges), 3))


    for n, edge in enumerate(G.edges):
        if(G.edges[edge][attribute] == 1):    
            edge_color[n, 0] = color_r[0]
            edge_color[n, 1] = color_r[1]
            edge_color[n, 2] = color_r[2]
            
        if(G.edges[edge][attribute] == -1):     
            edge_color[n, 0] = color_l[0]
            edge_color[n, 1] = color_l[1]
            edge_color[n, 2] = color_l[2]

    return edge_color

tstart = timeit.default_timer()
vistimes = get_times('../../../statistics')

# Load all the pickle graphs.
Gs = []

# User set variables.
endtime = 0 #float(sys.argv[1])*1e6
starttime = 0.015*1e6 #float(sys.argv[2])*1e6
graph_start = 0 #int(sys.argv[3])
graph_end = 30 #int(sys.argv[4])
xlength = 450e3 #float(sys.argv[5])*1e3
x_pixels = 2880 #int(sys.argv[6])

# Y max values for graph.
g0 = 50 #int(sys.argv[7])
g0max = 400 #int(sys.argv[12])
g1 = 20 #int(sys.argv[8])
g2 = 60 #int(sys.argv[9])
g3 = 100 #int(sys.argv[10])
g4 = 400 #int(sys.argv[11])
scale = (xlength/x_pixels)/1000

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
if endstep > len(os.listdir('../images/correlated')):
    endstep = len(os.listdir('../images/correlated')) + startstep - 1
    
    
inc = 10
Gs = np.arange(startstep+1,endstep-inc-2,inc)
#Gs = np.arange(4000,4001,1)
#times = times*1e5/1e6

# Number of faults plot.
l_number = np.zeros(len(Gs))
r_number = np.zeros(len(Gs))
tl_number = np.zeros(len(Gs))
tr_number = np.zeros(len(Gs))

# Fault lengths, averaged and total
l_length = np.zeros(len(Gs))
r_length = np.zeros(len(Gs))
tl_length = np.zeros(len(Gs))
tr_length = np.zeros(len(Gs))

# Fault slips
l_slip = np.zeros(len(Gs))
r_slip = np.zeros(len(Gs))
tl_slip = np.zeros(len(Gs))
tr_slip = np.zeros(len(Gs))


# Fault displacement
l_displacement = np.zeros(len(Gs))
r_displacement = np.zeros(len(Gs))
tl_displacement = np.zeros(len(Gs))
tr_displacement = np.zeros(len(Gs))

# Fault ages, averaged
l_age = np.zeros(len(Gs))
r_age = np.zeros(len(Gs))
tl_age = np.zeros(len(Gs))
tr_age = np.zeros(len(Gs))

times = np.zeros(len(Gs))

#############################################################################
### Loop to assign all values
#for time in range(len(Gs)):    
for time, file in enumerate(Gs):
    #file = 4000
    #print(file)
    # Load the graph for the current time.
    G_0 = pickle.load(open('./pickle/g_' + str(file-1) + '.p', 'rb')) 
    G = pickle.load(open('./pickle/g_' + str(file) + '.p', 'rb'))  
    G_1 = pickle.load(open('./pickle/g_' + str(file+1) + '.p', 'rb')) 
    
    #print(time, file)
    
    # Get the fault labels and polarities
    labels           = get_fault_labels(G)
    labels0           = get_fault_labels(G_0)
    labels1           = get_fault_labels(G_0)
    polarity         = get_fault_polarities(G)
            
        
    # Variables to hold values for this timestep
    displacements = np.zeros(len(labels))
    slips = np.zeros(len(labels))
    slips_x = np.zeros(len(labels))
    lengths = np.zeros(len(labels))
    ages = np.zeros(len(labels))
    active = np.zeros(len(labels))
    min_exist = np.zeros(len(labels))
    
    tdisplacements = np.zeros(len(labels))
    tslips = np.zeros(len(labels))
    tslips_x = np.zeros(len(labels))
    tlengths = np.zeros(len(labels))
    tnumber = np.zeros(len(labels))
    tages = np.zeros(len(labels))
    
    #for i, label in enumerate(labels):

    
    # Fill each fault with its value
    number = 0
    for n, label in enumerate(labels):
        
      for j, label0 in enumerate(labels0):
          if label == label0:
              min_exist[n] = 1
              

      fault = get_fault(G, label)
      if min_exist[n] == 1:
          for nn, node in enumerate(fault.nodes):
              if nn == 0:
                  if compute_node_values(fault, 'slip_rate','max') >= 1e-4:
                      displacements[n] = compute_node_values(fault, 'displacement','mean')
                      slips[n] = compute_node_values(fault, 'slip_rate','mean')
                      lengths[n] = total_length(fault)
                      ages[n] = compute_node_values(fault, 'fault_age','max')
                      slips_x[n] = compute_node_values(fault, 'slip_rate_x','max')
                      active[n] = 1
                  else:
                          displacements[n] = 0
                          slips[n] = 0
                          ages[n] = 0
                          slips_x[n] = 0
                          lengths[n] = 0
                          active[n] = 0
                   
                  tdisplacements[n] = compute_node_values(fault, 'displacement','mean')
                  tslips[n] = compute_node_values(fault, 'slip_rate','mean')
                  tlengths[n] = total_length(fault)
                  tslips_x[n] = compute_node_values(fault, 'slip_rate_x','max')
                  tages[n] = compute_node_values(fault, 'fault_age','max')
                  tnumber[n] = 1
                  
              else:
                break
                
    # Assign by polarity, where 0 are faults dipping to the right.
    for i in range(0,len(polarity)):
        if min_exist[i] == 1:
           if polarity[i] == 1:
               r_number[time] = r_number[time] + active[i]
               r_displacement[time] = r_displacement[time] + displacements[i]
               r_slip[time] = r_slip[time] + slips[i]
               r_length[time] = r_length[time] + lengths[i]
               r_age[time] = r_age[time] + ages[i]
            
               tr_number[time] = tr_number[time] + 1
               tr_displacement[time] = tr_displacement[time] + tdisplacements[i]
               tr_slip[time] = tr_slip[time] + tslips[i]
               tr_length[time] = tr_length[time] + tlengths[i]
               tr_age[time] = tr_age[time] + tages[i]
            
           elif polarity[i] == -1:
                l_number[time] = l_number[time] + active[i] 
                l_displacement[time] = l_displacement[time] + displacements[i]
                l_slip[time] = l_slip[time] + slips[i]
                l_length[time] = l_length[time] + lengths[i]
                l_age[time] = l_age[time] + ages[i]
            
                tl_number[time] = tl_number[time] + 1
                tl_displacement[time] = tl_displacement[time] + tdisplacements[i]
                tl_slip[time] = tl_slip[time] + tslips[i]
                tl_length[time] = tl_length[time] + tlengths[i]
                tl_age[time] = tl_age[time] + tages[i]

            
    
    #l_age[time] = l_age[time]/l_number[time]
    #r_age[time] = r_age[time]/r_number[time]
    
    #l_length[time] = l_length[time]/l_number[time]*scale
    #r_length[time] = r_length[time]/r_number[time]*scale
    
    
    times[time] = vistimes[file]/1e6;
    #t_slip_x[time] = sum(slips_x)
    
    # Write csv file for comparisons

csvdat = np.zeros((len(tl_age),6))
csvdat[:,0] = times
csvdat[:,1] = l_number + r_number
csvdat[:,2] = (l_length + r_length)*scale
csvdat[:,3] = (l_slip + r_slip)*1000
csvdat[:,4] = (l_displacement + r_displacement)/1000
csvdat[:,5] = (l_age + r_age)/1e6

csvdat = pd.DataFrame(csvdat)
csvdat.to_csv("data.csv",header=None)
    
##############################################################################
### Create plots

# Filter data to account for flickering faults that appear a single timestep.
#order = 9
#t_number = medfilt(t_number, order)
#l_number = medfilt(l_number, order)
#r_number = medfilt(r_number, order)

#t_displacement = medfilt(t_displacement, order)
#l_displacement = medfilt(l_displacement, order)
#r_displacement = medfilt(r_displacement, order)

#t_slip = medfilt(t_slip, order)
#l_slip = medfilt(l_slip, order)
#r_slip = medfilt(r_slip, order)

#t_length = medfilt(t_length, order)
#l_length = medfilt(l_length, order)
#r_length = medfilt(r_length, order)

#ta_age = medfilt(ta_age, order)
#la_age = medfilt(la_age, order)
#ra_age = medfilt(ra_age, order)

# Set stage information
s1 = 1
s2 = 7
s3 = 11
s4 = 24

yy = np.array([1e6, 1e6])
sx1 = np.array([0, s1])
sx2 = np.array([s1, s2])
sx3 = np.array([s2, s3])
sx4 = np.array([s3, s4])
sx5 = np.array([s4, 50])

def plot_stage(p, alpha=0.15):
    c1 = np.array((255/255,   193/255,   193/255))
    c2 = np.array((255/255,   210/255,   195/255))
    c3 = np.array((255/255,   224/255,   192/255))
    c4 = np.array((167/255,   205/255,   210/255))
    c5 = np.array((172/255,   170/255, 229/255))
    
    gsc = 175
    cb = np.array((gsc/255,   gsc/255,   gsc/255))
    
    #bv1 = [s1, s1]
    intv = 0.5
    bv2 = [s2-intv, s2+intv]
    bv3 = [s3-intv, s3+intv]
    bv4 = [s4-intv, s4+intv] 
    
    ax[p].fill_between(sx1,yy, color=c1, alpha=alpha, linewidth=1,zorder=0)
    ax[p].plot([s1,s1],[0, 1000], color='k',linestyle='--',zorder=1)
    
    #ax[p].fill_between(bv1,yy, color=cb, alpha=alpha, linewidth=0)    
    ax[p].fill_between(sx1,yy, color=c1, alpha=alpha, linewidth=1)
    ax[p].plot([s2,s2],[0, 1000], color='k',linestyle='--')
    
    #ax[p].fill_between(bv2,yy, color=cb, alpha=alpha, linewidth=1) 
    ax[p].fill_between(sx2,yy, color=c2, alpha=alpha, linewidth=1)
    ax[p].plot([s3,s3],[0, 1000], color='k',linestyle='--')
    
    #ax[p].fill_between(bv3,yy, color=cb, alpha=alpha, linewidth=1) 
    ax[p].fill_between(sx3,yy, color=c3, alpha=alpha, linewidth=1)
    ax[p].plot([s4,s4],[0, 1000], color='k',linestyle='--')
    
    #ax[p].fill_between(bv4,yy, color=cb, alpha=alpha, linewidth=1) 
    ax[p].fill_between(sx4,yy, color=c4, alpha=alpha, linewidth=1)
    #ax[p].plot([s4,s4],[0, 1000], color='k',linestyle='--')
    
    ax[p].fill_between(sx5,yy, color=c5, alpha=alpha, linewidth=1)  
    

H = pickle.load(open('./faultloc.p', 'rb')) 
f, ax = plt.subplots(4,1,figsize=(15,14), sharex=False)

labels = ["Active left dipping faults", "Active right dipping faults"]

nums= 10 
cl = sns.color_palette() #palette="muted", n_colors=nums
#colors = ([255/255,  163/255,  23/255,   0.85], [35/255,  119/255,  213/255,  0.85])
colors = (cl[3], cl[0])
#colorc = ([150/255,  150/255,  150/255,   0.4], [35/255,  119/255,  213/255,  0.65])
#colors = ([255/255,  107/255,  107/255,   1], [98/255,  118/255,  200/255,  1])
#colors = ([255/255,  163/255,  23/255,   0.85], [35/255,  119/255,  213/255,  0.85])




p = 0
al = 0.35
centerly = [225, 225]
centerlx = [0, 50]
plot_stage(p, alpha=al)
ax[p].plot(centerlx, centerly,'k:',linewidth=2)
nx.draw(H,
        pos = dict(zip([node for node in H], [(H.nodes[node]['realtime'], H.nodes[node]['x']) for node in H])),
        width = 7.5e2*get_edge_attribute(H, 'slip_rate'),
        node_size = 0.00000000001,
        node_color='white',
        node_shape='s',
        edge_color=get_p_colors(H, 'polarity', cl[3], cl[0]),
        ax=ax[p])



#Ht.set_zorder(20)


fs = 16
limits=ax[p].axis('on') # turns on axis
ax[p].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#ax[p].set_title('Fault location (map view)',fontweight='bold',fontsize=fs+2, loc='left')
ax[p].set_xlim([graph_start, graph_end])
#ax[p].set_xlabel('Time (Myr)',fontsize=12)
ax[p].set_ylabel('X (km)',fontsize=fs)
ax[p].set_ylim([g0,g0max])
ax[p].grid(linewidth=0.2,color='k')
plt.setp(ax[p].spines.values(), linewidth=1.25)
ax[p].tick_params(width=1.25,labelsize=fs, rotation = 90)
 
# First plot things for active faults on left side.  
p0 = 1
#ax[p0].stackplot(times,(tl_number + r_number),labels=["Cumulative faults"],colors=colorc, edgecolor=(1,1,1,1),linewidth=0.25)
#ax[p0].plot(times,(tl_number + r_number),'k',linewidth=0.75)
plot_stage(p0, alpha=al)
ax[p0].stackplot(times,l_number,r_number,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25,zorder=5)
#ax[p0].set_title('Number of faults',fontweight='bold',fontsize=fs+2)
ax[p0].set_xlim([graph_start, graph_end])
ax[p0].set_ylim([0, g1])
#ax[p0].set_xlabel('Time (Myr)',fontsize=12)
ax[p0].set_ylabel('Number of faults',fontsize=fs)
#ax[p0].legend(loc='upper right',fontsize=12)
ax[p0].grid(linewidth=0.2,color='k')
plt.setp(ax[p0].spines.values(), linewidth=1.25)
ax[p0].tick_params(width=1.25,labelsize=fs, rotation = 90)



p2 = 2
plot_stage(p2, alpha=al)
#ax[p2].stackplot(times,(tl_length + tr_length)*scale,labels=["Cumulative faults"],colors=colorc, edgecolor=(1,1,1,1),linewidth=0.25)
#ax[p2].plot(times,(tl_length + tr_length)*scale,'k',linewidth=0.75)
ax[p2].stackplot(times,l_length*scale, r_length*scale,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25,zorder=5)
#ax[p2].set_title('Cumulative length of faults',fontweight='bold',fontsize=fs+2)
ax[p2].set_xlim([graph_start, graph_end])
ax[p2].set_ylim([0, g4])
#ax[p2].set_xlabel('Time (Myr)',fontsize=fs)
ax[p2].set_ylabel('Length (km)',fontsize=fs)
#ax[p2].legend(loc='upper right',fontsize=12)
ax[p2].grid(linewidth=0.2,color='k')
plt.setp(ax[p2].spines.values(), linewidth=1.25)
ax[p2].tick_params(width=1.25,labelsize=fs, rotation = 90)


p3 = 3
plot_stage(p3, alpha=al)
#ax[p3].stackplot(times,(tl_displacement + tr_displacement)/1000,labels=["Cumulative faults"],colors=colorc, edgecolor=(1,1,1,1),linewidth=0.25)
#ax[p3].plot(times,(tl_displacement + tr_displacement)/1000,'k',linewidth=0.75)
ax[p3].stackplot(times,l_displacement/1000, r_displacement/1000,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25,zorder=5)
#ax[p3].set_title('Cumulative displacement of faults',fontweight='bold',fontsize=fs+2)
ax[p3].set_xlim([graph_start, graph_end])
ax[p3].set_ylim([0, g3])
ax[p3].set_xlabel('Time (Myr)',fontsize=fs)
ax[p3].set_ylabel('Displacement (km)',fontsize=fs)
#ax[p3].legend(loc='upper right',fontsize=fs)
ax[p3].grid(linewidth=0.2,color='k')
plt.setp(ax[p3].spines.values(), linewidth=1.25)
ax[p3].tick_params(width=1.25,labelsize=fs, rotation = 90)

plt.tight_layout()
plt.savefig('./graph_images/test.svg', dpi=800)


#p4 = 3
#plot_stage(p4, alpha=al)
#ax[p4].stackplot(times,(tl_slip + tr_slip)*1000,labels=["Cumulative faults"],colors=colorc, edgecolor=(1,1,1,1),linewidth=0.25)
#ax[p4].plot(times,(tl_slip + tr_slip)*1000,'k',linewidth=0.75)
#ax[p4].stackplot(times,l_slip*1000, r_slip*1000,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25)
#ax[p4].set_title('Cumulative slip rate of faults',fontweight='bold',fontsize=fs+2)
#ax[p4].set_xlim([graph_start, graph_end])
#ax[p4].set_ylim([0, 30])
#ax[p4].set_xlabel('Time (Myr)',fontsize=fs)
#ax[p4].set_ylabel('slip (mm/yr)',fontsize=fs)
#ax[p4].legend(loc='upper right',fontsize=12)
#ax[p4].grid(linewidth=0.2,color='k')
#plt.setp(ax[p4].spines.values(), linewidth=1.25)
#ax[p4].tick_params(width=1.25,labelsize=fs, rotation = 90)


# =============================================================================
# p1 = 2
# #ax[p1].stackplot(times,(tl_age+ tr_age)/1e6,labels=["Cumulative faults"],colors=colorc, edgecolor=(1,1,1,1),linewidth=0.25)
# #ax[p1].plot(times,(tl_age+ tr_age)/1e6,'k',linewidth=0.75)
# plot_stage(p1, alpha=al)
# ax[p1].stackplot(times,l_age/1e6,r_age/1e6,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25)
# ax[p1].set_title('Cumulative activity time of faults',fontweight='bold')
# ax[p1].set_xlim([graph_start, graph_end])
# ax[p1].set_ylim([0, g2])
# ax[p1].set_xlabel('Time (Myr)',fontsize=12)
# ax[p1].set_ylabel('Activity time (Myr)',fontsize=12)
# ax[p1].legend(loc='upper right',fontsize=12)
# ax[p1].grid(linewidth=0.2,color='k')
# plt.setp(ax[p1].spines.values(), linewidth=1.25)
# ax[p1].tick_params(width=1.25,labelsize=12)
# =============================================================================



# # First plot things for active faults on left side.  
# p0 = 0
# ax[p0,1].stackplot(times,tl_number,tr_number,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25)
# ax[p0,1].set_title('Number of faults',fontweight='bold')
# ax[p0,1].set_xlim([graph_start, graph_end])
# #ax[0].set_ylim([0, 30])
# ax[p0,1].set_ylabel('Number of faults',fontsize=12,fontweight='bold')
# ax[p0,1].legend(loc='upper left',fontsize=12)
# ax[p0,1].grid(linewidth=0.3)
# plt.setp(ax[p1,1].spines.values(), linewidth=1.25)
# ax[p0,1].tick_params(width=1.25,labelsize=12)

# p1 = 1
# ax[p1,1].stackplot(times,tl_displacement/1000,tr_displacement/1000,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25)
# ax[p1,1].set_title('Cumulative mean displacement of fault system',fontweight='bold')
# ax[p1,1].set_xlim([graph_start, graph_end])
# #ax[3]1set_ylim([0, 125])
# ax[p1,1].set_xlabel('Time (Myr)',fontsize=12,fontweight='bold')
# ax[p1,1].set_ylabel('Age (Myr)',fontsize=12,fontweight='bold')
# ax[p1,1].legend(loc='upper left',fontsize=12)
# ax[p1,1].grid(linewidth=0.3)
# plt.setp(ax[p1,1].spines.values(), linewidth=1.25)
# ax[p1,1].tick_params(width=1.25,labelsize=12)

# p2 = 2
# ax[p2,1].stackplot(times,tl_length*scale, tr_length*scale,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25)
# ax[p2,1].set_title('Cumulative length of fault system',fontweight='bold')
# ax[p2,1].set_xlim([graph_start, graph_end])
# #ax[2].set_ylim([0, 500])
# ax[p2,1].set_xlabel('Time (Myr)')
# ax[p2,1].set_ylabel('Cumulative fault length (km)',fontsize=12,fontweight='bold')
# ax[p2,1].legend(loc='upper left',fontsize=12)
# ax[p2,1].grid(linewidth=0.3)
# plt.setp(ax[p2,1].spines.values(), linewidth=1.25)
# ax[p2,1].tick_params(width=1.25,labelsize=12)

# p3 = 3
# ax[p3,1].stackplot(times,tr_slip*1000,tl_slip*1000,labels=labels,colors=colors, edgecolor=(1,1,1,1),linewidth=0.25)
# ax[p3,1].set_title('Cumulative mean slip rate of fault system',fontweight='bold')
# ax[p3,1].set_xlim([graph_start, graph_end])
# #ax[p3,1].set_ylim([0, 75])
# ax[p3,1].set_xlabel('Time (Myr)',fontsize=12,fontweight='bold')
# ax[p3,1].set_ylabel('Slip rate (mm/yr)',fontsize=12,fontweight='bold')
# ax[p3,1].legend(loc='upper left',fontsize=12)
# ax[p3,1].grid(linewidth=0.3)
# plt.setp(ax[p3,1].spines.values(), linewidth=1.25)
# ax[p3,1].tick_params(width=1.25,labelsize=12)


#plt.savefig('./graph_images/test.svg', dpi=800)
#plt.close('all')








stop = timeit.default_timer()

print('Polarity surface time: ', stop - tstart) 
