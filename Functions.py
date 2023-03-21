# Math, image processing and other useful libraries
from __future__ import print_function, unicode_literals, absolute_import, division
import os

import pandas as pd
import numpy as np
import cv2
from collections import OrderedDict
import copy
import math
import pickle
from matplotlib.ticker import MaxNLocator
from itertools import combinations

# Image processing
from skimage.measure import regionprops
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu
from skimage.morphology import extrema, skeletonize
from skimage.transform import probabilistic_hough_line
from skimage.draw import disk, circle_perimeter
from scipy.ndimage import gaussian_filter, grey_closing
from scipy.spatial import distance_matrix
from skimage import data, restoration, util
from roipoly import RoiPoly
from matplotlib_scalebar.scalebar import ScaleBar
from biosppy.signals import tools
from biosppy.stats import pearson_correlation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as pltc
import matplotlib.colors as colors
import seaborn as sns

# Widgets
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from IPython.display import display

# Feature Extraction (.py files by Teresa Parreira)
# from CytoSkeletonPropsMorph import CytoSkeletonPropsMorph
# from CytoSkeletonRegionPropsInt import RegionPropsInt
# from FreqAnalysis import FreqAnalysis
# from GLCM import GLCM

# Graph
import sknw
import networkx as nx
from scipy.signal import argrelextrema

# 
from skan import Skeleton, summarize,draw
from skan.csr import skeleton_to_csgraph, sholl_analysis,make_degree_image
import scipy as sp
import scipy.sparse
from matplotlib.patches import Circle
from ImageFeatures import ImageFeatures


#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC







def FeaturesFromCentroid(image,index_c,centroid,radius,threshold,line_length,line_gap,ImageLinesDF,plot):
    # image = [image, image index, image name, image label]
    
    # Get disks around centroid
    rr_big, cc_big     = disk((centroid[0],centroid[1]), radius+10, shape=(1040,1388))
    disk_pixels_big    = [(rr_big[x],cc_big[x]) for x in range(len(cc_big))]
    rr_small, cc_small = disk((centroid[0],centroid[1]), radius, shape=(1040,1388))
    disk_pixels_small  = [(rr_small[x],cc_small[x]) for x in range(len(cc_small))]

    # Get and Plot disk
    black = np.zeros((1040,1388))
    black[rr_big, cc_big] = 1
    #plt.imshow(black,cmap='gray')

    # Local texture
    local_texture = image[0] * black

    # Patch
    #patch = local_texture[min(rr):max(rr),min(cc):max(cc)].astype('int') # Convert values to int to avoid datatype problems

    # If patch == 0, we're analyzing background
    if len(np.unique(local_texture)) == 1:
        return ImageLinesDF

    lines = probabilistic_hough_line(local_texture, threshold=threshold, line_length=line_length, line_gap=line_gap)

    # Distance Matrix

    # Filter and analyze lines within disk
    angles    = []
    dist_med  = []
    triangleA = []
    line_size = []
    for line in lines:
        p0, p1 = line

        # Median point
        med_point = ((p0[0] + p1[0])/2,(p0[1] + p1[1])/2)

        if (round(med_point[1]),round(med_point[0])) in disk_pixels_small:

            # Remove from aux
            #lines.remove(line)

            # Prepare vectors
            center_med_vec = np.array(med_point) - np.array([centroid[1],centroid[0]])
            line_vec       = np.array(p0) - np.array(p1)
            center_p0      = p0 - np.array([centroid[1],centroid[0]])
            center_p1      = p1 - np.array([centroid[1],centroid[0]])

            # Statistics
            angle = np.arccos(np.dot(center_med_vec / np.linalg.norm(center_med_vec), line_vec / np.linalg.norm(line_vec)))*180/np.pi
            if angle > 90:
                angle = 180 - angle
            angles    += [angle]
            dist_med  += [np.linalg.norm(center_med_vec)]
            triangleA += [abs(0.5*np.cross(center_p0,center_p1))]
            line_size += [np.linalg.norm(line_vec)]
            
            if plot:
                colours = pltc.Reds_r(plt.Normalize(0, 90)(angle))
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]),linewidth=0.8,color=colours,alpha=1)
                
    # ADD ANGLES OF GIVEN CENTROID TO DATAFRAME
    if angles != [] and dist_med!=[] and triangleA !=[] and line_size!=[]:
        new       = pd.DataFrame(data = {'Name': [TextureDF['Name'][image[1]]], 'Label': [TextureDF['Label'][image[1]]], 'Angles': [angles], 'Distances to Centroid': [dist_med], 'Triangle Areas': [triangleA], 'Line Lengths': [line_size]},index=[index_c])
        ImageLinesDF = ImageLinesDF.append(new,ignore_index=False)
        
    if plot:
        #plt.scatter(row_c['Centroid'][1],row_c['Centroid'][0], s=80, edgecolors='r', facecolors='none')
        draw_circle = plt.Circle((centroid[1],centroid[0]), radius, fill=False, color = 'k')
        plt.gcf().gca().add_artist(draw_circle)
        #axes.set_aspect(1)
        
    return ImageLinesDF
    
    
    
def cv2toski(lines):
    lines = lines.reshape(lines.shape[0],lines.shape[2])
    new = []
    for line in lines:
        new += [((line[0],line[1]),(line[2],line[3]))]
        
    return new

def pylsdtoski(lines):
    new = []
    for line in lines:
        new += [((line[0],line[1]),(line[2],line[3]))]
        
    return new

def polar_to_cartesian(lines):
    res = []
    lines = lines.reshape((lines.shape[0],lines.shape[2]))
    for i in lines:
        rho,theta = i[0],i[1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        res += [((x1,y1),(x2,y2))]
        
    return res

# Useful functions
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_hist(feat,bins):
    cmap = pltc.Reds
    global new_cmap
    new_cmap = truncate_colormap(cmap, 0.3, 1, 300)
    
    global data
    data = ImageLinesDF.tail(1)
    global histog,bin_edges
    
    if feat == 'Distances to Centroid':
        bins = np.arange(0, 280 + 5, 5)
    if feat == 'Triangle Areas':
        bins = np.arange(0, 4600 + 5, 5)
    if feat == 'Line Lengths':
        # max(data[feat][data.index[0]])
        bins = np.arange(0, 220 + 5, 5)
    #if feat == 'Theta':
    #    bins = np.arange
    #if feat == 'Angle Difference':
    #    bins = np.arange
    
    histo = np.histogram(data[feat][data.index[0]],bins=bins)
    
    # AX1
    ax1 = plt.subplot(1,1,1)
    ax1.set_ylabel('Absolute Frequency')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    global colours
    #colours = pltc.Reds(plt.Normalize(0, max(histog[0]))(histog[0]),alpha=0.7)
    #colours = new_cmap(plt.Normalize(0, max(histo[0]))(histo[0]),alpha=0.7)

    #ax1.bar(histog[1][:-1],histog[0],color=colours,zorder=5)
    histog, bin_edges, patches = ax1.hist(data[feat][data.index[0]], bins=bins,color='k',alpha=0.7)
    #ax1.color = new_cmap(plt.Normalize(0, max(histog))(histog),alpha=0.7)
    for c, p in zip(histog, patches):
        plt.setp(p, 'facecolor', new_cmap(plt.Normalize(0, max(histog))(c), alpha = 0.7))
    
    # AX2
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Relative Frequency')  
    _ = ax2.hist(data[feat][data.index[0]], bins=bins, density=True, alpha=0)
    #ax2.plot(histog[1][:-1],(histog[0]/np.trapz(histog[0],x=histog[1][:-1])),'--',alpha=0,zorder=1)
    plt.grid(alpha=0.3)
    
    if feat == 'Angles':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Angle between Centroid and Line Segment',fontsize=12)
        ax1.set_xticks(bin_edges)
    if feat == 'Distances to Centroid':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Distance between Centroid and Line Segment',fontsize=12)
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        #ax1.set_xticks(np.linspace(0,130,10,endpoint=True,dtype=int))
        
    if feat == 'Triangle Areas':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Triangle Areas between Centroid and Line Segment',fontsize=12)
        #ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        
    if feat == 'Line Lengths':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Line Length',fontsize=12)
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        #ax1.set_xticks(np.linspace(0,170,10,endpoint=True,dtype=int))
        
    if feat == 'Theta':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Line Segment Angle',fontsize=12)
        ax1.set_xticks(bin_edges)
        
    if feat == 'Angle Difference':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Angle Difference (º)',fontsize=12)
        ax1.set_xticks(bin_edges)
        
    plt.show()
    
def plot_pie(feat,Max):
    cmap = pltc.Reds
    global new_cmap
    new_cmap = truncate_colormap(cmap, 0.3, 1, 300)
    
    global data
    data = ImageLinesDF.tail(1)
    
    
    # Pie chart
    labels = [feat, '-']
    if feat == 'Number of Lines':
        sizes = [data[feat][data.index[0]], Max]
        
    colors = [new_cmap(plt.Normalize(0, Max)(data[feat][data.index[0]]), alpha = 0.7),'w']

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, colors = colors, labels=[sizes[0],'-'], startangle=90)
    #draw circle
    centre_circle = plt.Circle((0,0),0.70,fc='white',ec='black')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.show()
    
def remove_not1D(feats_labels,feats_values):
    feats_labels_filtered = []
    feats_values_filtered = []
    for i in range(len(feats_values)):
        data_type = type(feats_values[i])
        if data_type != tuple and data_type != list and data_type != np.ndarray and data_type != str :
            feats_values_filtered.append(feats_values[i])
            feats_labels_filtered.append(feats_labels[i])
            
        if feats_labels[i] == 'Centroid':
            feats_values_filtered.append(feats_values[i])
            feats_labels_filtered.append(feats_labels[i])
                
#         else:
#             print('Removed feature: ', feats_labels[i]) 
    return feats_labels_filtered, feats_values_filtered

def branch(sk,ske,cyto_info,nuclei_info,plot):
    
    # Feature Names and Values
    res = []
    
    # Get branch data:
    branch_data = summarize(ske,find_main_branch=False)
    
    # Number of paths (1D):
    #global Npaths
    Npaths = ske.n_paths
    res += [('SKNW:Number of Branches',Npaths)]
    
    # Branch distance, Mean/Std Pixel Values, Eucl distance
    cols = ['branch-distance','mean-pixel-value','stdev-pixel-value','euclidean-distance']
    for col in cols:
        names    = tools.signal_stats(list(branch_data[col]))._names
        features = np.array(list(tools.signal_stats(list(branch_data[col]))))
        res += list(zip(['SKNW:' + str(col) +str(' ') + str(f) for f in names],features))
    
    # Path grouping
    #global br_type_lens
    br_type_nams = ['Endpoint-to-endpoint (isolated branch)','Junction-to-endpoints','Junction-to-junctions','Isolated cycles']
    
    
    for typ in np.unique(branch_data['branch-type']):
        data = branch_data[branch_data['branch-type'] == typ]
        
        res += [(str('SKNW:Number of ') + str(br_type_nams[typ]),len(data)),
                (str('SKNW:Ratio of ') + str(br_type_nams[typ]),len(data)/Npaths)]
        
        for col_ in cols:
            res += [(str('SKNW:Mean of ') + str(br_type_nams[typ]) + str(' ') + str(col_),np.mean(data[col_])),
                    (str('SKNW:Std of ') + str(br_type_nams[typ]) + str(' ') + str(col_),np.std(data[col_]))]
            
    if 3 not in np.unique(branch_data['branch-type']): #isolated cycles
        data = np.zeros_like(branch_data[branch_data['branch-type'] == 1])
        
        res += [(str('SKNW:Number of ') + str(br_type_nams[3]),0),
                (str('SKNW:Ratio of ') + str(br_type_nams[3]),0)]
        
        for col_ in cols:
            res += [(str('SKNW:Mean of ') + str(br_type_nams[3]) + str(' ') + str(col_),0),
                    (str('SKNW:Std of ') + str(br_type_nams[3]) + str(' ') + str(col_),0)]
        
    #br_type_lens = [len(branch_data[branch_data['branch-type'] == typ]) for typ in np.unique(branch_data['branch-type'] )]
    
#     res += [('SKNW:Number of Endpoint-to-endpoint (isolated branch)',br_type_lens[0]),
#             ('SKNW:Ration of Endpoint-to-endpoint (isolated branch)',br_type_lens[0]/Npaths),
#             ('SKNW:Number of Junction-to-endpoints',br_type_lens[1]),
#             ('SKNW:Number of Junction-to-endpoints',br_type_lens[1]/Npaths),
#             ('SKNW:Number of Junction-to-junctions',br_type_lens[2]),
#             ('SKNW:Number of Junction-to-junctions',br_type_lens[2]/Npaths),
#             ('SKNW:Number of Isolated cycles',br_type_lens[3]),
#             ('SKNW:Number of Isolated cycles',br_type_lens[3]/Npaths)]
    
    

    # Plot
    if plot: 
        # Connectivity Image
        fig_branch,ax_branch = plt.subplots(figsize=(8,8))
        ax_branch.imshow(make_degree_image(sk)) #yellow = junction to junction, #light green -junction to endpoint, #dark green - endpoint-to-endpoint
        ax_branch.set_title('Connectivity')
        ax_branch.set_xlim(min(cyto_info[1]), max(cyto_info[1]))
        ax_branch.set_ylim(min(cyto_info[0]), max(cyto_info[0]))
        fig_branch.show()
        
    return res
    
#r = branch(ske,[x_,y_],[row['ROImask'] * NucleiDeconvDF['Image'][row['Index']],centroid,cr],False)


def graphAnalysis(sk,infocyto,infonucl,mask,plot):
    global skeleton
    # infocyto = [x_,y_]
    # infonucl = [NucleiDeconvDF['Image'][row['Index']],centroid,cr] 
    #graph = sknw.build_sknw(skeleton,multi=False,**[{'iso':False}])
#     graph = sknw.build_sknw(sk)
    
#     # Get branch sizes and turtuosity
#     sizes = []
#     eucl  = []
#     global s,e,ps,ps_
#     for (s,e) in graph.edges():
#         ps = graph[s][e]['pts']
#         sizes += [len(ps)]
        
#         minx,maxx,miny,maxy = min(ps[:,0]),max(ps[:,0]),min(ps[:,1]),max(ps[:,1])
#         eucl += [np.sqrt((maxx - minx)**2 + (maxy - miny)**2)]

#     # PLOT
#     if plot:
#         plt.figure(figsize=(15,15))
#         plt.imshow(np.zeros_like(sk), cmap='gray')
#         # draw edges by pts
#         for (s,e) in graph.edges():
#             ps = graph[s][e]['pts']
#             plt.plot(ps[:,1], ps[:,0], 'white')
#         # draw node by o
#         nodes = graph.nodes()
#         ps_ = np.array([nodes[i]['o'] for i in nodes])
#         plt.plot(ps_[:,1], ps_[:,0], 'r.')
#         plt.title('Cytoskeleton Graph')
#         plt.axis('off')
#         plt.show()
        
    #pixel_graph, coordinates = skeleton_to_csgraph(skeleton * DeconvDF['Image'][row['Index']])
    ske = Skeleton((sk).astype(float)) 
#     skeleton_to_csgraph
#     branch_data = summarize(ske)
    
#     sizes = ske.path_lengths()
#     med_int = ske.path_means()
#     std_int = ske.path_stdev()
    
    # Branch Analysis
    branch_feats = branch(sk,ske,[infocyto[0],infocyto[1]],[mask * infonucl[0],infonucl[1],infonucl[2]],plot)
                   
    # Sholl Analysis
    sholl_feats,shollhist = sholl(sk,[infocyto[0],infocyto[1]],[mask * infonucl[0],infonucl[1],infonucl[2]],plot)
    
    #G = nx.from_scipy_sparse_matrix(pixel_graph)  ou nx.from_scipy_sparse_matrix(ske.graph) representação N+1??? nx.density
    
    return ske.graph,branch_feats + sholl_feats,shollhist

#graph,graph_feats = graphAnalysis(skeleton * DeconvDF['Image'][row['Index']],True)

# Quantitative Analysis
def quantitative_analysis(ImageLinesDF):
    feats = list(ImageLinesDF.columns[2:])
    cmap = pltc.Reds
    global new_cmap
    new_cmap = truncate_colormap(cmap, 0.3, 1, 300)
    
    #histog = np.histogram(np.array(feat_concat[~np.isnan(feat_concat)]),bins=30)
    # plt.plot(histog[1][:-1],histog[0]/np.sum(histog[0]),color=colors[labels.index(label)],linewidth=4,label = label,alpha=0.7)
    flag = False
    global feat
    for feat in feats:

        if feat == 'Angles':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=180)
            #plot_hist(feat=feat,bins=np.linspace(0,90,91))
            plot_hist(ImageLinesDF=ImageLinesDF,feat=feat,bins=np.linspace(0,90,10))

        elif feat == 'Distances to Centroid':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=30)
            plot_hist(feat=feat,bins=15)
            #aux = []
            #for i in range(len(histog[1])-1):
            #    aux += [np.pi*(histog[1][i+1])**2 - np.pi*(histog[1][i])**2]
            #ax2.plot(histog[1][:-1],(histog[0]/np.trapz(histog[0],x=histog[1][:-1]))/aux,'--',label='uga',alpha=0)
 
        elif feat == 'Triangle Areas':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=30)
            plot_hist(feat=feat,bins=15)
        
        elif feat == 'Line Lengths':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=30)
            plot_hist(feat=feat,bins=15)
            
        elif feat == 'Theta':
            plot_hist(feat=feat,bins=np.linspace(0,90,10))
            
        elif feat == 'Angle Difference':
            plot_hist(feat=feat,bins=np.linspace(0,90,10))
            
        elif feat == 'Number of Lines':
            plot_pie(feat=feat,Max=400)
            #flag = True
            
        else:
        #    #print("Fractal Dimension: " + str(ImageLinesDF[feat][ImageLinesDF[feat].index[0]]))
            print(str(feat) + ": " + str(ImageLinesDF.tail(1)[feat][ImageLinesDF.tail(1).index[0]]))
    
    global comb
    comb = combinations(range(len(feats[:6])), 2)
    data = ImageLinesDF.tail(1)
    for f in list(comb):
        fig, ax = plt.subplots()
        f1 = feats[f[0]]
        f2 = feats[f[1]]
        if f1 == 'Angles' or f1 == 'Theta' or f1 == 'Angle Difference':
            bin1 = np.linspace(0,90,10)
        else:
            bin1 = 15
        if f2 == 'Angles' or f2 == 'Theta' or f2 == 'Angle Difference':
            bin2 = np.linspace(0,90,10)
        else:
            bin2 = 15
            
        hi, xedges, yedges, image = plt.hist2d(data[feats[f[0]]][data.index[0]],data[feats[f[1]]][data.index[0]],bins=[bin1,bin2],cmap=cmap,alpha=0.7)   
        ax.set_xticks(xedges)
        ax.set_yticks(np.linspace(0,yedges[-1],10,endpoint=True,dtype=int))
        ax.set_facecolor(cmap(0))
        plt.xlabel(feats[f[0]])
        plt.ylabel(feats[f[1]])
        #plt.title('Line Segment Angles vs. Distances to Centroid')
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Absolute Frequency")
        plt.show()
        
#     # HEATMAP 1
#     fig, ax = plt.subplots()
#     data = ImageLinesDF.tail(1)
#     hi, xedges, yedges, image = plt.hist2d(data['Angles'][data.index[0]],data['Distances to Centroid'][data.index[0]],bins=[np.linspace(0,90,10), np.arange(0, 280 + 5, 5)],cmap=cmap,alpha=0.7)   
#     ax.set_facecolor(cmap(0))
#     plt.xlabel('Angle (º)')
#     plt.ylabel('Distance to Centroid (pixels)')
#     plt.title('Line Segment Angles vs. Distances to Centroid')
#     cbar = fig.colorbar(image, ax=ax)
#     cbar.set_label("Absolute Frequency")
#     plt.show()
    
#     # HEATMAP 2
#     fig, ax = plt.subplots()
#     data = ImageLinesDF.tail(1)
#     hi, xedges, yedges, image = plt.hist2d(data['Line Lengths'][data.index[0]],data['Distances to Centroid'][data.index[0]],bins=[30,30],cmap=cmap,alpha=0.7)   
#     plt.xlabel('Line Lengths (pixels)')
#     plt.ylabel('Distance to Centroid (pixels)')
#     plt.title('Line Segment Lengths vs. Distances to Centroid')
#     cbar = fig.colorbar(image, ax=ax)
#     cbar.set_label("Absolute Frequency")
#     plt.show()

def hist_bin(feat):
    if feat == 'Angles' or feat == 'Theta' or feat == 'Angle Difference':
        #bins  = np.linspace(0,90,18,endpoint=False,dtype=int)
        bin_ = np.linspace(0,95,19,endpoint=False,dtype=int)
        bin_x = np.linspace(0,95,19,endpoint=False,dtype=int)
    elif feat == 'Triangle Areas':
        bin_ = np.linspace(0,2400,150,endpoint=True,dtype=int)
        bin_x = np.arange(0,2400,200)
    elif feat == 'Distances to Centroid':
        bin_ = np.linspace(0,350,70,endpoint=True,dtype=int)
        bin_x = np.arange(0,350,20)
    elif feat == 'Line Lengths':
        bin_ = np.linspace(0,215,107,endpoint=True,dtype=int)
        bin_x = np.arange(0,215,10)
    else:
        bin_  = np.linspace(0,90,18,endpoint=True,dtype=int)
        bin_x = np.linspace(0,95,19,endpoint=False,dtype=int) 
    return bin_,bin_x

def hist_lim(feat):
    if feat == 'Angles' or 'Theta' or 'Angle Difference':
        return [0, 90]
        #return np.linspace(0,90,10,endpoint=True,dtype=int)
    elif feat == 'Distances to Centroid':
        return [0, 150]
        #return np.linspace(0,150,10,endpoint=True,dtype=int)
    elif feat == 'Triangle Areas':
        return [0, 500]
        #return np.linspace(0,800,10,endpoint=True,dtype=int)
    elif feat == 'Line Lengths':
        return [0, 40]
        #return np.linspace(0,40,10,endpoint=True,dtype=int)
    else:
        return 0
    
def create_separate_DFs(DF):
    global LSF
    LSF = DF[DF.columns[[x.startswith("LSF2D") for x in DF.columns]]]
    try:
        fts = tools.signal_stats(eval(LSF.loc[LSF.index[0]]['LSF2D:Angles']))._names
    except:
        fts = tools.signal_stats(LSF.loc[LSF.index[0]]['LSF2D:Angles'])._names
    res = pd.DataFrame()
    for ft in LSF.columns: 
        try:
            temp = np.array([list(tools.signal_stats(cell)) for cell in LSF[ft]])
        except:
            temp = np.array([list(tools.signal_stats(eval(cell))) for cell in LSF[ft]])
        res  = pd.concat([res, pd.DataFrame(temp,columns = [ft+str(" ")+i for i in fts])],axis=1)
    res.index = LSF.index
    
    # Concatenate with 1D features
    LSF = pd.concat([res, DF[DF.columns[[x.startswith("LSF1D") for x in DF.columns]]]],axis=1)
    
    DCF  = DF[DF.columns[[x.startswith("DCF") for x in DF.columns]]]
    DNF  = DF[DF.columns[[x.startswith("DNF") for x in DF.columns]]]
    SKNW = DF[DF.columns[[x.startswith("SKNW") for x in DF.columns]]]
    OTHERS = DF[DF.columns[[x.startswith("OTHERS") for x in DF.columns]]]
    FULL = pd.concat([LSF, DCF, DNF, SKNW, OTHERS],axis=1)
    
    return LSF,DCF,DNF,SKNW,OTHERS,FULL 





def resample_mine(image,f):
    c = np.zeros((int(image.shape[0]),int(image.shape[1]/2),int(image.shape[2]/2)))
    for i in range(len(image)):
        c[i] = scipy.ndimage.zoom(image[i], f, order=0)
    return c

# def counter():
#     def naturals(i):
#     count=i
#     while True:
#         yield count
#         count+=1

#     global counter
#     counter = naturals(0)
