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
from framework.ImageFeatures import ImageFeatures,getvoxelsize
from framework.Functions import cv2toski,pylsdtoski,polar_to_cartesian, remove_not1D, quantitative_analysis,hist_bin,hist_lim,branch,graphAnalysis
from framework.Importing import *
#from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation
from framework.Processing import process3Dnuclei,analyze_cell

#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC
      
def set_background(color): 
    """
    Set background color to a Jupyter Notebook cell
        - ```color``` = HEX code of the input color (allows opacity)
    """
    
    from IPython.display import HTML, display
    script = (         
        "var cell = this.closest('.code_cell');"         
        "var editor = cell.querySelector('.input_area');"         
        "editor.style.background='{}';"         
        "this.parentNode.removeChild(this)"     
    ).format(color)      
    
    display(HTML('<img src onerror="{}">'.format(script)))



def plot_nuclei_contours(CentroidsDF,imgIndex,ax):
    for index,row in CentroidsDF[imgIndex].iterrows():
        if type(imgIndex) != int:
            ax.plot(row['Centroid'][1],row['Centroid'][0],'o',color='r',markersize=7,zorder=5)
        else:
            ax.plot(row['Centroid'][1],row['Centroid'][0],'o',color='b',markersize=7,zorder=5)
            try:
                contourr  = row['Contour'][0]
                cr = contourr.reshape((contourr.shape[0],contourr.shape[2]))
            except:
                contourr  = row['Contour']
                cr = contourr.reshape((contourr.shape[0],contourr.shape[2]))
            ax.plot(cr[:,0],cr[:,1],'--',color='w',zorder=11,linewidth=2)
            

            
            
def intensity_plotter(ResultsRow,data,save):
    from matplotlib.colors import LinearSegmentedColormap
    
    # Color map
    colors = [(1, 1, 1), (1, 0, 0)] # first color is white, last is red
    cm     = LinearSegmentedColormap.from_list("Custom", colors, N=300)
    
    ##### FIGURE 1
    # Initialize figure 1
    fig,ax = plt.subplots(figsize=(8,8))
    ax.imshow(ResultsRow['Mask']*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']],cmap=cm)
    ax.axis('off')
    
    # Plot Nucleus Centroid and Cytoskeleton Centroid
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'.',color='#6495ED',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=15,zorder=8,fillstyle='none')
    ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8)
    
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'.',color='r',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=15,zorder=8,fillstyle='none',)
    ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=12,zorder=8)
    
    # Plot Nucleus Contour
    ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
    
    # Set x and y lims and title
    ax.set_ylim([min(ResultsRow['Offset'][0]),max(ResultsRow['Offset'][0])])
    ax.set_xlim([min(ResultsRow['Offset'][1]),max(ResultsRow['Offset'][1])])
    
    # Scale Bar
    scalebar = ScaleBar(1,"px",color='k',box_alpha=0,dimension='pixel-length') 
    ax.add_artist(scalebar)
    
    # Adjust and Show
    if save:
        plt.savefig(".//output1.png",format='png',transparent=True,bbox_inches='tight',dpi=300)
    fig.show()
    
    
    ##### FIGURE 2
    # Initialize figure 2
    fig,ax = plt.subplots(figsize=(8,8))
    intensity = data['CYTO']['Image'][ResultsRow['Img Index']] / np.max(data['CYTO']['Image'][ResultsRow['Img Index']])
    aux = intensity * ResultsRow['Mask']
    aux = aux / np.max(aux)
    ax.imshow(ResultsRow['Mask']*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']]*aux,cmap=cm)

    ax.axis('off')
    
    # Plot Nucleus Centroid and Cytoskeleton Centroid
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'.',color='#6495ED',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=15,zorder=8,fillstyle='none')
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8,alpha=0.5)
    
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'.',color='r',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=15,zorder=8,fillstyle='none',)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=12,zorder=8,alpha=0.5)
    
    # Plot Nucleus Contour
    #ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5,alpha=0.5)
         
    # Set x and y lims and title
    ax.set_ylim([min(ResultsRow['Offset'][0]),max(ResultsRow['Offset'][0])])
    ax.set_xlim([min(ResultsRow['Offset'][1]),max(ResultsRow['Offset'][1])])
    #ax.set_title(feat,fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cm), ax=ax, shrink = 0.4)
    cbar.set_label('Pixel Intensity',fontfamily='arial',fontsize=12)
    #cbar.set_ticks(np.linspace(0,90,10,endpoint=True))
    
    # Scale Bar
    scalebar = ScaleBar(1,"px",color='k',box_alpha=0,dimension='pixel-length') 
    ax.add_artist(scalebar)
    
    # Adjust and Show
    if save:
        plt.savefig(".//output.png",format='png',transparent=True,bbox_inches='tight',dpi=300)
    fig.show()
    
    return print('Done.')            
            
# intensity_plotter(ResultsRow=ResultsDF.loc[0],TextureDF=TextureDF,save=True)    
            
            

# def line_plotter(ResultsRow,TextureDF,feat,cmap,normalize_bounds,colorbar_label,overlay_sk,save):
#     fig,ax = plt.subplots(figsize=(8,8))
    
#     # Plot background
#     if overlay_sk == False:
#         ax.imshow(np.zeros((1040, 1388)),cmap='gray',alpha=0) 
#     else:
#         ax.imshow(1-ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']],cmap='gray')
#     ax.axis('off')
    
#     # Get bounds for color map
#     if normalize_bounds == 'default':
#         normalize_bounds = [0,np.max(ResultsRow[feat])]

#     # Plot Nucleus Centroid and Cytoskeleton Centroid
#     #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'.',color='#6495ED',markersize=5,zorder=8)
#     #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=15,zorder=8,fillstyle='none')
#     ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8)
    
#     #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'.',color='r',markersize=5,zorder=8)
#     #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=15,zorder=8,fillstyle='none',)
#     ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=12,zorder=8)
    
#     # Plot Nucleus Contour
#     ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
                
#     # Plot segments colored by feature value
#     for l in range(len(ResultsRow['Lines'])):
#             # Get line = [p0,p1]. 
#             p0, p1 = ResultsRow['Lines'][l]
#             colour   = cmap(plt.Normalize(normalize_bounds[0], normalize_bounds[1])(ResultsRow[feat][l]))
#             ax.plot((p0[0], p1[0]), (p0[1], p1[1]),linewidth=2,color=colour,alpha=1)

#     # Set x and y lims and title
#     ax.set_ylim([min(ResultsRow['Patches'][4]),max(ResultsRow['Patches'][4])])
#     ax.set_xlim([min(ResultsRow['Patches'][5]),max(ResultsRow['Patches'][5])])
#     ax.set_title(feat,fontsize=12)
    
#     # Colorbar
#     cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(normalize_bounds[0], normalize_bounds[1]), cmap=cmap), ax=ax, shrink = 0.4)
#     cbar.set_label(colorbar_label,fontfamily='arial',fontsize=12)
#     #cbar.set_ticks(np.linspace(0,90,10,endpoint=True))
    
#     # Scale Bar
#     scalebar = ScaleBar(1,"px",color='k',box_alpha=0,dimension='pixel-length') 
#     ax.add_artist(scalebar)
    
#     # Adjust and Show
#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     if save:
#         plt.savefig(".//output.png",format='png',transparent=True,bbox_inches='tight',dpi=300)
#     fig.show()
    
#     return print('Done.')

# Color map
# colors = [(0.5, 0.5, 0.5), (1, 0, 0)] # first color is black, last is red
# cm     = LinearSegmentedColormap.from_list("Custom", colors, N=300)

#cmap     = pltc.rainbow
#cm       = truncate_colormap(cmap, 0, 1, 300)


#line_plotter(ResultsRow=ResultsDF.loc[2],TextureDF=TextureDF,feat='LSF2D:Distances to Centroid',cmap=cm,normalize_bounds='default',colorbar_label='Degrees',overlay_sk=True,save=False)




# # Color map
# # colors = [(0.5, 0.5, 0.5), (1, 0, 0)] # first color is black, last is red
# # cm     = LinearSegmentedColormap.from_list("Custom", colors, N=300)

# #cmap     = pltc.rainbow
# #cm       = truncate_colormap(cmap, 0, 1, 300)

# #print('line_plotter')
# #line_plotter(ResultsRow=ResultsDF.loc[0],TextureDF=TextureDF,feat='LSF2D:Distances to Centroid',cmap=cm,normalize_bounds='default',colorbar_label='Pixels',overlay_sk=True,save=False)










    
    

#cmap     = pltc.rainbow
#cm       = truncate_colormap(cmap, 0, 1, 300)

#print('graph_plotter')
#data = graph_plotter(ResultsRow=ResultsDF.loc[0],feat='branch-type',cmap=cm,normalize_bounds='default',colorbar_label='Pixels',nodes=False,main_branch=True, save=True)
#line_plotter(ResultsRow=ResultsDF.loc[2],TextureDF=TextureDF,feat='LSF2D:Distances to Centroid',cmap=cm,normalize_bounds='default',colorbar_label='Degrees',overlay_sk=True,save=False)





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
    
    
#use: stackedbarplots(ResultsDF)    
def stackedbarplots(ResultsDF):
    r = [0,1,2,3]
    l,f1,f2,f3,f4 = 'Label','SKNW:Ratio of Endpoint-to-endpoint (isolated branch)','SKNW:Ratio of Junction-to-endpoints','SKNW:Ratio of Junction-to-junctions','SKNW:Ratio of Isolated cycles'
    data = pd.concat([ResultsDF[f1],ResultsDF[f2],ResultsDF[f3],ResultsDF[f4]],axis=1)
    
    colors = ["#2ECC71","#FFA500","#E74C3C","#BC544B"]
    labels = ['WT','Dup41_46','Del38_46','Mut394']
    pairs  = [(('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394'))]
    
    # From raw value to percentage
    totals = [i+j+k+w for i,j,k,w in zip(ResultsDF[f1], ResultsDF[f2], ResultsDF[f3],ResultsDF[f4])]
    greenBars = [i / j * 100 for i,j in zip(ResultsDF[f1], totals)]
    orangeBars = [i / j * 100 for i,j in zip(ResultsDF[f2], totals)]
    blueBars = [i / j * 100 for i,j in zip(ResultsDF[f3], totals)]
    fourBars = [i / j * 100 for i,j in zip(ResultsDF[f4], totals)]

    # plot
    barWidth = 0.85
    names = ('A','B','C','D','E')
    # Create green Bars
    plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
    # Create orange Bars
    plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
    # Create blue Bars
    plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)
    # Create last Bars
    plt.bar(r, fourBars, bottom=[i+j+k for i,j,k in zip(greenBars, orangeBars,blueBars)], color='#a3acff', edgecolor='white', width=barWidth)

    
    # Custom x axis
    #plt.xticks(r, names)
    #plt.xlabel("group")


    fig,ax = plt.subplots()
    sns.barplot(x=l, y=[f1,f2,f3,f4], data=data, estimator=sum, ci=95,  color='lightblue')
    plt.show()