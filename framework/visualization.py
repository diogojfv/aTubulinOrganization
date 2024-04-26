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
from framework.processing_DCF import *
from framework.Functions import cv2toski,pylsdtoski,polar_to_cartesian, remove_not1D, quantitative_analysis,hist_bin,hist_lim,branch,graphAnalysis
from framework.importing import *
#from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation
from framework.processing import *
from framework.processing_CNF import *

     
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

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_nuclei_contours(CentroidsDF,imgIndex,coordxy,ax):
    for index,row in CentroidsDF[imgIndex].iterrows():
        if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(coordxy[0],coordxy[1])):
            if type(imgIndex) != int:
                ax.plot(row['Centroid'][1],row['Centroid'][0],'o',color='r',markersize=7,zorder=5)
            else:
                ax.plot(row['Centroid'][1],row['Centroid'][0],'o',color='#6495ED',markersize=7,zorder=5)
                try:
                    contourr  = row['Contour'][0]
                    cr = contourr.reshape((contourr.shape[0],contourr.shape[2]))
                except:
                    contourr  = row['Contour']
                    cr = contourr.reshape((contourr.shape[0],contourr.shape[2]))
                    print('exception occured')
                ax.plot(cr[:,0],cr[:,1],'--',color='#6495ED',zorder=11,linewidth=3)
                
def plot_nuclei_contours2(ResultsRow,data,imgIndex,coordxy,ax):
    for index,row in data['NUCL_PRE'][data['NUCL_PRE']['Img Index'] == imgIndex].iterrows():
        if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(coordxy[0],coordxy[1])):
            if type(imgIndex) != int:
                ax.plot(row['Centroid'][1],row['Centroid'][0],'o',color='r',markersize=7,zorder=5)
            else:
                ax.plot(row['Centroid'][1],row['Centroid'][0],'o',color='#6495ED',markersize=7,zorder=5)
                try:
                    contourr  = row['Contour'][0]
                    cr = contourr.reshape((contourr.shape[0],contourr.shape[2]))
                except:
                    contourr  = row['Contour']
                    cr = contourr.reshape((contourr.shape[0],contourr.shape[2]))
                    print('exception occured')
                ax.plot(cr[:,0],cr[:,1],'--',color='#6495ED',zorder=11,linewidth=3)
                


    
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




### PLOTTERS
def intensity_plotter(ResultsRow,data,save):
    from matplotlib.colors import LinearSegmentedColormap
    
    # Color map
    colors = [(1, 1, 1), (1, 0, 0)] # first color is white, last is red
    cm     = LinearSegmentedColormap.from_list("Custom", colors, N=300)
    
    x_,y_ = ResultsRow['Mask']
    mask = retrieve_mask(ResultsRow['Mask'],ResultsRow['Image Size'])
    
    ##### FIGURE 1
    # Initialize figure 1
    fig,ax = plt.subplots(figsize=(8,8))
    ax.imshow(mask*retrieve_mask(ResultsRow['Skeleton'],ResultsRow['Image Size']),cmap=cm)
    ax.axis('off')
    
    # Plot Nucleus Centroid and Cytoskeleton Centroid
    ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'.',color='#6495ED',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=15,zorder=8,fillstyle='none')
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8)
    
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'.',color='r',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=15,zorder=8,fillstyle='none',)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=12,zorder=8)
    
    # Plot Nucleus Contour
    ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
    #x_,y_   = np.where((ResultsRow['Mask']*1) != 0)
    #plot_nuclei_contours2(ResultsRow,data,ResultsRow['Img Index'],[x_,y_],ax) 
    
    
    # Set x and y lims and title
    #ax.set_ylim([min(ResultsRow['Offset'][0]),max(ResultsRow['Offset'][0])])
    #ax.set_xlim([min(ResultsRow['Offset'][1]),max(ResultsRow['Offset'][1])])
    ax.set_ylim([min(x_),max(x_)])
    ax.set_xlim([min(y_),max(y_)])
    
    # Scale Bar
    scalebar = ScaleBar(0.1612500,"um",color='k',box_alpha=0,dimension='si-length',location='upper right') 
    ax.add_artist(scalebar)
    
    # Adjust and Show
    if save:
        plt.savefig(folder + str("\\") + str(save) + ".pdf",format='pdf',transparent=True,bbox_inches='tight')
    #fig.show()
    fig.tight_layout(pad=0)
    
    
    ##### FIGURE 2
    # Initialize figure 2
    fig2,ax = plt.subplots(figsize=(8,8))
    intensity = data['CYTO']['Image'][ResultsRow['Img Index']] / np.max(data['CYTO']['Image'][ResultsRow['Img Index']])
    aux = intensity * mask
    aux = aux / np.max(aux)
    #ax.imshow(ResultsRow['Mask']*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']]*aux,cmap=cm)
    ax.imshow(aux,cmap=cm)

    ax.axis('off')
    
    # Plot Nucleus Centroid and Cytoskeleton Centroid
    ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'.',color='#6495ED',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=15,zorder=8,fillstyle='none')
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8,alpha=0.5)
    
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'.',color='r',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=15,zorder=8,fillstyle='none',)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=12,zorder=8,alpha=0.5)
    
    # Plot Nucleus Contour
    ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5,alpha=0.5)
         
    # Set x and y lims and title
    ax.set_ylim([min(x_),max(x_)])
    ax.set_xlim([min(y_),max(y_)])
    #ax.set_title(feat,fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cm), ax=ax, shrink = 0.4)
    cbar.set_label('Pixel Intensity',fontfamily='arial',fontsize=12)
    #cbar.set_ticks(np.linspace(0,90,10,endpoint=True))
    
    # Scale Bar
    scalebar = ScaleBar(0.1612500,"um",color='k',box_alpha=0,dimension='si-length',location='lower right')  
    ax.add_artist(scalebar)
    
    # Adjust and Show
    if save:
        plt.savefig(folder + str("\\") + str(save) + "_deconv.pdf",format='pdf',transparent=True,bbox_inches='tight')
    #fig2.show()
    fig2.tight_layout(pad=0)
    
    return fig,fig2


def line_plotter(ResultsRow,data,feat,cmap,normalize_bounds,colorbar_label,line_data_origin,overlay,save):
    #%matplotlib qt
    from matplotlib.colors import LinearSegmentedColormap
    fig,ax = plt.subplots()
    
    # Plot background
    if overlay == None:
        ax.imshow(np.zeros(ResultsRow['Image Size']),cmap='gray',alpha=0)
    if overlay == 'deconv':
        #ax.imshow(1-ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']],cmap='gray')
        ax.imshow(np.max(ResultsRow['Patch:Deconvoluted Cyto'][1]) - ResultsRow['Patch:Deconvoluted Cyto'][1],cmap='gray',zorder=2)
        
    ax.axis('off')
    ax.axis('equal')
    
    # Get bounds for color map (either 'default' or [0,90], etc)
    if normalize_bounds == 'default':
        normalize_bounds = [0,np.max(ResultsRow[feat])]

    # Plot Nucleus Centroid and Cytoskeleton Centroid
    ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'.',color='#6495ED',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=15,zorder=8,fillstyle='none')
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8)
    
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'.',color='r',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=15,zorder=8,fillstyle='none',)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=12,zorder=8)
    
    # Plot Nucleus Contour
    ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
    #x_,y_   = np.where((ResultsRow['Mask']*1) != 0)
    x_,y_ = ResultsRow['Mask']
    #plot_nuclei_contours2(ResultsRow,data,ResultsRow['Img Index'],[x_,y_],ax) 
        
        
    # Plot segments colored by feature value
    #cmap     = pltc.rainbow_r
    #cmap = pltc.hsv
    for l in range(len(ResultsRow[line_data_origin])):
            # Get line = [p0,p1]. 
            p0, p1 = ResultsRow[line_data_origin][l]
            
            if feat != None:
                colour   = cmap(plt.Normalize(normalize_bounds[0], normalize_bounds[1])(ResultsRow[feat][l]))
                ax.plot((p0[0], p1[0]), (p0[1], p1[1]),linewidth=2,color=colour,alpha=1)
            else:
                cmap = pltc.rainbow_r
                ax.plot((p0[0], p1[0]), (p0[1], p1[1]),linewidth=2,color=cmap(0),alpha=1,zorder=5)
                
    
    
    # Colorbar
    if colorbar_label != None:
        cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(normalize_bounds[0], normalize_bounds[1]), cmap=cmap), ax=ax, shrink = 0.4)
        cbar.set_label(colorbar_label,fontfamily='arial',fontsize=12)
        #cbar.set_ticks(np.linspace(0,30,4,endpoint=True))

    # Scale Bar
    scalebar = ScaleBar(0.1612500,"um",color='k',box_alpha=0,dimension='si-length',location='lower right') 
    ax.add_artist(scalebar)

    # Set x and y lims and title
#     ax.set_ylim([min(ResultsRow['Offset'][0]),max(ResultsRow['Offset'][0])])
#     ax.set_xlim([min(ResultsRow['Offset'][1]),max(ResultsRow['Offset'][1])])
    
    
    
    
    # Calculate bounds
    x_min, x_max = np.min(x_), np.max(x_)
    y_min, y_max = np.min(y_), np.max(y_)

    # Determine the center of the bounding rectangle
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Calculate the required half-length to ensure the area is square
    half_length = max((x_max - x_min), (y_max - y_min)) / 2

    # Define square bounds
    min_window_x = x_center - half_length
    max_window_x = x_center + half_length
    min_window_y = y_center - half_length
    max_window_y = y_center + half_length
    
    ax.set_xlim([min_window_y, max_window_y])
    ax.set_ylim([min_window_x, max_window_x])
    print([max_window_y-min_window_y,max_window_x-min_window_x])
#     ax.set_ylim([min(x_),max(x_)])
#     ax.set_xlim([min(y_),max(y_)])
    #ax.set_title(feat,fontsize=12)
    
    # Adjust and Show
    fig.tight_layout(pad=0)
    
    if save != False:
        #plt.savefig(folder + str("\\") + str(save) + ".pdf",format='pdf',transparent=True,bbox_inches='tight')
        #plt.savefig(folder + str("\\") + str(save) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=800)
        plt.savefig(str(save) + ".pdf",format='pdf',transparent=True,bbox_inches='tight')
        #plt.savefig(str(save) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=800)
    #fig.show()
    
    
    return fig



def graph_plotter(ResultsRow,data,cmap,feat,normalize_bounds,colorbar_label,nodes,main_branch,overlay,scalebar,save):
#     try:
#         #%matplotlib qt
#     except:
#         pass
        
    # Get skeleton
    global ske
    img       = data['CYTO']['Image'][ResultsRow['Img Index']] / np.max(data['CYTO']['Image'][ResultsRow['Img Index']])
    intensity = retrieve_mask(ResultsRow['Mask'],ResultsRow['Image Size']) * img
    #ske       = Skeleton(skeleton_image=(ResultsRow['Mask']*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']]*(intensity/np.max(intensity))).astype(float),spacing=0.1612500) 
    ske       = Skeleton(skeleton_image = (retrieve_mask(ResultsRow['Skeleton'],ResultsRow['Image Size'])*img).astype(float),
                         spacing = ResultsRow['Resolution'][2])
                         
    mask = retrieve_mask(ResultsRow['Mask'],ResultsRow['Image Size'])
    # Initialize figure
    fig,ax = plt.subplots()
    if overlay == None:
        #ax.imshow(np.zeros_like(mask*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']]),cmap='gray',alpha=0)
        ax.imshow(np.zeros(ResultsRow['Image Size']),cmap='gray',alpha=0)
    if overlay == 'deconv':
        ax.imshow(np.max(mask*data['CYTO']['Image'][ResultsRow['Img Index']]) - mask*data['CYTO']['Image'][ResultsRow['Img Index']],cmap='gray',alpha=1)
    ax.axis('off')
    ax.axis('equal')
    
    ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8)
    ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
    x_,y_   = ResultsRow['Mask']
    #plot_nuclei_contours2(ResultsRow,data,ResultsRow['Img Index'],[x_,y_],ax)
    
    # Get feat
    if feat == 'branch-distance':
        feat_list = ske.path_lengths()
    if feat == 'mean-pixel-value':
        feat_list = ske.path_means()
    if feat == 'stdev-pixel-value':
        feat_list = ske.path_stdev()
    if feat == 'euclidean-distance':
        feat_list = summarize(ske,find_main_branch=False)['euclidean-distance']
    if feat == 'tortuosity':
        feat_list = ske.path_lengths() / summarize(ske,find_main_branch=False)['euclidean-distance'] 
    if feat == 'branch-type':
        feat_list = summarize(ske,find_main_branch=False)
    if feat == None:
        feat_list = np.ones((1,ske.n_paths))
#     if feat == 'CNF2D:Branch Orientation' or 'CNF2D:Branch Orientation PCA' or 'CNF2D:Local Average Branch Distance' or 'CNF2D:Mean Filament Thickness' or 'CNF2D:Local Average Bundling Score' or 'CNF2D:Local Average Branch Orientation' or 'CNF2D:Distances to Centroid':
#         feat_list = ResultsRow[feat]
    else:
        feat_list = ResultsRow[feat]
    
    
    # Get bounds for color map
    if normalize_bounds == 'default':
        normalize_bounds = [0,np.max(feat_list)]
    
    # Plot paths
    if feat != 'branch-type': # Draw all segments 1 by 1
        for e in range(ske.n_paths):
            if feat != None:
                try:
                    colour   = cmap(plt.Normalize(normalize_bounds[0], normalize_bounds[1])(feat_list[e]))
                    ax.plot(ske.path_coordinates(e)[:,1],ske.path_coordinates(e)[:,0],'-',linewidth=2,color=colour,alpha=1,zorder=2)  
                except:
                    pass
            else:
                ax.plot(ske.path_coordinates(e)[:,1],ske.path_coordinates(e)[:,0],linewidth=2,color='r',alpha=1,zorder=2)
    
#     if feat == 'tortuosity': # Draw all segments 1 by 1
#         n = 0
#         for e in range(ske.n_paths):
#             if feat != None and len(ske.path_coordinates(e)[:,0]) > 1:
#                 colour   = cmap(plt.Normalize(normalize_bounds[0], normalize_bounds[1])(feat_list[n]))
#                 ax.plot(ske.path_coordinates(e)[:,0],ske.path_coordinates(e)[:,1],linewidth=2,color=colour,alpha=1)  
#                 n = n + 1
#             else:
#                 ax.plot(ske.path_coordinates(e)[:,0],ske.path_coordinates(e)[:,1],linewidth=2,color='k',alpha=1)
    
    
    # Branch type
    if feat == 'branch-type': # Filter by branch-type
        btypes = ['endpoint-to-endpoint','junction-to-endpoint','junction-to-junction','isolated cycle']
        colors = ['#6E7E85','#FFD966','#6FA8DC','#744253']
    
        for b in range(3):
            data = feat_list[feat_list['branch-type'] == b]

            flag = True
            for ind in data.index:
                if flag:
                    ax.plot(ske.path_coordinates(ind)[:,1],ske.path_coordinates(ind)[:,0],linewidth=2,color=colors[b],label=btypes[b])
                    flag = False
                else:
                    ax.plot(ske.path_coordinates(ind)[:,1],ske.path_coordinates(ind)[:,0],linewidth=2,color=colors[b])
                
        leg = ax.legend(framealpha=0,loc=(1.04,0.7),labelcolor='linecolor')
        plt.setp(leg.texts, family='arial')
    
    if main_branch:
        fb    = summarize(ske,find_main_branch=True)
        datab = fb[fb['main'] == True]
        flag = True
        for indb in datab.index:
            if flag:
                    ax.plot(ske.path_coordinates(indb)[:,1],ske.path_coordinates(indb)[:,0],linewidth=3,color='k',label='main branch')
                    flag = False
            else:
                ax.plot(ske.path_coordinates(indb)[:,],ske.path_coordinates(indb)[:,0],linewidth=3,color='k')
        
        leg = ax.legend(framealpha=0,loc=(1.04,0.7),labelcolor='linecolor')
        plt.setp(leg.texts, family='arial')
    
        
    # Plot nodes
    if nodes:
        for e in range(ske.n_paths):
            ax.plot(ske.path_coordinates(e)[0][1],ske.path_coordinates(e)[0][0],'o',markersize=1,color='k')
            ax.plot(ske.path_coordinates(e)[-1][1],ske.path_coordinates(e)[-1][0],'o',markersize=1,color='k')

    
    # Set x and y lims and title
    #ax.set_xlim([min(ResultsRow['Offset'][1]),max(ResultsRow['Offset'][1])])
    #ax.set_ylim([min(ResultsRow['Offset'][0]),max(ResultsRow['Offset'][0])])
    
    ax.set_ylim([min(x_),max(x_)])
    ax.set_xlim([min(y_),max(y_)])
    #ax.set_title(feat,fontsize=12)
    
    # Colorbar
    if feat != None and feat !='branch-type':
        cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(normalize_bounds[0], normalize_bounds[1]), cmap=cmap), ax=ax, shrink = 0.4)
        cbar.set_label(colorbar_label,fontfamily='arial',fontsize=12)
        #cbar.set_ticks(np.linspace(0,90,10,endpoint=True))

    # Scale Bar
    if scalebar == True:
        scalebar = ScaleBar(0.1612500,"um",color='k',box_alpha=0,dimension='si-length') 
        ax.add_artist(scalebar)
    
    # save and show
    if save != False:
        plt.savefig(folder + str("\\") + str(save) + ".pdf",format='pdf',transparent=True,bbox_inches='tight')
    #fig.show()
    fig.tight_layout(pad=0)
    
    return fig



# import statannot
# def plot_barplot1(data):
#     cols = list(data.columns[1:])
#     #cols     = [x for x in data.columns if x.startswith("LSF2D") or x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]
#     #feats_1D = [x for x in df.columns if x.startswith("LSF2D")]
#     #feats    = [x for x in df.columns if x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]
    
#     # 4
#     colors   = ["#2ECC71","#FFA500","#E74C3C","#BC544B"]
#     labels   = ['WT','NP','P1','P2']
#     pairs    = [ (('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394')), (('Dup41_46', 'Del38_46')), (('Dup41_46', 'Mut394')),]  

#     # 6
#     #     colors   = ["#2ECC71","#DECF77","#5AB7BD","#FFA500","#E74C3C","#BC544B",]
#     #     labels   = ['WT','No transfection','Mock','Dup41_46','Del38_46','Mut394']
#     #     pairs    = [(('WT', 'No transfection')),(('WT', 'Mock')),(('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394'))] 

    

#     for f in cols:
#         try:
#             data[f] = data[f].astype(float)
#             print('floated')
#         except: 
#             continue

        
#         print(data.groupby(['Label']).describe()[[(f,'mean'),(f,'std')]])

#         fig,ax = plt.subplots()
#         sns.set_theme(style="white")
#         # 4
#         sns.barplot(x="Label", y=f, data=data,order=['WT','Dup41_46','Del38_46','Mut394'],capsize=.1,errorbar=('ci', 95),edgecolor=colors,fill=False,linewidth=2)
        

        
#         #cis = [container.get_yerr()[1]/2 for container in containers]
        
#         ax.set_xticks(ax.get_xticks(),labels,font='arial',color='k')
#         ax.set_yticks(ax.get_yticks(),font='arial',color='k')
        



#         patches = ax.patches
#         lines_per_err = 3

#         for i, line in enumerate(ax.get_lines()):
#             newcolor = patches[i // lines_per_err].get_edgecolor()
#             line.set_color(newcolor)

#         ax.set_xlabel(None)
#         plt.rcParams['axes.linewidth'] = 0.75  
#         statannot.add_stat_annotation(plt.gca(),data=data,x="Label",y=f, order=['WT','Dup41_46','Del38_46','Mut394'],box_pairs=pairs,test='t-test_ind',text_format="star",loc="outside",fontsize='small',comparisons_correction=None,color='k',linewidth=0.75)

#         sns.despine(left=True)
#         plt.grid(alpha=0.2,axis='y')




#         plt.show()
        
    
    
def plot_barplot_paper(data,feature=None):
    import statannot
    # SHOULD BE CHANGED
    if type(feature) != str:
        cols     = [x for x in data.columns if x.startswith("LSF2D") or x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS") or x.startswith("LSF")]
    if type(feature) == list:
        cols     = feature
    if type(feature) == str:
        cols = [feature]
    #feats_1D = [x for x in df.columns if x.startswith("LSF2D")]
    #feats    = [x for x in df.columns if x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]

    # 4
    colors   = ["#2ECC71","#E74C3C"]
    labels   = ['WT','M']
    pairs    = [ (('WT', 'Del38_46'))]  

    # 6
    #     colors   = ["#2ECC71","#DECF77","#5AB7BD","#FFA500","#E74C3C","#BC544B",]
    #     labels   = ['WT','No transfection','Mock','Dup41_46','Del38_46','Mut394']
    #     pairs    = [(('WT', 'No transfection')),(('WT', 'Mock')),(('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394'))] 
    

    
    for f in cols:
        
        try:
            data[f] = data[f].astype(float)
        except: 
            continue

        #print(data.groupby(['Label']).describe()[[(f,'mean'),(f,'std')]])
        # ---
        global grouped_data
        grouped_data = data.groupby(['Label'])[f]

        # Calculate mean and standard error
        mean = grouped_data.mean()
        std = grouped_data.std()
        n = grouped_data.count()
        stderr = std / np.sqrt(n)

        # Create a new DataFrame with the results
        result = pd.DataFrame({'Mean': mean, 'Std': std, 'Std_Err': stderr})
        print(result)

        # Figure
        fig,ax = plt.subplots(figsize=(4,4))
        ax.spines['bottom'].set_zorder(20)
        ax.spines['left'].set_zorder(20)

        
        # BEFORE
#         #sns.barplot(x="Label", y=f, data=data,order=['WT','Del38_46'],capsize=.1,errorbar=('ci', 95),edgecolor='k',fill=True,palette=colors,linewidth=2)
#         sns.barplot(x="Label", y=f, data=data,order=['WT','Del38_46'],capsize=.1,errorbar='se',edgecolor=colors,fill=False,palette=colors,linewidth=3)
#         #ax.yaxis.grid(False)
        # -------
        
        # AFTER
        # Bar plot with errorbar
        sns.barplot(x="Label", y=f, data=data,order=['WT','Del38_46'],capsize=.1,errorbar='se',edgecolor=colors,fill=False,palette=colors,linewidth=3,zorder=5)
        
        # Bar plot without errorbar (to cover the bottom errorbar caps)
        sns.barplot(x="Label", y=f, data=data,order=['WT','Del38_46'],fill=True,palette=['w','w'],edgecolor=colors,linewidth=3,zorder=10,errorbar=None)
        
        # Bar plot to hide bottom edge
        #sns.barplot(x="Label", y=f, data=data,order=['WT','Del38_46'],fill=True,palette=['w','w'],edgecolor=colors,linewidth=3,zorder=12,errorbar=None)
        
        
        # Configure x and y ticks
        ax.set_xticks(ax.get_xticks(),labels,font='arial',color='k',fontsize=14)
        ax.set_yticks(ax.get_yticks(),font='arial',color='k',fontsize=18,visible=False)

        # Color errorbars
        patches = ax.patches
        lines_per_err = 3

        for i, line in enumerate(ax.get_lines()):
            newcolor = patches[i // lines_per_err].get_edgecolor()
            line.set_color(newcolor)

            
        ax.set_xlabel(None)
        plt.rcParams['axes.linewidth'] = 2  
        ax.tick_params(left=True, bottom=False)
        
        # Statistic Test
        statannot.add_stat_annotation(plt.gca(),data=data,x="Label",y=f, order=['WT','Del38_46'],box_pairs=pairs,test='t-test_ind',text_format="star",loc="outside",fontsize='large',comparisons_correction=None,color='k',linewidth=2)

        sns.despine(right=True)
        ax.spines['left'].set_linewidth(2)
        

        # plt.savefig(folder + "//Barplots_Colors_October//" + str(f.split(':')[0]) + "-"  + str(f.split(':')[1]) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=500)
        try:
            #abc = []
            plt.savefig(folder + "//Barplots_Colors_October//" + str(f.split(':')[0]) + "-"  + str(f.split(':')[1]) + ".pdf",format='pdf',transparent=True,bbox_inches='tight')
        except:
            pass
        
        #plt.show()
        fig.tight_layout(pad=0)
        
        
        return fig
        

def plot_barplot_soraia(ResultsDF):
    import itertools
    import statannot
    

    cols     = [x for x in ResultsDF.columns if x == 'Label' or x.startswith("LSF1D") or x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("CNF1D") or x.startswith("OTHERS")]
    data     = ResultsDF[cols]
    #feats_1D = [x for x in df.columns if x.startswith("LSF2D")]
    #feats    = [x for x in df.columns if x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]

    # 
    labels   = np.unique(ResultsDF['Label'])
    colors   = provide_colors(labels)
    pairs = [((label1, label2)) for label1, label2 in itertools.combinations(labels, 2)]
    
    for f in cols:
        if f == 'Label':
            continue
            
        try:
            data[f] = data[f].astype(float)
        except: 
            print('EXCLUDING: ' + str(f))
            continue

        #print(ResultsDF.groupby(['Label']).describe()[[(f,'mean'),(f,'std')]])



        fig,ax = plt.subplots()
        sns.set_theme(style="white")
        # 4
        sns.barplot(x="Label", y=f, data=data,order=labels,capsize=.1,errorbar=('ci', 95),edgecolor=colors,fill=False,linewidth=2)
        

        
        #cis = [container.get_yerr()[1]/2 for container in containers]
        
        ax.set_xticks(ax.get_xticks(),labels,font='arial',color='k')
        ax.set_yticks(ax.get_yticks(),font='arial',color='k')
        



        patches = ax.patches
        lines_per_err = 3

        for i, line in enumerate(ax.get_lines()):
            newcolor = patches[i // lines_per_err].get_edgecolor()
            line.set_color(newcolor)

        ax.set_xlabel(None)
        plt.rcParams['axes.linewidth'] = 0.75  
        statannot.add_stat_annotation(plt.gca(),data=data,x="Label",y=f, order=labels,box_pairs=pairs,test='t-test_ind',text_format="star",loc="outside",fontsize='small',comparisons_correction=None,color='k',linewidth=0.75)

        sns.despine(left=True)
        plt.grid(alpha=0.2,axis='y')




        plt.show()