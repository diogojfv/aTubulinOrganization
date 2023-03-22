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
from framework.Functions import FeaturesFromCentroid, cv2toski,pylsdtoski,polar_to_cartesian, truncate_colormap, plot_hist, plot_pie, remove_not1D, quantitative_analysis,hist_bin,hist_lim,create_separate_DFs,branch,graphAnalysis
from framework.Importing import label_image,init_import
from framework.PreProcessingCYTO import cytoskeleton_preprocessing, df_cytoskeleton_preprocessing
from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation
from framework.Processing import process3Dnuclei,analyze_cell

#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC


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
            

            
            
def intensity_plotter(ResultsRow,TextureDF,save):
    from matplotlib.colors import LinearSegmentedColormap
    
    # Color map
    colors = [(1, 1, 1), (1, 0, 0)] # first color is white, last is red
    cm     = LinearSegmentedColormap.from_list("Custom", colors, N=300)
    
    ##### FIGURE 1
    # Initialize figure 1
    fig,ax = plt.subplots(figsize=(8,8))
    ax.imshow(ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']],cmap=cm)
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
    ax.set_ylim([min(ResultsRow['Patches'][4]),max(ResultsRow['Patches'][4])])
    ax.set_xlim([min(ResultsRow['Patches'][5]),max(ResultsRow['Patches'][5])])
    
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
    intensity = DeconvDF['Image'][ResultsRow['Img Index']] / np.max(DeconvDF['Image'][ResultsRow['Img Index']])
    aux = intensity * ResultsRow['Mask']
    aux = aux / np.max(aux)
    ax.imshow(ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']]*aux,cmap=cm)

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
    ax.set_ylim([min(ResultsRow['Patches'][4]),max(ResultsRow['Patches'][4])])
    ax.set_xlim([min(ResultsRow['Patches'][5]),max(ResultsRow['Patches'][5])])
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
            
            

def line_plotter(ResultsRow,TextureDF,feat,cmap,normalize_bounds,colorbar_label,overlay_sk,save):
    fig,ax = plt.subplots(figsize=(8,8))
    
    # Plot background
    if overlay_sk == False:
        ax.imshow(np.zeros((1040, 1388)),cmap='gray',alpha=0) 
    else:
        ax.imshow(1-ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']],cmap='gray')
    ax.axis('off')
    
    # Get bounds for color map
    if normalize_bounds == 'default':
        normalize_bounds = [0,np.max(ResultsRow[feat])]

    # Plot Nucleus Centroid and Cytoskeleton Centroid
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'.',color='#6495ED',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=15,zorder=8,fillstyle='none')
    ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8)
    
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'.',color='r',markersize=5,zorder=8)
    #ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=15,zorder=8,fillstyle='none',)
    ax.plot(ResultsRow['Cytoskeleton Centroid'][1],ResultsRow['Cytoskeleton Centroid'][0],'o',color='r',markersize=12,zorder=8)
    
    # Plot Nucleus Contour
    ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
                
    # Plot segments colored by feature value
    for l in range(len(ResultsRow['Lines'])):
            # Get line = [p0,p1]. 
            p0, p1 = ResultsRow['Lines'][l]
            colour   = cmap(plt.Normalize(normalize_bounds[0], normalize_bounds[1])(ResultsRow[feat][l]))
            ax.plot((p0[0], p1[0]), (p0[1], p1[1]),linewidth=2,color=colour,alpha=1)

    # Set x and y lims and title
    ax.set_ylim([min(ResultsRow['Patches'][4]),max(ResultsRow['Patches'][4])])
    ax.set_xlim([min(ResultsRow['Patches'][5]),max(ResultsRow['Patches'][5])])
    ax.set_title(feat,fontsize=12)
    
    # Colorbar
    cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(normalize_bounds[0], normalize_bounds[1]), cmap=cmap), ax=ax, shrink = 0.4)
    cbar.set_label(colorbar_label,fontfamily='arial',fontsize=12)
    #cbar.set_ticks(np.linspace(0,90,10,endpoint=True))
    
    # Scale Bar
    scalebar = ScaleBar(1,"px",color='k',box_alpha=0,dimension='pixel-length') 
    ax.add_artist(scalebar)
    
    # Adjust and Show
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save:
        plt.savefig(".//output.png",format='png',transparent=True,bbox_inches='tight',dpi=300)
    fig.show()
    
    return print('Done.')

# Color map
# colors = [(0.5, 0.5, 0.5), (1, 0, 0)] # first color is black, last is red
# cm     = LinearSegmentedColormap.from_list("Custom", colors, N=300)

#cmap     = pltc.rainbow
#cm       = truncate_colormap(cmap, 0, 1, 300)


#line_plotter(ResultsRow=ResultsDF.loc[2],TextureDF=TextureDF,feat='LSF2D:Distances to Centroid',cmap=cm,normalize_bounds='default',colorbar_label='Degrees',overlay_sk=True,save=False)


def graph_plotter(ResultsRow,cmap,feat,normalize_bounds,colorbar_label,nodes,main_branch,save):
    # Get skeleton
    img       = DeconvDF['Image'][ResultsRow['Img Index']] / np.max(DeconvDF['Image'][ResultsRow['Img Index']])
    intensity = ResultsRow['Mask'] * img
    ske       = Skeleton((ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']]*(intensity/np.max(intensity))).astype(float)) 
    
    # Initialize figure
    fig,ax = plt.subplots(figsize=(10,10))
    ax.imshow(np.zeros_like(ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']]),cmap='gray',alpha=0)
    ax.axis('off')
    
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
        feat_list = summarize(ske,find_main_branch=False)['euclidean-distance'] / ske.path_lengths()
    if feat == 'branch-type':
        feat_list = summarize(ske,find_main_branch=False)
    if feat == None:
        feat_list = np.ones((1,ske.n_paths))
    
    # Get bounds for color map
    if normalize_bounds == 'default':
        normalize_bounds = [0,np.max(feat_list)]
    
    # Plot paths
    if feat != 'branch-type': # Draw all segments 1 by 1
        for e in range(ske.n_paths):
            if feat != None:
                colour   = cmap(plt.Normalize(normalize_bounds[0], normalize_bounds[1])(feat_list[e]))
                ax.plot(ske.path_coordinates(e)[:,0],ske.path_coordinates(e)[:,1],linewidth=2,color=colour,alpha=1)     
            else:
                ax.plot(ske.path_coordinates(e)[:,0],ske.path_coordinates(e)[:,1],linewidth=2,color='r',alpha=1)
    
    # Branch type
    if feat == 'branch-type': # Filter by branch-type
        btypes = ['endpoint-to-endpoint','junction-to-endpoint','junction-to-junction','isolated cycle']
        colors = ['#6E7E85','#B7CECE','#BBBAC6','#744253']
    
        for b in range(3):
            data = feat_list[feat_list['branch-type'] == b]

            flag = True
            for ind in data.index:
                if flag:
                    ax.plot(ske.path_coordinates(ind)[:,0],ske.path_coordinates(ind)[:,1],linewidth=2,color=colors[b],label=btypes[b])
                    flag = False
                else:
                    ax.plot(ske.path_coordinates(ind)[:,0],ske.path_coordinates(ind)[:,1],linewidth=2,color=colors[b])
                
        leg = ax.legend(framealpha=0,loc=(1.04,0.7),labelcolor='linecolor')
        plt.setp(leg.texts, family='arial')
    
    if main_branch:
        fb    = summarize(ske,find_main_branch=True)
        datab = fb[fb['main'] == True]
        flag = True
        for indb in datab.index:
            if flag:
                    ax.plot(ske.path_coordinates(indb)[:,0],ske.path_coordinates(indb)[:,1],linewidth=3,color='k',label='main branch')
                    flag = False
            else:
                ax.plot(ske.path_coordinates(indb)[:,0],ske.path_coordinates(indb)[:,1],linewidth=3,color='k')
        
        leg = ax.legend(framealpha=0,loc=(1.04,0.7),labelcolor='linecolor')
        plt.setp(leg.texts, family='arial')
    
        
    # Plot nodes
    if nodes:
        for e in range(ske.n_paths):
            ax.plot(ske.path_coordinates(e)[0][0],ske.path_coordinates(e)[0][1],'o',markersize=1,color='k')
            ax.plot(ske.path_coordinates(e)[-1][0],ske.path_coordinates(e)[-1][1],'o',markersize=1,color='k')

    
    # Set x and y lims and title
    ax.set_ylim([min(ResultsRow['Patches'][5]),max(ResultsRow['Patches'][5])])
    ax.set_xlim([min(ResultsRow['Patches'][4]),max(ResultsRow['Patches'][4])])
    ax.set_title(feat,fontsize=12)
    
    # Colorbar
    if feat != None and feat !='branch-type':
        cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(normalize_bounds[0], normalize_bounds[1]), cmap=cmap), ax=ax, shrink = 0.4)
        cbar.set_label(colorbar_label,fontfamily='arial',fontsize=12)
        #cbar.set_ticks(np.linspace(0,90,10,endpoint=True))

    # Scale Bar
    scalebar = ScaleBar(1,"px",color='k',box_alpha=0,dimension='pixel-length') 
    ax.add_artist(scalebar)
    
    # save and show
    if save:
        plt.savefig(".//output.png",format='png',transparent=True,bbox_inches='tight',dpi=300)
    fig.show()
    
    

cmap     = pltc.rainbow
cm       = truncate_colormap(cmap, 0, 1, 300)

data = graph_plotter(ResultsRow=ResultsDF.loc[0],feat='branch-type',cmap=cm,normalize_bounds='default',colorbar_label='Pixels',nodes=False,main_branch=True, save=True)
#line_plotter(ResultsRow=ResultsDF.loc[2],TextureDF=TextureDF,feat='LSF2D:Distances to Centroid',cmap=cm,normalize_bounds='default',colorbar_label='Degrees',overlay_sk=True,save=False)





def graph_plotter(ResultsRow,cmap,feat,normalize_bounds,colorbar_label,nodes,main_branch,overlay,scalebar,save):
    # Get skeleton
    img       = DeconvDF['Image'][ResultsRow['Img Index']] / np.max(DeconvDF['Image'][ResultsRow['Img Index']])
    intensity = ResultsRow['Mask'] * img
    ske       = Skeleton((ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']]*(intensity/np.max(intensity))).astype(float)) 
    
    # Initialize figure
    fig,ax = plt.subplots(figsize=(10,10))
    if overlay == None:
        ax.imshow(np.zeros_like(ResultsRow['Mask']*TextureDF['Skeleton'][ResultsRow['Img Index']]),cmap='gray',alpha=0)
    if overlay == 'deconv':
        ax.imshow(np.max(ResultsRow['Mask']*DeconvDF['Image'][ResultsRow['Img Index']]) - ResultsRow['Mask']*DeconvDF['Image'][ResultsRow['Img Index']],cmap='gray',alpha=1)
    ax.axis('off')
    
    ax.plot(ResultsRow['Nucleus Centroid'][1],ResultsRow['Nucleus Centroid'][0],'o',color='#6495ED',markersize=12,zorder=8)
    ax.plot(ResultsRow['Nucleus Contour'][:,0],ResultsRow['Nucleus Contour'][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
     
    
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
    ax.set_xlim([min(ResultsRow['Patches'][5]),max(ResultsRow['Patches'][5])])
    ax.set_ylim([min(ResultsRow['Patches'][4]),max(ResultsRow['Patches'][4])])
    #ax.set_title(feat,fontsize=12)
    
    # Colorbar
    if feat != None and feat !='branch-type':
        cbar = fig.colorbar(pltc.ScalarMappable(norm=plt.Normalize(normalize_bounds[0], normalize_bounds[1]), cmap=cmap), ax=ax, shrink = 0.4)
        cbar.set_label(colorbar_label,fontfamily='arial',fontsize=12)
        #cbar.set_ticks(np.linspace(0,90,10,endpoint=True))

    # Scale Bar
    if scalebar == True:
        scalebar = ScaleBar(1,"um",color='k',box_alpha=0,dimension='si-length') 
        ax.add_artist(scalebar)
    
    # save and show
    if save != False:
        plt.savefig(".//" + str(save) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=1000)
    fig.show()
    
    

#cmap     = pltc.rainbow
#cm       = truncate_colormap(cmap, 0, 1, 300)

#print('graph_plotter')
#data = graph_plotter(ResultsRow=ResultsDF.loc[0],feat='branch-type',cmap=cm,normalize_bounds='default',colorbar_label='Pixels',nodes=False,main_branch=True, save=True)
#line_plotter(ResultsRow=ResultsDF.loc[2],TextureDF=TextureDF,feat='LSF2D:Distances to Centroid',cmap=cm,normalize_bounds='default',colorbar_label='Degrees',overlay_sk=True,save=False)

