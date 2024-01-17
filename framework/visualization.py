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
from framework.importing import *
#from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation
from framework.processing import process3Dnuclei,analyze_cell

     
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