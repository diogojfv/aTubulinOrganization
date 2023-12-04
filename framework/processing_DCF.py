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

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as pltc
import matplotlib.colors as colors

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
#from framework.PreProcessingCYTO import cytoskeleton_preprocessing, df_cytoskeleton_preprocessing
#from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation

#from framework.visualization import truncate_colormap, plot_hist, plot_pie
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC


def wavelets(ResultsRow,wavelet,level):
    import pywt
    
    patch = ResultsRow['Patches'][1]
    coeffs = pywt.wavedec2(patch, wavelet, level=level)

    EL  = np.sum(np.square(coeffs[0]))     # Energy of the low-frequency channel
    EH1 = np.sum(np.square(coeffs[1][0]))  # Energy of the First high-frequency channel
    EH2 = np.sum(np.square(coeffs[1][1]))  # Energy of the Second high-frequency channel
    EH3 = np.sum(np.square(coeffs[1][2]))  # Energy of the Third high-frequency channel

    return EL,EH1,EH2,EH3

# ResultsDF[["DCF:Wavelets - EL", "DCF:Wavelets - EH1", "DCF:Wavelets - EH2", "DCF:Wavelets - EH3"]] = ResultsDF.apply(lambda row: wavelets(row, 'haar', 3), axis=1, result_type='expand')



def est_area(ResultsRow):
    from skimage.filters import apply_hysteresis_threshold
    
    img = data['CYTO']['Image'][ResultsRow['Img Index']] * ResultsRow['Mask']
    res = apply_hysteresis_threshold(img,threshold_otsu(img)*0.6,threshold_otsu(img))
    
    #img_nucl_ = ResultsRow['Mask'] * (data['NUCL']['Image'][ResultsRow['Img Index']] / np.max(data['NUCL']['Image'][ResultsRow['Img Index']]))
    #img_nucl  = apply_hysteresis_threshold(img_nucl_,threshold_otsu(img_nucl_)*0.6,threshold_otsu(img_nucl_))
    # Find centroid:
    x_,y_   = np.where((ResultsRow['Mask']*1) != 0)
    imgIndex = ResultsRow['Img Index']
    
    for index,row in CentroidsDF[imgIndex].iterrows():
        if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(x_,y_)):
            centroid = (row['Centroid'][1],row['Centroid'][0])
            contour = row['Contour']
            idx = index
            break
    try:
        centroid
    except:
        centroid = (0,0)
        contour = CentroidsDF[imgIndex].loc[0]['Contour']
        idx = 0
        print(index,'centroid not found. set to (0,0)')
    

    #img_nucl_ = ResultsRow['Mask'] * (data['NUCL']['Image'][ResultsRow['Img Index']] / np.max(data['NUCL']['Image'][ResultsRow['Img Index']]))
    #img_nucl  = apply_hysteresis_threshold(img_nucl_,threshold_otsu(img_nucl_)*0.6,threshold_otsu(img_nucl_))
    img_nucl = cv2.fillPoly(np.zeros((1040,1388)),pts=CentroidsDF[imgIndex].loc[idx]['Contour'],color=(255,255,255))
    
    
    
    
    rprops_cyto,rprops_nucl = regionprops((res!=0)*1,res),regionprops((img_nucl!=0)*1,img_nucl)
    centro_cyto,centro_nucl = rprops_cyto[0].centroid,rprops_nucl[0].centroid
    w_centro_cyto,w_centro_nucl = rprops_cyto[0].weighted_centroid,rprops_nucl[0].weighted_centroid

    return rprops_cyto[0].area*(0.16125**2),rprops_nucl[0].area*(0.16125**2),rprops_nucl[0].area/rprops_cyto[0].area

# ResultsDF[['DCF:Area (scaled)', 'DNF:Area (scaled)', 'DCF:AreaRatio 6']] = ResultsDF.apply(lambda row: est_area(row), axis=1, result_type='expand')


