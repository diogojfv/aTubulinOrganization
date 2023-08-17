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
#from framework.PreProcessingCYTO import cytoskeleton_preprocessing, df_cytoskeleton_preprocessing
from framework.Processing import process3Dnuclei,analyze_cell
#from framework.visualization import truncate_colormap, plot_hist, plot_pie
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC



def excludeborder(vol):
    if len(vol.shape) == 2:
        pxs = list(zip(np.where(vol > 0)[0],np.where(vol > 0)[1]))
        Rx  = 1040
        Ry  = 1388
    if len(vol.shape) == 3:
        pxs = list(zip(np.where(vol > 0)[1],np.where(vol > 0)[2]))

    for x in pxs:
        if x[0] <= 2 or x[1] <= 2 or x[0] >= 1038 or x[1] >= 1386:
            return True
    return False



    
                    
            
            
#             def nuclei_preprocessing(image,otsu_thr,area_thr):
#                 img = copy.deepcopy(image)

#                 # Otsu
#                 thresh_ori = threshold_otsu(img)
#                 binary_ori = img > thresh_ori*0.5
#                 image_     = binary_ori * img

#                 # Get contour
#                 contour, hierarchy = cv2.findContours(image_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#                 # Remove small contours
#                 add_contours = []
#                 for cnt in contour:
#                     if cv2.contourArea(cnt) >= 600:
#                         add_contours.append(cnt)

#                 # Get nuclei mask
#                 fillcontours = np.zeros_like(image)
#                 cv2.fillPoly(fillcontours, pts = add_contours, color=(255,255,255))

#                 # Result with normalized background color and noise removed
#                 res = fillcontours/255 * img

#                 return res






def df_nuclei_preprocessing(NUCL_df,dir_nucldec,dir_masks,algorithm,algorithm_specs,plot,save):
    """
    Perform nuclei preprocessing on a dataframe.
            - ```NUCL_df```: data['NUCL_DECONV'] or data['3D']
            - ```dir_nucldec```: directory for dataset folder
            - ```dir_masks```: directory to save the nuclei masks obtained
            - ```algorithm```: contour
            - ```algorithm_specs```: 
            - ```plot```: whether to plot the results
            - ```save```: save the result
    """
    # Nuclei segmentation
    otsu_count = 0
    for index,row in NUCL_df.iterrows():
        print(">>>>>> SEGMENTATION: Image " + str(row['Name']))
        nuclei_segmentation(row['Image'],row['Name'],dir_nucldec, dir_masks,algorithm,[algorithm_specs[0][otsu_count],algorithm_specs[1]])
        
        if type(algorithm_specs[0]) == list and row['Name'].split('_')[-1] == 'ch00.tif':
            otsu_count +=1

    # Nuclei preprocessing
    for index,row in NUCL_df.iterrows():
        NUCL_PRE = nuclei_preprocessing(row['Image'],row['Name'],dir_masks,plot,save)
        
    return NUCL_PRE