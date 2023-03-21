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
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, laplace, threshold_yen, rank
from skimage.util import img_as_ubyte
from skimage.morphology import extrema, skeletonize, disk
from skimage.transform import probabilistic_hough_line
#from skimage.draw import disk, circle_perimeter
from scipy.ndimage import gaussian_filter, grey_closing
from scipy.spatial import distance_matrix
from skimage import data, restoration, util
from roipoly import RoiPoly
from matplotlib_scalebar.scalebar import ScaleBar
from biosppy.signals import tools
from biosppy.stats import pearson_correlation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage import filters

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
from skan import Skeleton, summarize,draw
from skan.csr import skeleton_to_csgraph, sholl_analysis,make_degree_image
import scipy as sp
import scipy.sparse
from matplotlib.patches import Circle
from framework.ImageFeatures import ImageFeatures
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC
import tifffile as tiffio

import scipy.misc





def label_image(number):
    if (number >= 8 and number <= 11) or number == 29 or number == 30:
        label = 'WT'
    elif number >= 15 and number <= 20:
        label = 'Mock'
    elif number >= 33 and number <= 38:
        label = 'No transfection'
    elif number >= 39 and number <= 44:
        label = 'Del38_46'
    elif number >= 58 and number <= 66:
        label = 'Dup41_46'
    elif number >= 69 and number <= 74:
        label = 'Mut394'
    else:
        label = None
    return label


def init_import(folder, options):
    res = OrderedDict()
    
    # RGB NUCLEI + TUBULIN IMAGES
    if "RGB" in options: 
        OriginalDF = pd.DataFrame(columns=['Name','Label','Image'])
        for img in os.listdir(folder + "\\RGB"):
            path       = folder + "\\RGB\\" + img
            image      = cv2.imread(path,cv2.IMREAD_COLOR)  # Size: (1040, 1388, 3)
            img_id     = int(img.split('_')[0])+1 # add 1 to keep the same id as deconvoluted imgs
            new        = pd.DataFrame(data={'Name': [img], 'Label': [label_image(img_id)], 'Image': [image]}, index = [img_id])
            OriginalDF = pd.concat([OriginalDF, new], axis=0,ignore_index=False)
        res["RGB"] = OriginalDF
        print(">>> [RGB] added.")
        
    # GRAY-SCALE (2D) DECONVOLUTED CYTOSKELETON IMAGES
    if "CYTO_DECONV" in options:
        DeconvDF = pd.DataFrame(columns=['Name','Label','Image'])
        for img in os.listdir(folder + "\\CYTO_DECONV"):
            path     = folder + "\\CYTO_DECONV\\" + img
            if img.split('_')[0] == 'Synthetic':
                image    = cv2.imread(path,cv2.IMREAD_GRAYSCALE)  # Size: (1040,1388)
                img_id   = int(img.split('_')[1])
                new      = pd.DataFrame(data={'Name': [img], 'Label': ['Synthetic'], 'Image': [image]}, index = ['S'+str(img_id)])
                DeconvDF = pd.concat([DeconvDF, new], axis=0,ignore_index=False)
            else:
                image    = cv2.imread(path,-1)  # Size: (1040,1388)
                img_id   = int(img.split('_')[1])
                new      = pd.DataFrame(data={'Name': [img], 'Label': [label_image(img_id)], 'Image': [image]}, index = [img_id])
                DeconvDF = pd.concat([DeconvDF, new], axis=0,ignore_index=False)
        res["CYTO_DECONV"] = DeconvDF
        print(">>> [CYTO_DECONV] added.")
        
    # GRAY-SCALE (2D) DECONVOLUTED NUCLEI IMAGES
    if "NUCL_DECONV" in options:
        NucleiDeconvDF = pd.DataFrame(columns=['Name','Label','Image'])
        for img in os.listdir(folder + "\\NUCL_DECONV"):
            path           = folder + "\\NUCL_DECONV\\" + img
            #image          = nuclei_preprocessing(cv2.imread(path,-1))
            image          = cv2.imread(path,-1)
            img_id         = int(img.split('_')[1])
            new            = pd.DataFrame(data={'Name': [img], 'Label': [label_image(img_id)], 'Image': [image]}, index = [img_id])
            NucleiDeconvDF = pd.concat([NucleiDeconvDF, new], axis=0,ignore_index=False)
        res["NUCL_DECONV"] = NucleiDeconvDF
        print(">>> [NUCL_DECONV] added.")
        
    # 3D GRAY-SCALE SEPARATED RGB CHANNELS
    if "3D" in options:
        TenDF = pd.DataFrame(columns=['Name','Channel','Label','Image'])
        for img in os.listdir(folder + "\\3D"):
            path    = folder + "\\3D\\" + img
            image   = tiffio.imread(path)
            try:
                # NUMERO no nome
                img_id  = int(img.split('_')[0])
            except:
                # MAX_NUMERO no nome
                img_id  = int(img.split('_')[1])
            
            try:
                new     = pd.DataFrame(data={'Name': [img], 'Channel': int(img.split('_')[-1][3]), 'Label': [label_image(img_id)], 'Image': [image]}, index = [img_id])
            except:
                new     = pd.DataFrame(data={'Name': [img], 'Channel': int(img.split('_')[-2][3]), 'Label': [label_image(img_id)], 'Image': [image]}, index = [img_id])
            TenDF   = pd.concat([TenDF, new], axis=0,ignore_index=False)
        res["3D"] = TenDF
        print(">>> [3D] added.")
        
    return res