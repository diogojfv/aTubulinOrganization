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

from framework.Functions import cv2toski,pylsdtoski,polar_to_cartesian, remove_not1D, quantitative_analysis,hist_bin,hist_lim,branch,graphAnalysis
from framework.importing import *
from framework.preprocessingCYTO import *
#from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation
from framework.processing import *
from framework.visualization import *
from framework.analysis import plot_barplot
from framework.processing_LSF import *
from framework.processing_DCF import *
from framework.processing_CNF import *
from framework.feature_extraction import *
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC

def get_deconvcyto_patch(ResultsRow,data):
    img  = data['CYTO'].loc[ResultsRow['Img Index']]['Image'] / np.max(data['CYTO'].loc[ResultsRow['Img Index']]['Image'])
    mask = retrieve_mask(ResultsRow['Mask'],ResultsRow['Image Size'])
    
    img_masked = img * mask
    
    # NORMALIZATION
    img_masked = (img_masked-np.min(img_masked))/(np.max(img_masked)-np.min(img_masked))
    #img_masked = img_masked / np.max(img_masked)
    
    patch = img_masked[min(ResultsRow['Mask'][0]):max(ResultsRow['Mask'][0]),min(ResultsRow['Mask'][1]):max(ResultsRow['Mask'][1])]
    
    return img_masked,patch

def get_deconvnucl_patch(ResultsRow,data):
    img  = data['NUCL'].loc[ResultsRow['Img Index']]['Image'] / np.max(data['NUCL'].loc[ResultsRow['Img Index']]['Image'])
    mask = retrieve_mask(ResultsRow['Mask'],ResultsRow['Image Size'])
    
    img_masked = img * mask
    
    # NORMALIZATION
    img_masked = (img_masked-np.min(img_masked))/(np.max(img_masked)-np.min(img_masked))
    #img_masked = img_masked / np.max(img_masked)
    
    patch = img_masked[min(ResultsRow['Mask'][0]):max(ResultsRow['Mask'][0]),min(ResultsRow['Mask'][1]):max(ResultsRow['Mask'][1])]
    
    return img_masked,patch



def df_feature_extractor(ResultsDF,listfeats):
    resDF = pd.DataFrame()
    
    count = 0
    for index,row in ResultsDF.iterrows():
        auxDF = feature_extractor(ResultsRow,listfeats)
        
        resDF = pd.concat([resDF,auxDF],axis=0,ignore_index=True)
        
        count += 1
        print(">>> Progress: " + str(round((count / len(ResultsDF))*100,3)) + "%",count)
        
    return resDF

def feature_extractor(ResultsRow,data,features):
    ### DCFs
    # CYTO
    feats_all                    = processingDCF(img             = get_deconvcyto_patch(ResultsRow,data)[1],
                                                 mask            = (get_deconvcyto_patch(ResultsRow,data)[1] != 0)*1,
                                                 skel            = retrieve_mask(ResultsRow['Skeleton'],ResultsRow['Image Size']),
                                                 listfeats       = features,
                                                 resolution      = ResultsRow['Resolution'],
                                                 original_folder = ResultsRow['Path'])
    feats_labels_, feats_values_ = feats_all.print_features(print_values = False)
    #print(feats_labels_)
    #feats_labels_, feats_values_ = remove_not1D(feats_labels_,feats_values_)
    feats_labels_                = ['DCF:' + ftf for ftf in feats_labels_]
    #print(feats_labels_)
    
    # NUCL
    feats_all_n                      = processingDCF(img             = get_deconvnucl_patch(ResultsRow,data)[1],
                                                     mask            = (get_deconvnucl_patch(ResultsRow,data)[1] != 0)*1,
                                                     skel            = 'None',
                                                     listfeats       = features,
                                                     resolution      = ResultsRow['Resolution'],
                                                     original_folder = ResultsRow['Path'])
    feats_labels_n_, feats_values_n_ = feats_all_n.print_features(print_values = False)
    #print(feats_labels_n_)
    #feats_labels_n_, feats_values_n_ = remove_not1D(feats_labels_n_,feats_values_n_)
    feats_labels_n_                  = ['DNF:' + ftn for ftn in feats_labels_n_]
    #print(feats_labels_n_)
                                                     
    
    ### LSFs
    LSFs = line_segment_features(ResultsRow = ResultsRow,
                                 listfeats  = features)
                                                     
    ### CNFs
    CNFs = cyto_network_features(ResultsRow = ResultsRow,
                                 listfeats  = features)
    
    # Amplify Results Row:
    #auxDF = pd.DataFrame(columns = ResultsRow.columns + list(feats_labels_) + list(feats_labels_n_) + [xç for xç,yç in LSFs] + [xg for xg,yg in CNFs])
    #print([m for m in ResultsRow.index.tolist()])
    #print(ResultsRow.index.tolist())
    #print([m[0][0] for m in ResultsRow.index.tolist()])
    #print([m[0][0] for m in ResultsRow.index.tolist()]+list(feats_labels_) + list(feats_labels_n_))
    #print([m[0][0] for m in ResultsRow.index.tolist()] + list(feats_labels_) + list(feats_labels_n_) + [xç for xç,yç in LSFs] + [xg for xg,yg in CNFs])
    auxDF = pd.DataFrame(columns = [m[0] for m in ResultsRow.index.tolist()] + list(feats_labels_) + list(feats_labels_n_) + [xç for xç,yç in LSFs] + [xg for xg,yg in CNFs])
    
        
    new       = pd.Series(list(ResultsRow.values) + feats_values_ + feats_values_n_ +  [yç for xç,yç in LSFs] + [yg for xg,yg in CNFs], index=auxDF.columns)
                                                     
    auxDF = pd.concat([auxDF,new.to_frame().T],axis=0,ignore_index=True)
    
    return auxDF
     