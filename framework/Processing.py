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
from framework.Functions import cv2toski,pylsdtoski,polar_to_cartesian, remove_not1D, quantitative_analysis,hist_bin,hist_lim,create_separate_DFs,branch,graphAnalysis
from framework.Importing import label_image,init_import
#from framework.PreProcessingCYTO import cytoskeleton_preprocessing, df_cytoskeleton_preprocessing
#from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation

#from framework.visualization import truncate_colormap, plot_hist, plot_pie
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC




def ROI_centroid(data,img_id,ROIcoords):
    centroid_list = []
    
    if 'NUCL_PRE' in data.keys():
        data_ = data['NUCL_PRE'][data['NUCL_PRE']['Img Index'] == img_id]
        axa = 0
        axb = 1
    if '3D' in data.keys():
        data_ = data['3D'][data['3D']['Img Index'] == img_id]
        axa = 1
        axb = 2
    
    # GET: centroid inside ROI indexes
    for idx,row in data_.iterrows():
        if (round(row['Centroid'][axa]),round(row['Centroid'][axb])) in list(zip(ROIcoords[axa],ROIcoords[axb])):
            centroid_list += [idx]
    if centroid_list == []:  print("Error: No centroids within ROI"); return
    if len(centroid_list) > 1: print("Warning: More than 1 centroid identified within ROI");
    
    # PLOT: first centroid identified and nucleus contour
    centroid = data_.loc[centroid_list[0]]['Centroid']
    
    return centroid_list[0],centroid

def line_segment_features(features,original_img,img_index,mask,patch,xy,centroid,plot):
    # original_img = original skeleton image
    # img_index    = image index
    # mask         = ROI mask of desired cell 
    # patch        = np.array with skeleton p atch
    # xy           = [x_,y_] = [(x1,x2),(y1,y2)]
    # centroids    = Centroids[image index] dataframe
    
    # Create flags
    flagAlpha, flagDtC, flagTri, flagLL, flagTheta, flagAG, flagStdAD, flagLLD, flagStdLLD, flagPAD, flagMCM, flagTAD = False,False,False,False,False,False,False,False,False,False,False,False
    
    # Features
    features1D,features2D = [],[]
    
    # Get offset
    x_ = xy[0]
    y_ = xy[1]
    
    # Get patch
    if mask.any() != None:
        aux__   = original_img * mask
        x_,y_   = np.where(mask != 0)
        patch   = aux__[min(x_):max(x_),min(y_):max(y_)]
        
    # cytoskeleton centroid
    aa = aux__ * 1
    cytocenter = regionprops((aa!=0)*1,aa)[0].centroid
     
    
    # HOUGH ANALYSIS
    lsd   = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV,2.5,0.001,0,90,-200,0.5,2048)
    lines = cv2toski(lsd.detect((patch * 255).astype(np.uint8))[0])
    lines = [((round(min(y_) + l[0][0],3),round(min(x_) + l[0][1],3)),(round(min(y_) + l[1][0],3),round(min(x_) + l[1][1],3))) for l in lines] # Recenter (Fix offset)
    
    # Number of Lines
    if 'LSF1D:Number of Lines' in features:
        N = len(lines)
        features1D += [('LSF1D:Number of Lines',N)]
    
    # Distance matrix features
    median_points = [((line[0][0] + line[1][0])/2,(line[0][1] + line[1][1])/2) for line in lines]
    d             = distance_matrix(median_points,median_points); np.fill_diagonal(d,np.max(d));
    d_0           = distance_matrix(median_points,median_points); np.fill_diagonal(d_0,0);
    
    # Intracluster metrics
    if 'LSF1D:Complete Diameter Distance' in features:
        max_d          = np.max(d)
        features1D    += [('LSF1D:Complete Diameter Distance',max_d)]
    
    if 'LSF1D:Average Diameter Distance' in features:
        avg_diam_dist  = np.sum(d_0) / (len(lines)*(len(lines) - 1))
        features1D    += [('LSF1D:Average Diameter Distance',avg_diam_dist)]
    
    
    center         = (np.mean(np.array(median_points)[:,0]), np.mean(np.array(median_points)[:,1]))
    cent_diam_dist = 2*np.sum([np.linalg.norm(np.array(m)-np.array(center)) for m in median_points]) / len(lines)
    
    # Raise flags
    for f in features:
        if f == 'LSF2D:Alpha':
            flagAlpha   = True
            angles      = []
        if f == 'LSF2D:Distances to Centroid':
            flagDtC     = True
            dist_med    = []
        if f == 'LSF2D:Triangle Areas':
            flagTri     = True
            triangleA   = []
        if f == 'LSF2D:Line Lengths':
            flagLL      = True
            line_size   = []
        if f == 'LSF2D:Theta':
            flagTheta   = True
            thetas      = []
        if f == 'LSF2D:Angle Difference':
            flagAG      = True
            close_angle = []
        if f == 'LSF2D:Std. Angle Difference':
            flagStdAD   = True
            std_locals  = []
        if f == 'LSF2D:Local Line Distance':
            flagLLD     = True
            prox        = []
        if f == 'LSF2D:Std. Local Line Distance':
            flagStdLLD  = True
            std_dists   = []
        if f == 'LSF2D:PAD':
            flagPAD     = True
            PADs        = []
        if f == 'LSF1D:MCM':
            flagMCM     = True
            thetas_w    = []
            prev_v = np.array([0,0])
            prev_vs = [np.array([0,0])]
        if f == 'LSF1D:TAD':
            flagTAD     = True
            
    

                                
    
    # ----------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    ind = 0
    for line in lines:
        p0, p1 = line

        # Prepare vectors - MANDATORY
        med_point      = ((p0[0] + p1[0])/2,(p0[1] + p1[1])/2)
        center_med_vec = np.array(med_point) - np.array([centroid[1],centroid[0]])
        line_vec       = np.array(p1) - np.array(p0)
        center_p0      = p0 - np.array([centroid[1],centroid[0]])
        center_p1      = p1 - np.array([centroid[1],centroid[0]])
        
        ### Features
        # ALPHA
        if flagAlpha == True:
            angle = np.arccos(np.dot(center_med_vec / np.linalg.norm(center_med_vec), line_vec / np.linalg.norm(line_vec)))*180/np.pi
            if angle > 90: angle = 180 - angle;
            
        if flagTheta == True:
            # THETA
            if p0[0] != p1[0]:
                #theta = np.arctan(abs(p1[1]-p0[1])/abs(p1[0]-p0[0]))*180/np.pi
                theta = np.arctan(line_vec[1]/line_vec[0])*180/np.pi
            else:
                theta = 90
            if not 0 < theta < 180:  theta = 180 + theta;
        
        if flagMCM == True:
            # MAIN VECTOR
            if np.linalg.norm(prev_v + line_vec) > np.linalg.norm(prev_v):
                prev_v = prev_v + line_vec
            elif np.linalg.norm(prev_v - line_vec) > np.linalg.norm(prev_v):
                prev_v = prev_v - line_vec
            prev_vs += [prev_v]
            theta_w = np.linalg.norm(line_vec) * theta * np.pi/180 
            thetas_w += [theta_w]
        
        ### LOCAL FEATURES
        # CLOSEST LINES vs. THIS LINE
        copy_d            = copy.deepcopy(d[ind])
        dists_to_medpoint = copy_d
        
        # get indices of the 5 closest lines
        if flagStdAD  == True:
            closest_angles = []
        if flagStdLLD == True:
            prox_lines     = []
        if flagLLD    == True:
            prox_lines     = []
        if flagPAD    == True:
            thetas_        = []
        if flagTAD    == True:
            mean_angles    = []
        
        if flagStdAD == True or flagStdLLD == True or flagLLD == True or flagPAD == True or flagTAD == True:
            for _ in range(5):
                # calculate angle of the _'th closest line - MANDATORY
                min_val          = np.min(dists_to_medpoint)
                close_line_ind   = np.where(dists_to_medpoint == min_val)[0][0]
                p0_c, p1_c       = lines[close_line_ind] 
                med_point_c      = ((p0_c[0] + p1_c[0])/2,(p0_c[1] + p1_c[1])/2)
                center_med_vec_c = np.array(med_point_c) - np.array([centroid[1],centroid[0]])
                line_vec_c       = np.array(p1_c) - np.array(p0_c)

                # ALPHA

                try:
                    angle_c          = np.arccos(np.dot(center_med_vec_c / np.linalg.norm(center_med_vec_c), line_vec_c / np.linalg.norm(line_vec_c)))*180/np.pi
                except:
                    print('error found in angle_c - arccos')
                    angle_c = 0  
                if angle_c > 90: angle_c = 180 - angle_c;

                # THETA

                if p0_c[0] != p1_c[0]: 
                    theta_c = np.arctan(line_vec_c[1]/line_vec_c[0])*180/np.pi
                else: 
                    theta_c = 90
                if not 0 < theta_c < 180:  theta_c = 180 + theta_c;

                # Add to lists
                if flagStdAD  == True:
                    closest_angles = closest_angles + [abs(theta - theta_c)]
                if flagStdLLD == True or flagLLD == True:
                    prox_lines     = prox_lines + [min_val]
                if flagPAD == True:
                    thetas_        = thetas_ + [theta_c]


                # Next line
                dists_to_medpoint[close_line_ind] = max_d
           
        
        # Add features to list
        if flagAlpha == True:
            angles      += [round(angle,3)]
            
        if flagDtC   == True:
            dist_med    += [round(np.linalg.norm(center_med_vec),3)]
        
        if flagTri    == True:
            triangleA   += [round(abs(0.5*np.cross(center_p0,center_p1)),3)]
        
        if flagLL     == True:
            line_size   += [round(np.linalg.norm(line_vec),3)]
       
        if flagTheta  == True:
            thetas      += [round(theta,3)]
            
        if flagStdAD  == True:
            std_locals  += [round(np.std(closest_angles),3)]
            
        if flagLLD    == True:
            prox        += [round(np.mean(prox_lines),3)]
            
        if flagStdLLD == True:
            std_dists   += [round(np.std(prox_lines),3)]
        
        if flagPAD    == True:
            PADs        += [round(np.sqrt(sum((np.array(thetas_) - np.mean(thetas_))**2) / 5),3)]
        
        if flagTAD    == True:
            mean_angles += [np.mean(thetas_)]  
        
        if flagAG == True:
            close_angle += [round(np.mean(closest_angles),3)]
        
        
        
        # next line
        ind = ind + 1
        
    # SAVE FEATURES
    if flagAlpha == True:
        features2D += [('LSF2D:Angles',angles)]

    if flagDtC   == True:
        features2D += [('LSF2D:Distances to Centroid',dist_med)]

    if flagTri    == True:
        features2D += [('LSF2D:Triangle Areas',triangleA)]

    if flagLL     == True:
        features2D += [('LSF2D:Line Lengths',line_size)]

    if flagTheta  == True:
        features2D += [('LSF2D:Theta',thetas)]

    if flagStdAD  == True:
        features2D += [('LSF2D:Std. Angle Difference',std_locals)]

    if flagLLD    == True:
        features2D += [('LSF2D:Local Line Distance',prox)]

    if flagStdLLD == True:
        features2D += [('LSF2D:Std. Local Line Distance',std_dists)]

    if flagPAD    == True:
        features2D += [('LSF2D:PAD',PADs)]
    
    if flagAG     == True:
        features2D += [('LSF2D:Angle Difference',close_angle)]

    
    # OOP, HI, Main Vector Magnitude, TAD
    if 'LSF1D:OOP' in features:
        oop = OOP(thetas)
        features1D += [('LSF1D:OOP',oop)]
    
    if 'LSF1D:HI' in features:
        hi  = HI(thetas)
        features1D += [('LSF1D:HI',hi)]
        
    if flagMCM == True:
        mcm = np.linalg.norm(prev_v)
        features1D += [('LSF1D:MCM',mcm)]
        
    if 'LSF1D:TAD' in features:
        tad = np.sqrt(sum((np.array(mean_angles) - np.mean(thetas))**2) / N)
        features1D += [('LSF1D:TAD',tad)]
    
    # RADIAL SCORE
    if 'LSF1D:Radial Score' in features:
        gridpoints   = subsample_mask(mask,5)
        mat_scores   = radialscore(lines,gridpoints,x_,y_)
        radialSC     = round(np.max(mat_scores),3)
        radialSC_pos = [np.argwhere(mat_scores == np.max(mat_scores))[0]]
        features2D  += [('LSF2D:Radial Pos',radialSC_pos)]
        features1D  += [('LSF1D:Radial Score',radialSC)]
    

                                                                                         
    return lines, median_points, cytocenter, features2D, features1D


def analyze_cell(rowROI,data,algorithm_cyto,algorithm_nuclei,LSFparams,features):
    # rowROI - dataframe with specific ROI corresponding to a cell
    # data   - data with any key:
                  # - RGB
                  # - CYTO_DECONV with algorithm CYTO_DECONV
                  # - NUCL_DECONV with algorithm NUCL_PRE
                  # - 
                    
    # Useful variables:
    img_id = rowROI['Index']
    mask   = rowROI['ROImask']
    name   = rowROI['Name']
    label  = label_image(img_id)
    
    # R and G channel of the cytoskeleton
    try:
        global orig_cysk
        tmp        = copy.deepcopy(data['RGB']['Image'][img_id])
        tmp[:,:,0] = 0
        orig_cysk  = cv2.cvtColor(tmp,cv2.COLOR_RGB2GRAY)
    except:
        pass
    
    # Skeleton patch, for Hough Analysis
    global x_,y_,patch,aux_
    aux_    = mask * (data['CYTO_PRE']['Skeleton'][img_id] * 1)
    #x_,y_   = np.where(aux_ != 0) #x_,y_   = np.where((mask*1) != 0)
    x_,y_   = np.where((mask*1) != 0)
    patch   = aux_[min(x_):max(x_),min(y_):max(y_)]
    
    # Deconvoluted cytoskeleton patch 
    global x_f,y_f,patch_f,aux_f
    if algorithm_cyto == 'deconvoluted':
        aux_f   = mask * (data['CYTO_DECONV']['Image'][img_id] / np.max(data['CYTO_DECONV']['Image'][img_id]))
#     if algorithm == 'original':
#         aux_f   = mask * (orig_cysk / np.max(orig_cysk))
    x_f,y_f = np.where(aux_f != 0)
    patch_f = aux_f[min(x_f):max(x_f),min(y_f):max(y_f)]
    patch_f_norm = patch_f / np.max(aux_f)
    
    # GET and PLOT centroid
    centroid_id,centroid = ROI_centroid(data,img_id,[x_,y_])
     
    # Deconvoluted nuclei patch
    if algorithm_cyto == 'deconvoluted':
        aux_n = np.zeros_like(data['CYTO_DECONV']['Image'][img_id])
        rownuc = data['NUCL_PRE'].loc[centroid_id]
        
        #a = list(zip(rownuc['Nucleus Mask'][0],rownuc['Nucleus Mask'][1]))
        aux_n[rownuc['Nucleus Mask'][0],rownuc['Nucleus Mask'][1]] = 1
        
        aux_n_ = aux_n * data['NUCL_DECONV']['Image'][img_id] / np.max(data['NUCL_DECONV']['Image'][img_id])
#         aux_n_    = mask * ( / np.max(NucleiDeconvDF['Image'][text_img[1]]))


    try:
        contourr = data['NUCL_PRE'].loc[centroid_id]['Contour'] 
        cr       = contourr.reshape((contourr.shape[0],contourr.shape[2]))
    except:
        contourr = data['NUCL_PRE'].loc[centroid_id]['Contour'][0]
        cr       = contourr.reshape((contourr.shape[0],contourr.shape[2]))
#     patch_n  = aux_n[min(cr[:,1]):max(cr[:,1]),min(cr[:,0]):max(cr[:,0])]
#     patch_n_norm = patch_n / np.max(aux_n)
    patch_n = aux_n_[min(rownuc['Nucleus Mask'][0]):max(rownuc['Nucleus Mask'][0]),min(rownuc['Nucleus Mask'][1]):max(rownuc['Nucleus Mask'][1])]
    patch_n_norm = patch_n_norm = patch_n / np.max(aux_n)
    
#     # PROCESSING: Line Segment Analysis
#     global lines, median_points, features2D, features1D
    lines, median_points, cytocenter, LSF2D, LSF1D = line_segment_features(features,aux_,img_id,mask,patch,(x_,y_),centroid,False)
    

#     # TEXTURAL ANALYSIS
    skel_w_int_full = aux_ * aux_f   
    skel_w_int = skel_w_int_full[min(x_):max(x_),min(y_):max(y_)]
    AAI = getAAI(skel_w_int)
    
    # PROCESSING: **CYTOSKELETONS**
    feats_all                    = ImageFeatures((patch_f_norm *255).astype(np.uint8),data['CYTO_DECONV'].loc[img_id]['Path'])
    feats_labels_, feats_values_ = feats_all.print_features(print_values = False)
    feats_labels_, feats_values_ = remove_not1D(feats_labels_,feats_values_)
    feats_labels_                = ['DCF:' + ftf for ftf in feats_labels_]
 
    # PROCESSING: **NUCLEI**
    feats_all_n                      = ImageFeatures((patch_n_norm *255).astype(np.uint8),data['NUCL_DECONV'].loc[img_id]['Path'])
    feats_labels_n_, feats_values_n_ = feats_all_n.print_features(print_values = False)
    feats_labels_n_, feats_values_n_ = remove_not1D(feats_labels_n_,feats_values_n_)
    feats_labels_n_                  = ['DNF:' + ftn for ftn in feats_labels_n_]
    
#     # PROCESSING: Graph Analysis
    global int_ske, graph, graph_res, shollhist, cncd, pxlcount
    int_ske         = ((aux_ * 1) * aux_f) / np.max(aux_f) #1040x1388
#     graph,graph_res,shollhist = graphAnalysis(int_ske,[x_,y_],[aux_n / np.max(patch_n),centroid,cr],mask,plot)
    
#     # PROCESSING: Others
#     cncd = Others(aux_f,aux_n,lines)

    # Add to DataFrame
    global ResultsDF,new
    if 'ResultsDF' not in globals():
            ResultsDF = pd.DataFrame(columns = ['Name'] + ['Img Index'] + ['Label'] + ['Mask'] + ['Patches'] + ['Nucleus Contour'] + ['Nucleus Centroid'] + ['Cytoskeleton Centroid'] + ['Lines'] + [xç for xç,yç in LSF2D] + [xe for xe,ye in LSF1D] + ['DCF:AAI'] + list(feats_labels_n_))
    new          = pd.Series([name] + [img_id] + [label] + [mask] + [[patch,patch_f,patch_n,aux_* orig_cysk,x_,y_]] + [cr] + [centroid] + [cytocenter] + [lines] +  [yç for xç,yç in LSF2D] + [ye for xe,ye in LSF1D] + [AAI] + feats_values_n_,index=ResultsDF.columns)
#     ImageLinesDF = ImageLinesDF.append(new,ignore_index=True)
    ResultsDF = pd.concat([ResultsDF,new.to_frame().T],axis=0,ignore_index=True)
         
    
#     # Add to DataFrame
#     global ImageLinesDF,new
#     if 'ResultsDF' not in globals():
#             ImageLinesDF = pd.DataFrame(columns = ['Name'] + ['Img Index'] + ['Label'] + ['Mask'] + ['Patches'] + ['Nucleus Centroid'] + ['Cytoskeleton Centroid'] + ['Nucleus Contour'] + ['Centrossome'] + ['Lines'] + ['Graph'] + ['Sholl Hist'] + [xç for xç,yç in features2D] + [xe for xe,ye in features1D] + ['DCF:AAI'] + ['DCF:Fractal Dim B Skeleton'] + ['DCF:Fractal Dim D Skeleton'] + ['DCF:Fractal Dim Skeleton'] + ['DCF:Fractal Dim Grayscale'] + ['DNF:Nuclei Grayscale Fractal Dim '] + list(feats_labels_n_) + [graph_ft for graph_ft in list(zip(*graph_res))[0]] + [x for x,y in cncd])
#     new          = pd.Series([DeconvDF['Name'][text_img[1]]] + [text_img[1]] + [DeconvDF['Label'][text_img[1]]] + [mask] + [[patch,patch_f,patch_n,aux_* orig_cysk,x_,y_]] + [centroid] + [cytocenter] + [cr] + [radialSC_pos] + [lines] + [[graph]] + [shollhist] + [yç for xç,yç in features2D] + [ye for xe,ye in features1D] + [AAI] + fractal_values_b + fractal_values_d + fdske + fd_deconv + fd_nuc + feats_values_n_ + [graph_ft for graph_ft in list(zip(*graph_res))[1]] + [y for x,y in cncd],index=ImageLinesDF.columns)
#     ImageLinesDF = ImageLinesDF.append(new,ignore_index=True)
        
    return ResultsDF


# # Analyze Cell
# def analyze_cell_old(text_img,mask,hough_params,centroids,OriginalDF,DeconvDF,NucleiDeconvDF,algorithm,plot):
#     # INPUTS:
#     # text_img                             = [skeleton, index,texture w/ intensity]
#     # mask                                 = binary mask
#     # hough_params                         = [thr, line length, line gap]
#     # centroids                            = DataFrame with nuclei ID's, masks, centroids and contours from image
#     # OriginalDF, DeconvDF, NucleiDeconvDF = Datasets
#     # plot                                 = True/False
    
#     global orig_cysk
#     tmp        = copy.deepcopy(OriginalDF['Image'][text_img[1]])
#     tmp[:,:,0] = 0
#     orig_cysk  = cv2.cvtColor(tmp,cv2.COLOR_RGB2GRAY)
    
#     global patch_aux,centroid,cr,patch_n,centroid
    
    
#     # Texture patch for Hough Analysis
#     global x_,y_,patch,aux_
#     aux_    = mask * (text_img[0] * 1)
#     x_,y_   = np.where((mask*1) != 0)
#     patch   = aux_[min(x_):max(x_),min(y_):max(y_)]
    
#     # Deconvoluted cytoskeleton patch 
#     global x_f,y_f,patch_f,aux_f
#     if algorithm == 'deconvoluted':
#         aux_f   = mask * (DeconvDF['Image'][text_img[1]] / np.max(DeconvDF['Image'][text_img[1]]))
#     if algorithm == 'original':
#         aux_f   = mask * (orig_cysk / np.max(orig_cysk))
#     x_f,y_f = np.where(aux_f != 0)
#     patch_f = aux_f[min(x_f):max(x_f),min(y_f):max(y_f)]
     
        
#     # PROCESSING: Line Segment Analysis
#     global lines, median_points, centroid_list, centroid, features2D, features1D
#     lines, median_points, centroid_list, centroid, features2D, features1D = line_segment_features(text_img[0],text_img[1],mask,patch,(x_,y_),centroids,plot)
    

#     # Deconvoluted nuclei patch
#     if algorithm == 'deconvoluted':
#         aux_n    = mask * (NucleiDeconvDF['Image'][text_img[1]] / np.max(NucleiDeconvDF['Image'][text_img[1]]))
#     if algorithm == 'original':
#         aux_n    = mask * (OriginalDF['Image'][text_img[1]][:,:,0] / np.max(OriginalDF['Image'][text_img[1]][:,:,0]))
    
#     try:
#         contourr = centroids.loc[centroid_list[0]]['Contour'] 
#         cr       = contourr.reshape((contourr.shape[0],contourr.shape[2]))
#     except:
#         contourr = centroids.loc[centroid_list[0]]['Contour'][0]
#         cr       = contourr.reshape((contourr.shape[0],contourr.shape[2]))
#     patch_n  = aux_n[min(cr[:,1]):max(cr[:,1]),min(cr[:,0]):max(cr[:,0])]

  
#     # TEXTURAL ANALYSIS
    
#     # AAI
#     #global AAI
#     #AAI = getAAI(patch_f)
#     AAI = 0
    
#     # PROCESSING: **CYTOSKELETONS**
#     b,d              = fractal_dimension_grayscale(patch_f)
#     fractal_values_b = [round(b,3)]
#     fractal_values_d = [round(d,3)]
#     fdske            = [round(fractal_dimension(patch),3)]
        
#     feats_all                    = ImageFeatures((patch_f *255).astype(np.uint8))
#     feats_labels_, feats_values_ = feats_all.print_features(print_values = False)
#     feats_labels_, feats_values_ = remove_not1D(feats_labels_,feats_values_)
#     feats_labels_                = ['DCF:' + ftf for ftf in feats_labels_]
 
#     # PROCESSING: **NUCLEI**
#     b_n,d_n = fractal_dimension_grayscale(patch_n)
#     fractal_values_n_b = [round(b_n)]
#     fractal_values_n_d = [round(d_n)]         
    
#     feats_all_n                      = ImageFeatures((patch_n *255).astype(np.uint8))
#     feats_labels_n_, feats_values_n_ = feats_all_n.print_features(print_values = False)
#     feats_labels_n_, feats_values_n_ = remove_not1D(feats_labels_n_,feats_values_n_)
#     feats_labels_n_                  = ['DNF:' + ftn for ftn in feats_labels_n_]
    
#     # PROCESSING: Graph Analysis
#     int_ske         = ((text_img[0] * 1) * aux_f) / np.max(patch_f) #1040x1388
#     graph,graph_res = graphAnalysis(int_ske,[x_,y_],[aux_n / np.max(patch_n),centroid,cr],mask,plot)
    
#     # PROCESSING: Others
#     cncd = Others(aux_f,aux_n,lines)
    
    
#     # Add to DataFrame
#     global ImageLinesDF,new
#     if 'ImageLinesDF' not in globals():
#             ImageLinesDF = pd.DataFrame(columns = ['Name'] + ['Img Index'] + ['Label'] + ['Mask'] + ['Patches'] + ['Lines'] + ['Graph'] + [xç for xç,yç in features2D] + [xe for xe,ye in features1D] + ['DCF:AAI'] + ['DCF:Fractal Dim B'] + ['DCF:Fractal Dim D'] + ['DCF:Fractal Dim Skeleton'] + list(feats_labels_) + ['DNF:Nuclei Fractal Dim B'] + ['DNF:Nuclei Fractal Dim D'] + list(feats_labels_n_) + [graph_ft for graph_ft in list(zip(*graph_res))[0]] + [x for x,y in cncd])
#     new          = pd.Series([DeconvDF['Name'][text_img[1]]] + [text_img[1]] + [DeconvDF['Label'][text_img[1]]] + [mask] + [[patch,patch_f,patch_n]] + [lines] + [[graph]] + [yç for xç,yç in features2D] + [ye for xe,ye in features1D] + [AAI] + fractal_values_b + fractal_values_d + fdske + feats_values_ + fractal_values_n_b + fractal_values_n_d + feats_values_n_ + [graph_ft for graph_ft in list(zip(*graph_res))[1]] + [y for x,y in cncd],index=ImageLinesDF.columns)
#     ImageLinesDF = ImageLinesDF.append(new,ignore_index=True)
        

#     return ImageLinesDF,mask,patch,x_,y_,patch_f,patch_n




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
    sholl_feats,sholl_hist = sholl(sk,[infocyto[0],infocyto[1]],[mask * infonucl[0],infonucl[1],infonucl[2]],plot)
    
    #G = nx.from_scipy_sparse_matrix(pixel_graph)  ou nx.from_scipy_sparse_matrix(ske.graph) representação N+1??? nx.density
    
    return ske.graph,branch_feats + sholl_feats,sholl_hist

def sholl(img,cyto_info,nuclei_info,plot):
    #img         = skeleton img intensified   #img = skeleton * DeconvDF['Image'][row['Index']]
    # cyto info  = [x_,y_]
    #nuclei info = [nuclei img   , centroid,contour (cr)]
    
    # Get cytoskeleton centroid
    cytoCentroid = regionprops((img!=0)*1,img)[0].centroid #without offset
    
    # Get skeleton object
    ske = Skeleton((img).astype(float))
    
    # Get max radii and radii vector
    maxradii = max(max([np.linalg.norm((nuclei_info[1][0]-min(cyto_info[0]),nuclei_info[1][1]-min(cyto_info[1]))),
                        np.linalg.norm((nuclei_info[1][0]-max(cyto_info[0]),nuclei_info[1][1]-max(cyto_info[1]))),
                        np.linalg.norm((nuclei_info[1][0]-min(cyto_info[0]),nuclei_info[1][1]-max(cyto_info[1]))),
                        np.linalg.norm((nuclei_info[1][0]-max(cyto_info[0]),nuclei_info[1][1]-min(cyto_info[1])))]),
                   max([np.linalg.norm((cytoCentroid[0]-min(cyto_info[0]),cytoCentroid[1]-min(cyto_info[1]))),
                        np.linalg.norm((cytoCentroid[0]-max(cyto_info[0]),cytoCentroid[1]-max(cyto_info[1]))),
                        np.linalg.norm((cytoCentroid[0]-min(cyto_info[0]),cytoCentroid[1]-max(cyto_info[1]))),
                        np.linalg.norm((cytoCentroid[0]-max(cyto_info[0]),cytoCentroid[1]-min(cyto_info[1])))]))
    radii = np.arange(0,maxradii+5,5)

    #global center_Cy,shell_radii_Cy,counts_Cy,center_Nc,shell_radii_Nc,counts_Nc
    center_Cy,shell_radii_Cy,counts_Cy = sholl_analysis(ske, shells=radii,center = (cytoCentroid[0],cytoCentroid[1]))
    center_Nc,shell_radii_Nc,counts_Nc = sholl_analysis(ske, shells=radii,center = nuclei_info[1])

    # Build table
    tableCy     = pd.DataFrame({'radius': shell_radii_Cy, 'crossings': counts_Cy})
    tableNc     = pd.DataFrame({'radius': shell_radii_Nc, 'crossings': counts_Nc})
    
    fts         = tools.signal_stats(list(tableCy['crossings']))._names
    tempCy      = np.array(list(tools.signal_stats(list(tableCy['crossings']))))
    tempNc      = np.array(list(tools.signal_stats(list(tableNc['crossings']))))
    sholl_feats = list(zip(['SKNW:Sholl Crossings Cyto ' + str(f) for f in fts],tempCy)) + list(zip(['SKNW:Sholl Crossings Nuclei ' + str(fn) for fn in fts],tempNc))


    # Plot results
    if plot:
        # Create figure
        fig_sholl, ax_sholl = plt.subplots(figsize=(8, 8)) #draw.overlay_skeleton_2d_class(ske, axes=ax_sholl)
        
        # Plot skeleton
        ax_sholl.imshow(img,cmap='gray')

        # Plot sholl lines, cyto and muclei centroid, nuclei contour. Set lims
        [ax_sholl.add_patch(c) for c in [Circle((center_Cy[1], center_Cy[0]),radius=r, fill=False, edgecolor='r',ls=':') for r in shell_radii_Cy]]
        [ax_sholl.add_patch(n) for n in [Circle((nuclei_info[1][1], nuclei_info[1][0]), radius=r, fill=False, edgecolor='#6495ED',ls=':') for r in shell_radii_Nc]]
        ax_sholl.plot(nuclei_info[1][1],nuclei_info[1][0],'o',color='#6495ED',markersize=8,zorder=8)
        ax_sholl.plot(cytoCentroid[1],cytoCentroid[0],'o',color='r',markersize=8,zorder=8)
        ax_sholl.plot(nuclei_info[2][:,0],nuclei_info[2][:,1],'--',color='#6495ED',zorder=11,linewidth=2.5)
        ax_sholl.set_xlim(min(cyto_info[1]), max(cyto_info[1]))
        ax_sholl.set_ylim(min(cyto_info[0]), max(cyto_info[0]))
        fig_sholl.show()

        # HISTOGRAM
        fig_sholl_hist, ax_sholl_hist = plt.subplots(figsize=(8, 8))
        ax_sholl_hist.plot('radius', 'crossings', data=tableCy,color='r')
        ax_sholl_hist.plot('radius', 'crossings', data=tableNc,color='#6495ED')
        ax_sholl_hist.grid()
        ax_sholl_hist.set_xlabel('radius')
        ax_sholl_hist.set_ylabel('crossings')

        fig_sholl_hist.show()
        
    return sholl_feats,[tableCy,tableNc]
                   
#sholl_feats = sholl(skeleton * DeconvDF['Image'][row['Index']],[x_,y_],[row['ROImask'] * NucleiDeconvDF['Image'][row['Index']],centroid,cr],True)

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

def newtheta(lines):
    thetas = []
    for line in lines:
        p0,p1 = line
        line_vec = np.array(p1) - np.array(p0)

        if p0[0] != p1[0]:
                #theta = np.arctan(abs(p1[1]-p0[1])/abs(p1[0]-p0[0]))*180/np.pi
                theta = np.arctan(line_vec[1]/line_vec[0])*180/np.pi
        else:
            theta = 90

        if not 0 < theta < 180:  theta = 180 + theta;

        thetas += [theta]
    return thetas

def getAAI(patch):
    aga       = np.histogram(255 * (patch/np.max(patch)),bins=256)
    local_min = argrelextrema(aga[0], np.less)[0]
    AAI       = np.sum(aga[0][local_min[0]:local_min[-1]] * aga[1][local_min[0]:local_min[-1]]) / len(np.where(patch != 0)[0])
    
    return AAI

def Others(img_cyto,img_nucl,lines):
    # Cyto-Nuc Centroid Divergence:
    rprops_cyto,rprops_nucl = regionprops((img_cyto!=0)*1,img_cyto),regionprops((img_nucl!=0)*1,img_nucl)
    centro_cyto,centro_nucl = rprops_cyto[0].centroid,rprops_nucl[0].centroid
    w_centro_cyto,w_centro_nucl = rprops_cyto[0].weighted_centroid,rprops_nucl[0].weighted_centroid

    return [('OTHERS:Cytoskeleton-Nuclei Centroid Distance',np.linalg.norm((centro_cyto[0] - centro_nucl[0],centro_cyto[1] - centro_nucl[1]))),
            ('OTHERS:Weighted Cytoskeleton-Nuclei Centroid Distance',np.linalg.norm((w_centro_cyto[0] - w_centro_nucl[0],w_centro_cyto[1] - w_centro_nucl[1]))),
            ('OTHERS:Area Ratio (Cyto vs. Nucl)',rprops_nucl[0].area/rprops_cyto[0].area)]

def HI(angles):
    theta_rad = np.array(angles)*np.pi/180
    bins      = np.linspace(0,np.pi,180,endpoint=True)
    hist      = np.histogram(theta_rad,bins=bins,density=True)
    hist      = (hist[0],hist[1][:-1])
    #hist      = (hist[0], [(hist[1][i] + hist[1][i+1])/2 for i in range(len(hist[1])-1)])

    def get_index_hist1(theta,hist1):
        if theta == 0:
            return 0
        if hist1[-1] <= theta <= np.pi:
            return len(hist1)-1
        
        count = 0
        for i in range(len(hist1)):
            if theta > hist1[i]:
                count = count + 1
            else:
                break

        return count
    
    def mylog(val):
        if val == 0:
            return 0
        else:
            return np.log10(val)
    
    inds  = [get_index_hist1(theta,hist[1]) for theta in theta_rad]
    ent   = [hist[0][i] for i in inds]
    xlogx = [x*mylog(x) for x in ent]
    HI    = 10**(-np.sum(xlogx)) / len(theta_rad)

    return HI
    
def OOP(angles):
    theta_rad = np.array(angles)*np.pi/180

    OrderTensors = []
    for ang in theta_rad:
        re = np.cos(ang)
        im = np.sin(ang)
        OT = np.array([[re*re,re*im],[im*re,im*im]])
        OrderTensors += [2*OT - np.array([[1,0],[0,1]])]

    MOT11 = np.mean([ot[0,0] for ot in OrderTensors])
    MOT12 = np.mean([ot[0,1] for ot in OrderTensors])
    MOT22 = np.mean([ot[1,1] for ot in OrderTensors])
    MOT   = np.array([[MOT11,MOT12],[MOT12,MOT22]])

    from numpy.linalg import eig
    evals,evecs=eig(MOT)

    return np.max(evals)

def subsample_mask(mask,frac):
    x          = np.arange(0, mask.shape[0]-1, frac)
    y          = np.arange(0, mask.shape[1]-1, frac)
    meshpoints = np.dstack(np.meshgrid(x, y)).reshape(-1, 2) 
    grelha = np.zeros_like(mask)

    for point in meshpoints:
        grelha[point[0],point[1]] = 1

    gridpoints = mask * grelha
    
    return gridpoints

def radialscore(lines,gridpoints,x_,y_):
    global x_points,y_points,points,mat_scores,scores,score,h_i_s,angles
    x_points,y_points = np.where(gridpoints == 1)
    points = list(zip(x_points,y_points))
    
    
    mat_scores = np.zeros((gridpoints.shape[0],gridpoints.shape[1]))
    maxscore = -1
    scores = []
    ind = 0
    for p in points:
        angles = []
        ind = ind + 1
        for line in lines:
            p0, p1 = line
            p0     = (min(y_) + p0[0],min(x_) + p0[1])
            p1     = (min(y_) + p1[0],min(x_) + p1[1])

            # Prepare vectors
            med_point      = ((p0[0] + p1[0])/2,(p0[1] + p1[1])/2)
            center_med_vec = np.array(med_point) - np.array([p[0],p[1]])
            line_vec       = np.array(p1) - np.array(p0)
            center_p0      = p0 - np.array([p[0],p[1]])
            center_p1      = p1 - np.array([p[0],p[1]])

            # Statistics
            try:
                angle = np.arccos(np.dot(center_med_vec / np.linalg.norm(center_med_vec), line_vec / np.linalg.norm(line_vec)))*180/np.pi
                
            except:
                print('error in arccos. Set to 0')
                angle = 0
            if angle > 90:
                angle = 180 - angle
                
            angles += [angle]
                
        # Compute score
        h_i_s = np.histogram(angles,bins = np.linspace(0,90,10),density=True)
        score = np.sum(h_i_s[0][:4])
        
        if score > maxscore:
            maxscore = score
            maxpoint = p
            
        scores = scores + [score]
        
    mat_scores[x_points,y_points] = scores
    return mat_scores


# def analyze_cell(text_img,mask,hough_params,centroids,OriginalDF,DeconvDF,NucleiDeconvDF,algorithm,plot):
#     # INPUTS:
#     # text_img                             = [skeleton, index,texture w/ intensity]
#     # mask                                 = binary mask
#     # hough_params                         = [thr, line length, line gap]
#     # centroids                            = DataFrame with nuclei ID's, masks, centroids and contours from image
#     # OriginalDF, DeconvDF, NucleiDeconvDF = Datasets
#     # plot                                 = True/False
    
#     global orig_cysk
#     tmp        = copy.deepcopy(OriginalDF['Image'][text_img[1]])
#     tmp[:,:,0] = 0
#     orig_cysk  = cv2.cvtColor(tmp,cv2.COLOR_RGB2GRAY)
    
#     global patch_aux,centroid,cr,patch_n,centroid
    
    
#     # Texture patch for Hough Analysis
#     global x_,y_,patch,aux_
#     aux_    = mask * (text_img[0] * 1)
#     x_,y_   = np.where((mask*1) != 0)
#     patch   = aux_[min(x_):max(x_),min(y_):max(y_)]
    
#     # Deconvoluted cytoskeleton patch 
#     global x_f,y_f,patch_f,aux_f
#     if algorithm == 'deconvoluted':
#         aux_f   = mask * (DeconvDF['Image'][text_img[1]] / np.max(DeconvDF['Image'][text_img[1]]))
# #     if algorithm == 'original':
# #         aux_f   = mask * (orig_cysk / np.max(orig_cysk))
#     x_f,y_f = np.where(aux_f != 0)
#     patch_f = aux_f[min(x_f):max(x_f),min(y_f):max(y_f)]
#     patch_f_norm = patch_f / np.max(aux_f)
    
#     # GET and PLOT centroid
#     global centroid_list,centroid
#     centroid_list = []
    
#     # GET: centroid inside ROI indexes
#     for idx,row in centroids.iterrows():
#         if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(x_,y_)): 
#             centroid_list += [idx]
#     if centroid_list == []:  print("Error: No centroids within ROI"); return 0,0,0,0,0,0;
#     if len(centroid_list) > 1: print("Warning: More than 1 centroid identified within ROI");
    
#     # PLOT: first centroid identified and nucleus contour
#     centroid = centroids.loc[centroid_list[0]]['Centroid']
    

     
        
#     # PROCESSING: Line Segment Analysis
#     global lines, median_points, features2D, features1D
#     lines, median_points, centroid_list, centroid, cytocenter, radialSC_pos, features2D, features1D = line_segment_features(text_img[0],text_img[1],mask,patch,(x_,y_),centroids,plot)
    

#     # Deconvoluted nuclei patch
#     if algorithm == 'deconvoluted':
#         aux_n    = mask * (NucleiDeconvDF['Image'][text_img[1]] / np.max(NucleiDeconvDF['Image'][text_img[1]]))

#     try:
#         contourr = centroids.loc[centroid_list[0]]['Contour'] 
#         cr       = contourr.reshape((contourr.shape[0],contourr.shape[2]))
#     except:
#         contourr = centroids.loc[centroid_list[0]]['Contour'][0]
#         cr       = contourr.reshape((contourr.shape[0],contourr.shape[2]))
#     patch_n  = aux_n[min(cr[:,1]):max(cr[:,1]),min(cr[:,0]):max(cr[:,0])]
#     patch_n_norm = patch_n / np.max(aux_n)

  
#     # TEXTURAL ANALYSIS
#     skel_w_int_full = aux_ * aux_f   
#     skel_w_int = skel_w_int_full[min(x_):max(x_),min(y_):max(y_)]
#     AAI = getAAI(skel_w_int)
#     #AAI = 0
    
#     # PROCESSING: **CYTOSKELETONS**
#     b,d              = fractal_dimension_grayscale(patch)
#     fractal_values_b = [round(b,3)]
#     fractal_values_d = [round(d,3)]
#     fdske            = [round(fractal_dimension(patch),3)]
#     fd_deconv        = [round(fractal_dimension(patch_f_norm),3)]
        
#     feats_all                    = ImageFeatures((patch_f_norm *255).astype(np.uint8))
#     feats_labels_, feats_values_ = feats_all.print_features(print_values = False)
#     feats_labels_, feats_values_ = remove_not1D(feats_labels_,feats_values_)
#     feats_labels_                = ['DCF:' + ftf for ftf in feats_labels_]
 
#     # PROCESSING: **NUCLEI**
#     b_n,d_n = fractal_dimension_grayscale(patch_n_norm)
#     fractal_values_n_b = [round(b_n)]
#     fractal_values_n_d = [round(d_n)] 
#     fd_nuc  = [fractal_dimension(patch_n_norm)]
    
#     feats_all_n                      = ImageFeatures((patch_n_norm *255).astype(np.uint8))
#     feats_labels_n_, feats_values_n_ = feats_all_n.print_features(print_values = False)
#     feats_labels_n_, feats_values_n_ = remove_not1D(feats_labels_n_,feats_values_n_)
#     feats_labels_n_                  = ['DNF:' + ftn for ftn in feats_labels_n_]
    
#     # PROCESSING: Graph Analysis
#     global int_ske, graph, graph_res, shollhist, cncd, pxlcount
#     int_ske         = ((text_img[0] * 1) * aux_f) / np.max(aux_f) #1040x1388
#     graph,graph_res,shollhist = graphAnalysis(int_ske,[x_,y_],[aux_n / np.max(patch_n),centroid,cr],mask,plot)
    
#     # PROCESSING: Others
#     cncd = Others(aux_f,aux_n,lines)
    
#     # Add to DataFrame
#     global ImageLinesDF,new
#     if 'ResultsDF' not in globals():
#             ImageLinesDF = pd.DataFrame(columns = ['Name'] + ['Img Index'] + ['Label'] + ['Mask'] + ['Patches'] + ['Nucleus Centroid'] + ['Cytoskeleton Centroid'] + ['Nucleus Contour'] + ['Centrossome'] + ['Lines'] + ['Graph'] + ['Sholl Hist'] + [xç for xç,yç in features2D] + [xe for xe,ye in features1D] + ['DCF:AAI'] + ['DCF:Fractal Dim B Skeleton'] + ['DCF:Fractal Dim D Skeleton'] + ['DCF:Fractal Dim Skeleton'] + ['DCF:Fractal Dim Grayscale'] + ['DNF:Nuclei Grayscale Fractal Dim '] + list(feats_labels_n_) + [graph_ft for graph_ft in list(zip(*graph_res))[0]] + [x for x,y in cncd])
#     new          = pd.Series([DeconvDF['Name'][text_img[1]]] + [text_img[1]] + [DeconvDF['Label'][text_img[1]]] + [mask] + [[patch,patch_f,patch_n,aux_* orig_cysk,x_,y_]] + [centroid] + [cytocenter] + [cr] + [radialSC_pos] + [lines] + [[graph]] + [shollhist] + [yç for xç,yç in features2D] + [ye for xe,ye in features1D] + [AAI] + fractal_values_b + fractal_values_d + fdske + fd_deconv + fd_nuc + feats_values_n_ + [graph_ft for graph_ft in list(zip(*graph_res))[1]] + [y for x,y in cncd],index=ImageLinesDF.columns)
#     ImageLinesDF = ImageLinesDF.append(new,ignore_index=True)
        
#     return ImageLinesDF



# def line_segment_features(original_img,img_index,mask,patch,xy,centroids,plot):
#     # original_img = original skeleton image
#     # img_index    = image index
#     # mask         = ROI mask of desired cell 
#     # patch        = np.array with skeleton p atch
#     # xy           = [x_,y_] = [(x1,x2),(y1,y2)]
#     # centroids    = Centroids[image index] dataframe
    
#     # Get offset
#     x_ = xy[0]
#     y_ = xy[1]
    
#     # Get patch
#     if mask.any() != None:
#         aux__   = original_img * mask
#         x_,y_   = np.where(mask != 0)
#         patch   = aux__[min(x_):max(x_),min(y_):max(y_)]
        
 
#     # GET and PLOT centroid
#     centroid_list = []
    
#     # GET: centroid inside ROI indexes
#     for idx,row in centroids.iterrows():
#         if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(x_,y_)): 
#             centroid_list += [idx]
#     if centroid_list == []:  print("Error: No centroids within ROI"); return 0,0,0,0,0,0;
#     if len(centroid_list) > 1: print("Warning: More than 1 centroid identified within ROI");
    
#     # PLOT: first centroid identified and nucleus contour
#     centroid = centroids.loc[centroid_list[0]]['Centroid']
        
#     # cytoskeleton centroid
#     aa = aux__ * 1
#     cytocenter = regionprops((aa!=0)*1,aa)[0].centroid
     
    
#     # HOUGH ANALYSIS
#     lsd   = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV,2.5,0.001,0,90,-200,0.5,2048)
#     lines = cv2toski(lsd.detect((patch * 255).astype(np.uint8))[0])
#     lines = [((round(min(y_) + l[0][0],3),round(min(x_) + l[0][1],3)),(round(min(y_) + l[1][0],3),round(min(x_) + l[1][1],3))) for l in lines] # Recenter (Fix offset)
    
#     # Number of Lines
#     N = len(lines)
    
#     # Distance matrix features
#     median_points = [((line[0][0] + line[1][0])/2,(line[0][1] + line[1][1])/2) for line in lines]
#     d             = distance_matrix(median_points,median_points); np.fill_diagonal(d,np.max(d));
#     d_0           = distance_matrix(median_points,median_points); np.fill_diagonal(d_0,0);
    
#     # Intracluster metrics
#     max_d          = np.max(d)
#     avg_diam_dist  = np.sum(d_0) / (len(lines)*(len(lines) - 1))
#     center         = (np.mean(np.array(median_points)[:,0]), np.mean(np.array(median_points)[:,1]))
#     cent_diam_dist = 2*np.sum([np.linalg.norm(np.array(m)-np.array(center)) for m in median_points]) / len(lines)
    
#     # HOUGH ANALYSIS
#     angles,dist_med,triangleA,line_size,thetas,close_angle,std_locals,std_dists,prox,PADs,thetas_w = [],[],[],[],[],[],[],[],[],[],[]
    

#     prev_v = np.array([0,0])
#     prev_vs = [np.array([0,0])]
#     ind = 0
#     for line in lines:
#         p0, p1 = line

#         # Prepare vectors
#         med_point      = ((p0[0] + p1[0])/2,(p0[1] + p1[1])/2)
#         center_med_vec = np.array(med_point) - np.array([centroid[1],centroid[0]])
#         line_vec       = np.array(p1) - np.array(p0)
#         center_p0      = p0 - np.array([centroid[1],centroid[0]])
#         center_p1      = p1 - np.array([centroid[1],centroid[0]])
        
#         ### Features
#         # ALPHA
#         angle = np.arccos(np.dot(center_med_vec / np.linalg.norm(center_med_vec), line_vec / np.linalg.norm(line_vec)))*180/np.pi
#         if angle > 90: angle = 180 - angle;
            
#         # THETA
#         if p0[0] != p1[0]:
#             #theta = np.arctan(abs(p1[1]-p0[1])/abs(p1[0]-p0[0]))*180/np.pi
#             theta = np.arctan(line_vec[1]/line_vec[0])*180/np.pi
#         else:
#             theta = 90
#         if not 0 < theta < 180:  theta = 180 + theta;
        
#         # MAIN VECTOR
#         if np.linalg.norm(prev_v + line_vec) > np.linalg.norm(prev_v):
#             prev_v = prev_v + line_vec
#         elif np.linalg.norm(prev_v - line_vec) > np.linalg.norm(prev_v):
#             prev_v = prev_v - line_vec
#         prev_vs += [prev_v]
#         theta_w = np.linalg.norm(line_vec) * theta * np.pi/180 
#         thetas_w += [theta_w]
        
#         ### LOCAL FEATURES
#         # CLOSEST LINES vs. THIS LINE
#         copy_d            = copy.deepcopy(d[ind])
#         dists_to_medpoint = copy_d
        
#         # get indices of the 5 closest lines
#         closest_angles = []
#         prox_lines     = []
#         thetas_        = []
#         mean_angles    = []
        
#         for _ in range(5):
#             # calculate angle of the _'th closest line
#             min_val          = np.min(dists_to_medpoint)
#             close_line_ind   = np.where(dists_to_medpoint == min_val)[0][0]
#             p0_c, p1_c       = lines[close_line_ind] 
#             med_point_c      = ((p0_c[0] + p1_c[0])/2,(p0_c[1] + p1_c[1])/2)
#             center_med_vec_c = np.array(med_point_c) - np.array([centroid[1],centroid[0]])
#             line_vec_c       = np.array(p1_c) - np.array(p0_c)
            
#             # ALPHA
#             try:
#                 angle_c          = np.arccos(np.dot(center_med_vec_c / np.linalg.norm(center_med_vec_c), line_vec_c / np.linalg.norm(line_vec_c)))*180/np.pi
#             except:
#                 print('error found in angle_c - arccos')
#                 angle_c = 0  
#             if angle_c > 90: angle_c = 180 - angle_c;
            
#             # THETA
#             if p0_c[0] != p1_c[0]: 
#                 theta_c = np.arctan(line_vec_c[1]/line_vec_c[0])*180/np.pi
#             else: 
#                 theta_c = 90
#             if not 0 < theta_c < 180:  theta_c = 180 + theta_c;
        
#             # Add to lists
#             closest_angles = closest_angles + [abs(theta - theta_c)]
#             prox_lines     = prox_lines + [min_val]
#             thetas_        = thetas_ + [theta_c]
            
#             # Next line
#             dists_to_medpoint[close_line_ind] = max_d
           
#         mean_angles += [np.mean(thetas_)]  
#         # ---
        
#         # Add features to list
#         angles      += [round(angle,3)]
#         dist_med    += [round(np.linalg.norm(center_med_vec),3)]
#         triangleA   += [round(abs(0.5*np.cross(center_p0,center_p1)),3)]
#         line_size   += [round(np.linalg.norm(line_vec),3)]
#         thetas      += [round(theta,3)]
#         close_angle += [round(np.mean(closest_angles),3)]
#         std_locals  += [round(np.std(closest_angles),3)]
#         prox        += [round(np.mean(prox_lines),3)]
#         std_dists   += [round(np.std(prox_lines),3)]
#         PADs        += [round(np.sqrt(sum((np.array(thetas_) - np.mean(thetas_))**2) / 5),3)]
        
#         # next line
#         ind = ind + 1
    
#     # OOP, HI, Main Vector Magnitude, TAD
#     oop = OOP(thetas)
#     hi  = HI(thetas)
#     mcm = np.linalg.norm(prev_v)
#     tad = np.sqrt(sum((np.array(mean_angles) - np.mean(thetas))**2) / N)
    
#     # RADIAL SCORE
#     gridpoints   = subsample_mask(mask,5)
#     mat_scores   = radialscore(lines,gridpoints,x_,y_)
#     radialSC     = round(np.max(mat_scores),3)
#     radialSC_pos = [np.argwhere(mat_scores == np.max(mat_scores))[0]]
    
#     # SAVE FEATURES
#     features2D = [('LSF2D:Angles',angles),
#                   ('LSF2D:Distances to Centroid',dist_med),
#                   ('LSF2D:Triangle Areas',triangleA),
#                   ('LSF2D:Line Lengths',line_size),
#                   ('LSF2D:Theta',thetas),
#                   ('LSF2D:Angle Difference',close_angle),
#                   ('LSF2D:Std. Angle Difference',std_locals),
#                   ('LSF2D:Local Line Distance',prox),
#                   ('LSF2D:Std. Local Line Distance',std_dists),
#                   ('LSF2D:PAD',PADs)]
    
#     features1D = [('LSF1D:Number of Lines',N),
#                   ('LSF1D:Radial Score',radialSC),
#                   ('LSF1D:Complete Diameter Distance',max_d),
#                   ('LSF1D:Average Diameter Distance',avg_diam_dist),
#                   ('LSF1D:TAD',tad),
#                   ('LSF1D:OOP',oop),
#                   ('LSF1D:HI',hi),
#                   ('LSF1D:MCM',mcm)]
                                                                                         
#     return lines, median_points, centroid_list, centroid, cytocenter, radialSC_pos, features2D, features1D

def df_analyze_cell(data,ROIsDF):
    # Get labels and remove synthetic images
    labels  = list(np.unique(ROIsDF['Label']))
    try:
        labels.remove('Synthetic')
    except:
        pass

    count = 0
    for index,row in ROIsDF.iterrows():
        # Analyse cell from ROI
        #ResultsDF = analyze_cell([skeleton, row['Index']],row['ROImask'],[2,2.5,1],Centroids[row['Index']],OriginalDF,DeconvDF,NucleiDeconvDF,'deconvoluted',False)
        ResultsDF = analyze_cell(row,data,algorithm_cyto,algorithm_nuclei,LSFparams,features,plot)
        
        # Print progress
        print(">>> Progress: " + round((count / len(ROIs))*100) + "%")
        count += 1

    return ResultsDF

def process3Dnuclei(dir_masks):
    from skimage.morphology import ball, binary_dilation
    from tifffile import imread, imwrite
    from stardist.models import StarDist3D
    from csbdeep.utils import Path, normalize
    from stardist.geometry import dist_to_coord3D
    import napari

    for img in os.listdir(dir_masks):
        # Get image
        path     = dir_masks + '/' + img
        print(path)
        nuc_mask = imread(path)
        idx = int(path.split('/')[1].split('_')[0])
        if idx == 16 or idx == 18 or idx == 20 or idx == 34 or idx == 36 or idx == 38:
            continue
        # Initialize DataFrame to put the Centroids in:
        #isolated_nucleus = pd.DataFrame(columns=['ID','Mask With ID','Centroid','Contour'])

        global aux,bin_nuclei,imgg
        # Obtain isolated nucleus
        org = data['3D'].loc[int(idx)].head(1)['Image'].values[0]
        maxorg = np.max(org)
        orgnorm = org / maxorg
        
        nuid = 0
        for nucleus in np.unique(nuc_mask):
            if nucleus != 0: #Not background
                
                aux  = nuc_mask == nucleus
                mask_w_id = aux*nuc_mask
                bin_nuclei = np.where(mask_w_id>0.5, 1, 0)
                bin_nuclei = binary_dilation(bin_nuclei,footprint=ball(3))
                
                imgg = orgnorm * bin_nuclei
                
                xp,yp,zp = np.where(bin_nuclei>0)
                pixels   = [xp,yp,zp]

                #print(bin_nuclei.shape,np.unique(bin_nuclei))
                global feats_all_n,feats_values_n_
                feats_all_n                      = ImageFeatures((imgg *255).astype(np.uint8),os.getcwd() + "\\Datasets\\Set 3D\\3D\\" + img)
                feats_labels_n_, feats_values_n_ = feats_all_n.print_features(print_values = False)
                feats_labels_n_, feats_values_n_ = remove_not1D(feats_labels_n_,feats_values_n_)
                feats_labels_n_                  = ['DNF:' + ftn for ftn in feats_labels_n_]
                
                
                
                if 'nucDF' not in globals():
                    global nucDF
                    nucDF = pd.DataFrame(columns = ['Name'] + ['Img Index'] + ['Label'] + ['ID'] + ['Nucleus Mask'] + list(feats_labels_n_))
                
                if (feats_labels_n_, feats_values_n_) != ([],[]):
                    try:
                        new   = pd.Series([os.getcwd() + "\\Datasets\\Set 3D\\3D\\" + img] + [idx] + [label_image(idx)] + [nucleus] + [pixels] + feats_values_n_,index=nucDF.columns)

                        nucDF = pd.concat([nucDF,new.to_frame().T],axis=0,ignore_index=True)
                    except: 
                        pass
                
                
                
            nuid += 1
            print(100*nuid / len(np.unique(nuc_mask)))
#         # Add Centroids obtained from the image to dict:
#         Centroids[int(img.split('_')[1])] = isolated_nucleus
        
    
#         # Print progress
#         print("Image " + str(int(img.split('_')[1])) + " done.")
#     print('Done. Printing dict.')

    return nucDF