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
from framework.processing_DCF import *
from framework.Functions import cv2toski,pylsdtoski,polar_to_cartesian, remove_not1D, quantitative_analysis,hist_bin,hist_lim,branch,graphAnalysis
from framework.importing import *
#from framework.PreProcessingCYTO import cytoskeleton_preprocessing, df_cytoskeleton_preprocessing
#from framework.PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation
from framework.processing import *
#from framework.visualization import truncate_colormap, plot_hist, plot_pie
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC

def cyto_network_features(ResultsRow,listfeats):
    """
    - ResultsRow   = ResultsDF row
    - features     = list of features
    """
    
    # Initialize CNF list
    res = []
    
    if 'CNF1D:Number of Branches' in listfeats:
        res += [('CNF1D:Number of Branches', branches_number(ResultsRow))]
        
    if 'CNF2D:Branch Lengths' in listfeats:
        res += [('CNF2D:Branch Lengths', graph_path_lengths(ResultsRow))]
    
    if 'CNF2D:Branch Orientations' in listfeats:
        res += [('CNF2D:Branch Orientations', graph_orientation(ResultsRow))]
        
    if 'CNF2D:Branch Orientations PCA' in listfeats:
        res += [('CNF2D:Branch Orientations PCA', graph_orientation_pca(ResultsRow))]
        
    if 'CNF2D:Local Average Branch Distances' in listfeats:
        res += [('CNF2D:Local Average Branch Distances', graph_compactness_bundling_parallelism(ResultsRow,neighborhood=5)[0])]
    
    if 'CNF2D:Local Average Bundling Score' in listfeats:
        res += [('CNF2D:Local Average Bundling Score', graph_compactness_bundling_parallelism(ResultsRow,neighborhood=5)[1])]
        
    if 'CNF2D:Local Average Branch Orientation' in listfeats:
        res += [('CNF2D:Local Average Branch Orientation', graph_compactness_bundling_parallelism(ResultsRow,neighborhood=5)[2])]
        
    if 'CNF2D:Distances to Centroid' in listfeats:
        res += [('CNF2D:Distances to Centroid', graph_cytonuc_dist(ResultsRow))]
    #if 'CNF2D:Mean Filament Thickness' in listfeats:
        
    #cont.
    return res

# ResultsDF['CNF2D:Branch Orientation']               = [graph_orientation(getske(row,data)) for index,row in ResultsDF.iterrows()]
# ResultsDF['CNF2D:Branch Orientation PCA']           = [graph_orientation_pca(getske(row,data)) for index,row in ResultsDF.iterrows()]
# ResultsDF['CNF2D:Local Average Branch Distance']    = [graph_compactness_bundling_parallelism(getske(row,data),neighborhood=5)[0] for index,row in ResultsDF.iterrows()]
# ResultsDF['CNF2D:Local Average Bundling Score']     = [graph_compactness_bundling_parallelism(getske(row,data),neighborhood=5)[1] for index,row in ResultsDF.iterrows()]
# ResultsDF['CNF2D:Local Average Branch Orientation'] = [graph_compactness_bundling_parallelism(getske(row,data),neighborhood=5)[2] for index,row in ResultsDF.iterrows()]
# ResultsDF['CNF2D:Mean Filament Thickness']          = [graph_thickness(row) for index,row in ResultsDF.iterrows()]
#ResultsDF['CNF2D:Distances to Centroid (scaled)']            = [graph_cytonuc_dist(row) for index,row in ResultsDF.iterrows()]

#from scipy.stats import circvar
#from framework.Processing import newtheta,OOP

#ResultsDF['CNF1D:Circular Variance'] =  [circvar(np.array(l)*np.pi/180,low=0,high=np.pi) for l in ResultsDF['CNF2D:Branch Orientation']]
#ResultsDF['CNF1D:OOP'] = [OOP(anglist) for anglist in ResultsDF['CNF2D:Branch Orientation']]
#ResultsDF['CNF1D:N over A (scaled)'] = ResultsDF['SKNW:Number of Branches'] / (ResultsDF['DCF:Area 2']*0.16125**2)

def getske(ResultsRow,data):
    img = retrieve_mask(ResultsRow['Skeleton'],ResultsRow['Image Size'])
    #img       = ResultsRow['Patch:Skeleton Max'] 
    ske       = Skeleton(skeleton_image = img.astype(float),
                         spacing        = 0.1612500) # ResultsRow['Resolution'][2] ou [1] ou os dois
    
    
#     img       = data['CYTO']['Image'][ResultsRow['Img Index']] / np.max(data['CYTO']['Image'][ResultsRow['Img Index']])
#     intensity = ResultsRow['Mask'] * img
#     ske       = Skeleton(skeleton_image=(ResultsRow['Mask']*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']]*(intensity/np.max(intensity))).astype(float),spacing=0.1612500) 
    
    return ske

def graph_path_lengths(ResultsRow):
    ske = getske(ResultsRow,data)
    return ske.path_lengths()
                
def branch_middle(path):
    L = len(path)
    if L % 2 == 0: # L Even
        mid = np.mean([path[int(L/2)-1],path[int(L/2)]],axis=0)
    if L % 2 != 0: # L Odd
        mid = path[int(L//2)]

    return mid

def graph_middles(sk):
    return [branch_middle(sk.path_coordinates(b)) for b in range(sk.n_paths)]

def graph_thickness(ResultsRow,data):
    from scipy import ndimage
    
    # Get skeleton and texture
    sk      = ResultsRow['Mask']*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']]
    texture = ResultsRow['Mask']*data['CYTO_PRE']['Texture'][ResultsRow['Img Index']]
    
    # Get Euclidean Distance Transform
    edt   = ndimage.distance_transform_edt(texture)
    edtsk = edt*sk
    
    # Get skeleton with intensities related to thickness
    ske = Skeleton(skeleton_image=edtsk.astype(float),spacing=0.1612500) 
    
    # Mean filament thickness
    return ske.path_means()

def curvaturepixel(x, y, path_array):
    # Find the closest points on the path flanking both sides of (x, y)
    idx = np.argmin((path_array[:, 0] - x)**2 + (path_array[:, 1] - y)**2)

    # Ensure we have valid indices for the neighboring points
    idx1 = max(0, idx - 1)
    idx2 = min(len(path_array) - 1, idx + 1)

    # Extremities
    if idx1 == 0 or idx2 == len(path_array)-1:
        return 0

    x1, y1 = path_array[idx1]
    x2, y2 = path_array[idx2]

    # Compute the curvature using the provided formula
    curvature_val = (
        2 * np.abs((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y)) /
        np.sqrt(((x - x1)**2 + (y - y1)**2) *
                ((x - x2)**2 + (y - y2)**2) *
                ((x1 - x2)**2 + (y1 - y2)**2)))

    return curvature_val
    
def branch_curvature(path):
    return [curvaturepixel(pix[0],pix[1],path) for pix in path]

def branch_orientation(path):
    tangent_vectors = np.diff(path, axis=0) 
    angles = np.arctan2(tangent_vectors[:, 0], tangent_vectors[:, 1])
    angles_deg = np.degrees(angles)
    orientation_deg = np.mean(angles_deg)
    
    if orientation_deg >= 0:
        return orientation_deg
    if orientation_deg < 0:
        return orientation_deg + 180
    
def graph_orientation(ResultsRow):
    sk = getske(ResultsRow,data)
    return [branch_orientation(sk.path_coordinates(b)) for b in range(sk.n_paths)]

def branch_orientation_pca(path):
    # PCA
    _, _, V = np.linalg.svd(path - np.mean(path, axis=0))

    # The first principal component represents the major axis
    major_axis = V[0, :] 

    # Calculate the angle of the major axis
    orientation_deg = np.degrees(np.arctan2(major_axis[0], major_axis[1]))

    if orientation_deg >= 0:
        return orientation_deg
    if orientation_deg < 0:
        return orientation_deg + 180

def graph_orientation_pca(ResultsRow):
    sk = getske(ResultsRow,data)
    return [branch_orientation_pca(sk.path_coordinates(b)) for b in range(sk.n_paths)]



def graph_compactness_bundling_parallelism(ResultsRow,neighborhood = 5):
    sk = getske(ResultsRow,data)
    # Get median points and orientations
    median_points = graph_middles(sk)
    orientations  = graph_orientation(ResultsRow)
    
    # Distance matrix
    d             = distance_matrix(median_points,median_points); np.fill_diagonal(d,np.max(d));
    d_0           = distance_matrix(median_points,median_points); np.fill_diagonal(d_0,0);
    max_d         = np.max(d)
    
    # Create result lists
    local_distances    = []
    local_bundlescores = []
    local_angles       = []
    
    for br in range(sk.n_paths):
        brcoords = sk.path_coordinates(br)
        
        med_point   = median_points[br]
        orientation = orientations[br] 
        
        
        # Distance matrix for given branch
        copy_d            = copy.deepcopy(d[br])
        dists_to_medpoint = copy_d
        
        # Create local results lists
        closest_angles = []
        prox_br        = []
        
        for _ in range(neighborhood):
            # calculate angle of the _'th closest line - MANDATORY
            min_val          = np.min(dists_to_medpoint)
            close_line_ind   = np.where(dists_to_medpoint == min_val)[0][0]
            
            # Compactness
            med_point_c      = median_points[close_line_ind]
            prox_br        += [min_val]
            
            # Parallelism
            orientation_c    = orientations[close_line_ind]
            smallest_angle = min(abs(orientation - orientation_c), 180 - abs(orientation - orientation_c))
            closest_angles = [smallest_angle]

            # Bundling
            bundle_score = smallest_angle * min_val
        
            # Next branch
            dists_to_medpoint[close_line_ind] = max_d
        
        
        # Compactness
        local_distances += [round(np.mean(prox_br),3)]
        
        # Bundling
        local_bundlescores += [round(np.mean(bundle_score),3)]
        
        # Parallelism
        local_angles += [round(np.mean(closest_angles),3)]
        
    return local_distances,local_bundlescores,local_angles


def branch_cytonuc_dist(path,c):
    # Median point
    med_point  = branch_middle(path)
    
    # Median point - Centroid array
    center_med_vec = np.array(med_point) - np.array([c[1],c[0]])
    
    return round(np.linalg.norm(center_med_vec)*0.1612500,3)

def graph_cytonuc_dist(ResultsRow):
    #x_,y_   = np.where((ResultsRow['Mask']*1) != 0)
    #imgIndex = ResultsRow['Img Index']

#     for index,row in CentroidsDF[imgIndex].iterrows():
#         if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(x_,y_)):
#             centroid = (row['Centroid'][1],row['Centroid'][0])
#             break
#     try:
#         centroid
#     except:
#         centroid = (0,0)
#         print('centroid not found. set to (0,0)')

    centroid = ResultsRow['Nucleus Centroid']
    ske      = getske(ResultsRow,data)
        
    #ske = getske(ResultsRow,data)
    #ske       = Skeleton(skeleton_image=(ResultsRow['Mask']*data['CYTO_PRE']['Skeleton'][ResultsRow['Img Index']]).astype(float)) 
    
    return [branch_cytonuc_dist(ske.path_coordinates(b),centroid) for b in range(ske.n_paths)]

def branch_cytonuc_rs(path,c):
    orient = branch_orientation(path)
    
    angle = np.arccos(np.dot(center_med_vec / np.linalg.norm(center_med_vec), line_vec / np.linalg.norm(line_vec)))*180/np.pi
    if angle > 90: angle = 180 - angle;

#v,branch_orientation(v),branch_orientation_pca(v)
#graph_middles(ske)
#graph_orientation(ske)
#graph_orientation_pca(ske)
#graph_compactness_bundling_parallelism(ske,neighborhood = 5)[0]
#graph_compactness_bundling_parallelism(ske,neighborhood = 5)[1]
#graph_compactness_bundling_parallelism(ske,neighborhood = 5)[2]
#graph_thickness(ResultsRow)

def branches_number(ResultsRow):
    ske = getske(ResultsRow,data)
    return ske.n_paths
    

#ResultsDF['SKNW:Number of Branches'] = [branches_number(row) for _,row in ResultsDF.iterrows()]