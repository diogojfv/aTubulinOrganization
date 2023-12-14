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
from framework.Processing import centroid_find
#from framework.visualization import truncate_colormap, plot_hist, plot_pie
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC

def lines_theta(ResultsRow):
    lines = ResultsRow['Lines LinReg']
    thetas = []
    for line in lines:
        p0, p1 = line
        line_vec = np.array(p1) - np.array(p0)
        
        if p0[0] != p1[0]:
            #theta = np.arctan(abs(p1[1]-p0[1])/abs(p1[0]-p0[0]))*180/np.pi
            theta = np.arctan(line_vec[1]/line_vec[0])*180/np.pi
        else:
            theta = 90
        if not 0 < theta < 180:  theta = 180 + theta;
            
        thetas += [theta]
    return thetas

#ResultsDF["LSF2D:Theta (LinReg)"] = [lines_theta(row) for index,row in ResultsDF.iterrows()]

def lines_cytonuc_dist(ResultsRow,CentroidsDF):
    centroid = centroid_find(ResultsRow,CentroidsDF)
    
    dists = []
    for line in ResultsRow['Lines LinReg']:
        p0, p1 = line
        
        med_point      = ((p0[0] + p1[0])/2,(p0[1] + p1[1])/2)
        center_med_vec = np.array(med_point) - np.array([centroid[0],centroid[1]])
        dists += [round(np.linalg.norm(center_med_vec)*0.16125,3)]
        
    return dists

#ResultsDF["LSF2D:Distances to Centroid (LinReg) (scaled)"] = [lines_cytonuc_dist(row) for index,row in ResultsDF.iterrows()]


def lines_cytonuc_alpha(ResultsRow,CentroidsDF):
    centroid = centroid_find(ResultsRow,CentroidsDF)
    
    alphas = []
    for line in ResultsRow['Lines LinReg']:
        p0, p1 = line
        
        med_point      = ((p0[0] + p1[0])/2,(p0[1] + p1[1])/2)
        center_med_vec = np.array(med_point) - np.array([centroid[0],centroid[1]])
        line_vec       = np.array(p1) - np.array(p0)
        angle = np.arccos(np.dot(center_med_vec / np.linalg.norm(center_med_vec), line_vec / np.linalg.norm(line_vec)))*180/np.pi
        if angle > 90: angle = 180 - angle;
        
        alphas += [round(angle,3)]
        
    return alphas

# ResultsDF["LSF2D:Alphas (LinReg)"] = [lines_cytonuc_alpha(row) for index,row in ResultsDF.iterrows()]

def lines_angdiff(ResultsRow):
    centroid = centroid_find(ResultsRow)
    
    median_points = [((line[0][0] + line[1][0])/2,(line[0][1] + line[1][1])/2) for line in ResultsRow['Lines LinReg']]
    d             = distance_matrix(median_points,median_points); np.fill_diagonal(d,np.max(d));
    d_0           = distance_matrix(median_points,median_points); np.fill_diagonal(d_0,0);
    
    ind = 0
    for line in ResultsRow['Lines LinReg']:
        p0, p1 = line
        
        med_point      = ((p0[0] + p1[0])/2,(p0[1] + p1[1])/2)
        center_med_vec = np.array(med_point) - np.array([centroid[0],centroid[1]])
        line_vec       = np.array(p1) - np.array(p0)
        
        ### LOCAL FEATURES
        # CLOSEST LINES vs. THIS LINE
        copy_d            = copy.deepcopy(d[ind])
        dists_to_medpoint = copy_d
        closest_angles = []
        
        for _ in range(5):
            # calculate angle of the _'th closest line - MANDATORY
            min_val          = np.min(dists_to_medpoint)
            close_line_ind   = np.where(dists_to_medpoint == min_val)[0][0]
            p0_c, p1_c       = lines[close_line_ind] 
            med_point_c      = ((p0_c[0] + p1_c[0])/2,(p0_c[1] + p1_c[1])/2)
            center_med_vec_c = np.array(med_point_c) - np.array([centroid[1],centroid[0]])
            line_vec_c       = np.array(p1_c) - np.array(p0_c)

            closest_angles = closest_angles + [abs(theta - theta_c)]
        
    return # FAZER

def lines_CVar(ResultsRow):
    from scipy.stats import circvar
    
    angles = ResultsRow['LSF2D:Theta (LinReg)']
    
    return circvar(np.array(angles)*np.pi/180,low=0,high=np.pi)

# ResultsDF['LSF1D:Circular Variance (LinReg)'] =  [lines_CVar(row) for index,row in ResultsDF.iterrows()]

def lines_OOP(ResultsRow):
    angles = ResultsRow["LSF2D:Theta (LinReg)"]
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

# ResultsDF['LSF1D:OOP (LinReg)'] = [lines_OOP(row) for _,row in ResultsDF.iterrows()]


def lines_N(ResultsRow):
    return len(ResultsRow["Lines LinReg"])

# ResultsDF['LSF1D:Number of Lines (LinReg)'] = [lines_N(row) for _,row in ResultsDF.iterrows()]


def lines_RS_cent_dist(ResultsRow,CentroidsDF):
    x_,y_   = np.where((ResultsRow['Mask']*1) != 0)
    imgIndex = ResultsRow['Img Index']
    # Find centroid
    for index,row in CentroidsDF[imgIndex].iterrows():
        if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(x_,y_)):
            centroid = (row['Centroid'][1],row['Centroid'][0])
            break
    try:
        centroid
    except:
        centroid = (0,0)
        print('centroid not found. set to (0,0)')
    
    rspos = np.array([ResultsRow['LSF:Radial Pos 2'][1],ResultsRow['LSF:Radial Pos 2'][0]])
    return np.linalg.norm(np.array(centroid) - rspos)*0.16125

#ResultsDF["LSF1D:RS NucCent Distance (LinReg) (scaled)"] = [lines_RS_cent_dist(row) for index,row in ResultsDF.iterrows()]



# OTHERS 
# ResultsDF['LSF1D:N over A (LinReg)'] = ResultsDF['LSF1D:Number of Lines (LinReg)'] / (ResultsDF['DCF:Area (scaled)'])
