# # Math, image processing and other useful libraries
# from __future__ import print_function, unicode_literals, absolute_import, division
# import os
# import pandas as pd
# import numpy as np
# import cv2
# from collections import OrderedDict
# import copy
# import math
# import pickle
# from matplotlib.ticker import MaxNLocator
# from itertools import combinations

# # Image processing
# from skimage.measure import regionprops
# from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu
# from skimage.morphology import extrema, skeletonize
# from skimage.transform import probabilistic_hough_line
# from skimage.draw import disk, circle_perimeter
# from scipy.ndimage import gaussian_filter, grey_closing
# from scipy.spatial import distance_matrix
# from skimage import data, restoration, util
# from roipoly import RoiPoly
# from matplotlib_scalebar.scalebar import ScaleBar
# from biosppy.signals import tools
# from biosppy.stats import pearson_correlation
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# # Plotting
# import matplotlib.pyplot as plt
# import matplotlib.cm as pltc
# import matplotlib.colors as colors
# import seaborn as sns

# # Widgets
# import ipywidgets as widgets
# from ipywidgets import interact, interactive, fixed, interact_manual
# from IPython.display import display

# # Nuclei Segmentation
# from auxiliary_functions_segmentation import segment_patches

# # Feature Extraction (.py files by Teresa Parreira)
# from CytoSkeletonPropsMorph import CytoSkeletonPropsMorph
# from CytoSkeletonRegionPropsInt import RegionPropsInt
# from FreqAnalysis import FreqAnalysis
# from GLCM import GLCM

# # 
# from Functions import img_getTexture, label_image, FeaturesFromCentroid, cv2toski,pylsdtoski,init_import,getCentroids,polar_to_cartesian, truncate_colormap, plot_hist, plot_pie, remove_not1D, analyze_cell, quantitative_analysis,hist_bin,hist_lim 
# from fractal_dimension import fractal_dimension
# from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC

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
#from analyze_cell import analyze_cell
#from line_segment_features import line_segment_features
from matplotlib.patches import Circle
from ImageFeatures import ImageFeatures
from Functions import label_image, FeaturesFromCentroid, cv2toski,pylsdtoski,init_import,polar_to_cartesian, truncate_colormap, plot_hist, plot_pie, remove_not1D, quantitative_analysis,hist_bin,hist_lim,create_separate_DFs,branch,graphAnalysis
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC
from skimage.filters import apply_hysteresis_threshold


def remove_redundant(DF, corr_thr):
    redExtractedFeat = DF.columns
    trmv = []
    red_feature_vector = copy.deepcopy(DF).to_numpy()
    #red_feature_vector = red_feature_vector.iloc[: , 8:].to_numpy()
    
    while 1:
        oldF = redExtractedFeat
        
        # combine current set of features, two at a time
        comb = combinations(range(len(oldF)), 2) 
        
        for f in list(comb):
            
            # find correlation between the two features in analysis
            corr = pearson_correlation(red_feature_vector[:, f[0]], red_feature_vector[:, f[1]])[0]
            
            if corr > corr_thr:
                #print("Highly Corr: ", oldF[f[0]], oldF[f[1]], corr)
                trmv += [redExtractedFeat[f[0]]]
                redExtractedFeat = np.delete(oldF, f[0])
                red_feature_vector = np.delete(red_feature_vector, f[0], axis=1)
                break
                
        if len(redExtractedFeat) == len(oldF):
            break
    
    return red_feature_vector,redExtractedFeat    

def thr_redundant_analysis(threshold_range,group_feat):
    plt.figure()
    for thr in np.arange(threshold_range[0],threshold_range[1],threshold_range[2]):
        # Removal of Redundant Features
        redFeatMatrix,redFeatLabels = remove_redundant(group_feat,thr) 

        # Normalize features using StandardScaler
        redFeatMatrix = StandardScaler().fit_transform(redFeatMatrix)

        # PCA
        redFeatMatrix               = StandardScaler().fit_transform(redFeatMatrix)
        pca2                        = PCA(n_components='mle', random_state=0, svd_solver = 'full')
        features_group_pca          = pca2.fit_transform(redFeatMatrix)
        
        # Plot variances
        plt.title('Explained Variances of PC0, PC1 and both')
        plt.scatter(thr,pca2.explained_variance_ratio_[0], color='k',alpha=0.5)
        plt.scatter(thr,pca2.explained_variance_ratio_[1], color='g',alpha=0.5)
        plt.scatter(thr,pca2.explained_variance_ratio_[2], color='b',alpha=0.5)
        plt.scatter(thr,sum(pca2.explained_variance_ratio_[:2]), color='r',alpha=0.5)
        plt.scatter(thr,sum(pca2.explained_variance_ratio_[:3]), color='m',alpha=0.5)
        plt.legend(['PC0','PC1','PC2','PC0+PC1','PC0+PC1+PC2'])
        plt.xlabel('Correlation Threshold')
        plt.ylim([0,1])
        plt.tight_layout()
        
#         # PC VS. MAP
#         for pc in range(len(pca2.explained_variance_ratio_)):
#             PC = np.zeros(len(features_group_pca[:, 0]))
#             if pc == 0:
#                 PC = features_group_pca[:, 0]
#             else:
#                 for i in range(pc+1):
#                     PC = PC + features_group_pca[:, i]*pca2.explained_variance_ratio_[i]
                    
#             plt.subplot(2,1,2)
#             plt.title('PC vs. MAP')
#             if pc == 0:
#                 plt.scatter(thr,pearsonr(preprocessing.minmax_scale(MAP), preprocessing.minmax_scale(PC))[0],color='m',alpha=0.5)
#             else:
#                 plt.scatter(thr,pearsonr(preprocessing.minmax_scale(MAP), preprocessing.minmax_scale(PC))[0],color='b',alpha=0.5)
#             plt.legend(['Only PC0','Weighted Sum PC0 and PC1'])
#             plt.xlabel('Different redundancy correlation threshold')
#             plt.tight_layout()




def plot_pca_coefficients(redFeatLabels,redFeatMatrix):
    # STD normalization, PCA
    redFeatMatrix               = StandardScaler().fit_transform(redFeatMatrix)
    pca_mat                     = PCA(n_components='mle', random_state=0, svd_solver = 'full')
    feats_PCA                   = pca_mat.fit_transform(redFeatMatrix)
    
    # Plot
    for i in range(2):
        plt.figure(figsize=(15,15))
        global sort_
        sort_ = np.stack((pca_mat.components_[i,:],redFeatLabels)).T
        sort_ = sort_[sort_[:,0].argsort()[::-1]]
        plt.bar(sort_[:,1],sort_[:,0])
        plt.xticks(np.arange(len(sort_[:,1])), (sort_[:,1]),rotation=45, ha='right',fontsize=8)
        plt.ylim([-1,1])
        plt.ylabel('Coefficients')
        plt.xlabel('Features')
        plt.show()
        
def plot_PCA(array,labels,ResultsDF):
    # Colors and Markers
    colors  = ["#3498DB","#E74C3C","#95A5A6","#ABE6FF","#CA6F1E","#2ECC71"]
    markers = ["o","v","^","p","h","s"]
    
    print('working...')
    # STD normalization, PCA
    redFeatMatrix               = StandardScaler().fit_transform(array)
    pca                         = PCA(n_components='mle', random_state=0, svd_solver = 'full')
    feats_PCA                   = pca.fit_transform(redFeatMatrix)

    # Labels
    ls = list(np.unique(ResultsDF['Label']))
    
    c  = [colors[ls.index(x)] for x in ResultsDF['Label']]
    m  = [markers[ls.index(y)] for y in ResultsDF['Label']]

    # Plot 2D
    #%matplotlib inline
    fig, ax = plt.subplots(figsize=(15,15))
    #plt.figure(figsize=(15,15))
    clusterpoints = feats_PCA[:,0:2]

    for p in range(len(ls)):
        points = clusterpoints[ResultsDF['Label'] == ls[p]]
        ax.scatter(points[:,0],points[:,1],c=colors[p],marker=markers[p],label=ls[p],alpha=0.7)

    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend()

    # mid point
    for p in range(len(ls)):
        points = clusterpoints[ResultsDF['Label'] == ls[p]]
        ax.scatter(np.mean(points,axis=0)[0],np.mean(points,axis=0)[1],c=colors[p],marker=markers[p],label=ls[p],s=300,edgecolors='k',zorder=5)
    plt.show()

    # STATS
    print('Final Number of Features: ' + str(len(labels)))
    print('Variances from PC0, PC1 and PC2: ' + str(pca.explained_variance_ratio_[:3]))
    print('Total Variance Explained (2D) = ' + str(sum(pca.explained_variance_ratio_[:2])))
    print('Total Variance Explained (3D) = ' + str(sum(pca.explained_variance_ratio_[:3])))

    
    # # 3D
    # %matplotlib qt
    # fig = plt.figure(figsize=(15,15))
    # ax = fig.add_subplot(projection='3d')
    # for p in range(len(c)):
    #     try:
    #         ax.scatter(feats_PCA[p,0],feats_PCA[p,1],feats_PCA[p,2],c=c[p],marker=m[p],label=ResultsDF['Label'][p])
    #         plt.xlabel('PC1')
    #         plt.ylabel('PC2')
    #         plt.zlabel('PC3')
    #     except:
    #         continue

    #pickle.dump(fig, open('PCA_LSF_95.fig.pickle', 'wb'))


# Perform interactive PCA visualization
def plot_PCA_1(array,labels,ResultsDF,wid_list):
    # Colors and Markers
    colors  = ["#3498DB","#E74C3C","#95A5A6","#ABE6FF","#CA6F1E","#2ECC71"]
    markers = ["o","v","^","p","h","s"]
    
    # STD normalization, PCA
    redFeatMatrix               = StandardScaler().fit_transform(array)
    pca                         = PCA(n_components='mle', random_state=0, svd_solver = 'full')
    feats_PCA                   = pca.fit_transform(redFeatMatrix)

    # Labels
    ls = np.unique(ResultsDF['Label'])
    c = [colors[list(ls).index(x)] for x in ResultsDF['Label']]
    m = [markers[list(ls).index(y)] for y in ResultsDF['Label']]

    # Plot 2D
    fig, ax = plt.subplots(figsize=(15,15))
    #plt.figure(figsize=(15,15))
    clusterpoints = feats_PCA[:,0:2]

    for p in range(len(ls)):
        points = clusterpoints[ResultsDF['Label'] == ls[p]]
        if wid_list[p] == True:
            ax.scatter(points[:,0],points[:,1],c=colors[p],marker=markers[p],label=ls[p],zorder=4,alpha=0.7)
        else:
            ax.scatter(points[:,0],points[:,1],c='w',marker=markers[p],label=ls[p],zorder=1,alpha=0.7)

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()

    # mid point
    for p in range(len(ls)):
        points = clusterpoints[ResultsDF['Label'] == ls[p]]
        if wid_list[p] == True:
            ax.scatter(np.mean(points,axis=0)[0],np.mean(points,axis=0)[1],c=colors[p],marker=markers[p],label=ls[p],s=300,edgecolors='k',zorder=5,alpha=0.5)
        else:
            ax.scatter(points[:,0],points[:,1],c='w',marker=markers[p],label=ls[p],zorder=1,alpha=0.5)

    plt.show()