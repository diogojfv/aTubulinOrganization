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
from framework.Importing import label_image,label_image_soraia,init_import
#from PreProcessingNUCL import excludeborder, nuclei_preprocessing, df_nuclei_preprocessing, nuclei_segmentation
#from Processing import process3Dnuclei,analyze_cell
#from framework.visualization import truncate_colormap, plot_hist, plot_pie
#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC



def cytoskeleton_preprocessing(image, algorithm, parameters,plot):
    """
    Preprocessing of a cytoskeleton image
        - image      = [image, image index]
        - algorithm  = string with any algorithm
        - parameters = [sigmas, gamma]
        - plot       = bool
    """

    # Imports
    from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, laplace, threshold_yen, rank
    from skimage.util import img_as_ubyte
    from skimage.morphology import extrema, skeletonize, disk
    from skimage import filters
    
    global texture,skeleton,e,s
    
    if algorithm == 1:
        # Hessian detection
        hessian_img = hessian(image[0],black_ridges=False,sigmas=parameters[0],mode='reflect',gamma=parameters[1])
        
        # Repair image to apply to cv2 algorithm
        hessian_img = hessian_img.astype(np.uint8)  # hessian_img *= 1 # or 255
        hessian_img = 1 - hessian_img
        
        # Get texture
        texture = hessian_img
        
        # Connect
        binary_adaptive = threshold_local(texture, 3,'median')
        #o = grey_closing(TextureDF['Image'][11], size=(5,5))
        otsu = threshold_otsu(binary_adaptive)
        texture = binary_adaptive > otsu
        
        # Skeleton
        skeleton = skeletonize(texture)
        
    if algorithm == 'contour':
        
        # Contour Detection
        contours, hierarchy = cv2.findContours(image[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  

        # Filter contours with area above certain threshold
        filt_contours = []
        remo_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                filt_contours.append(contour)
            else:
                remo_contours.append(contour)

        # Draw contours
        contour_out = np.zeros_like(image[0])
        removed_out = np.zeros_like(contour_out)
        cv2.drawContours(contour_out, filt_contours, -1, 255, 3)
        cv2.drawContours(removed_out, remo_contours, -1, 255, 3)

        # Draw inside contour lines
        contour_int = np.zeros_like(image[0]) 
        cv2.fillPoly(contour_int, pts = filt_contours, color=(255,255,255))

        # Hessian detection
        hessian_img = hessian(image[0],black_ridges=False,sigmas=parameters[0],mode='reflect',gamma=parameters[1])
        
        # Repair image to apply to cv2 algorithm
        hessian_img = hessian_img.astype(np.uint8)  # hessian_img *= 1 # or 255
        hessian_img = 1 - hessian_img
        
        
        # Get texture image
        texture = contour_int * hessian_img
        
        # Skeleton
        #skeleton = skeletonize(texture)
        
        return texture, contour_out, contour_int
        

        #
        #thr = threshold_otsu(skeleton)
        #a = cv2.Canny((skeleton * 255).astype(np.uint8), 1*thr, 0.75*thr)
    
    if algorithm == 'sato':
        # METHOD 2
        texture = sato(image[0], sigmas=parameters[0], black_ridges = False, mode = 'reflect')
        thr = threshold_otsu(texture)
        texture[texture > 0.3*thr] = 1
        skeleton = skeletonize(texture)
        
    if algorithm == 'meijing':
        texture = meijering(image[0], sigmas = parameters[0], black_ridges = False, mode = 'reflect')

        thr = threshold_otsu(texture)
        #cont = cv2.Canny((meij * 255).astype(np.uint8), 1*thr, 0.75*thr)

        texture[texture > 0.2*thr] = 1
        
        # Skeleton
        skeleton = skeletonize(texture)
        
    if algorithm == 'synthetic':
        # image = [image, image index, image name, image label]
        skeleton = image[0]
        #skeleton = skeletonize(image[0]/255)
        texture = skeleton
        
    if algorithm == 'new':
        ini = image[0] / np.max(image[0])
        gau = gaussian_filter(ini, sigma=1)
        s = sato(gau,black_ridges=False,sigmas=[1],mode='reflect') 
        h = hessian(s,black_ridges=False,sigmas=[0.006],mode='reflect') 
        texture = ((1 - h)!=0)*1
        skeleton = skeletonize(texture)
        
    if algorithm == 'soraia':
        ini = image[0] / np.max(image[0])
        gau = gaussian_filter(ini, sigma=1)
        s = sato(gau,black_ridges=False,sigmas=[1],mode='reflect') 
        h = hessian(s,black_ridges=False,sigmas=[0.9],mode='reflect') 
        texture = ((1 - h)!=0)*1
        skeleton = skeletonize(texture)
        
    if algorithm == 'original':
        global f
        
        # Remove blue channel and convert to grayscale
        tmp        = copy.deepcopy(image[0])
        tmp[:,:,0] = 0
        grey       = cv2.cvtColor(tmp,cv2.COLOR_RGB2GRAY)

        # Median filter for noise removal
        grey_       = rank.median(img_as_ubyte(grey), disk(2))

        # CLAHE for color adjustment
        clahe      = cv2.createCLAHE(clipLimit = 2, tileGridSize = (8,8))
        gra        = clahe.apply(grey_)

        # Gaussian filter and Otsu thresholding
        thresh_ori = threshold_otsu(gra)
        binary_ori = gra > thresh_ori*0.8
        temp       = binary_ori * gra
        gau = gaussian_filter(temp, sigma=0.1)

        # Sato filter
        s = sato(gau,black_ridges=False,sigmas=[3],mode='reflect') 

        # LAPLACIAN OPERATOR
        d = laplace(s)

        # CLAHE
        clahe  = cv2.createCLAHE(clipLimit =2, tileGridSize=(8,8))
        e = clahe.apply(((d - d.min()) * (1/(d.max() - d.min()) * 255)).astype('uint8'))

        # OTSU
        thresh_ori = threshold_otsu(e)
        binary_ori = e > thresh_ori
        f = binary_ori * e

        # HYSTERESIS THRESHOLDING
        low = 0#int(np.max(f)*0.2)
        high = 1#int(np.max(f)*0.4)
        minima = filters.apply_hysteresis_threshold(f, low, high)

        # Yen thresholding
        # PROBLEMA: BACKGROUND COLOR NOS ~150 EM VEZ DE 0. COMO CORRIGIR? YEN NÃO FUNCIONA BEM PARA FIBRAS MENOS INTENSAS
        thresh_yen = threshold_yen(minima)
        binary_yen = minima > thresh_yen
        texture_pre = binary_yen * minima
        texture = texture_pre * temp

        # Skeletonization
        if len(np.where(minima*1 == 1)[0]) > len(np.where(minima*1 == 0)[0]):
            skeleton_pre = skeletonize(1-texture_pre*1)
        else:
            skeleton_pre = skeletonize(texture_pre*1)

        # Contour Detection
        contours, hierarchy = cv2.findContours((skeleton_pre*1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  

        # Filter contours with area above certain threshold
        remo_contours = []
        for contour in contours:
            if len(contour) <= 3:
                remo_contours.append(contour)

        removed_out = np.zeros_like(texture)
        cv2.fillPoly(removed_out, pts = remo_contours, color=(255,255,255))

        # Get final skeleton
        skeleton = skeleton_pre*1 - removed_out/255
 

    if plot:
        #plt.close('all')
        #%matplotlib qt
        plt.figure(figsize=(15,15)); plt.title('original');              plt.imshow(ini,cmap='gray');
        plt.show()
        plt.figure(figsize=(15,15)); plt.title('gaussian');              plt.imshow(gau,cmap='gray');
        plt.show()
        plt.figure(figsize=(15,15)); plt.title('sato');                  plt.imshow(s,cmap='gray');
        plt.show()
        plt.figure(figsize=(15,15)); plt.title('hessian');               plt.imshow(h,cmap='gray');
        plt.show()
        plt.figure(figsize=(15,15)); plt.title('texture');               plt.imshow(texture,cmap='gray');
        plt.show()
        plt.figure(figsize=(15,15)); plt.title('skeleton');              plt.imshow(skeleton,cmap='gray');
        plt.show()

        # ORIGINAL
#         plt.figure(figsize=(15,15)); plt.title('Original');                      plt.imshow(grey,cmap='gray'); 
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('Original -> Median Filter');     plt.imshow(grey_,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('Median Filter -> CLAHE');        plt.imshow(gra,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('CLAHE -> Otsu');                 plt.imshow(temp,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('Otsu -> Gaussian filter');       plt.imshow(gau,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('Gaussian filter -> Sato');       plt.imshow(s,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('Sato -> Laplace');               plt.imshow(d,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('Laplace -> CLAHE');              plt.imshow(e,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('CLAHE -> Otsu and Hyst. Thr.');  plt.imshow(minima,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('Otsu and Hyst. Thr. -> Yen Thr -> * by temp -> TEXTURE'); plt.imshow(texture,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('TEXTURE -> SKELETON');           plt.imshow(skeleton_pre,cmap='gray');
#         plt.show()
#         plt.figure(figsize=(15,15)); plt.title('SKELETON -> SKELETON FILT.');    plt.imshow(skeleton,cmap='gray');
#         plt.show()

    return texture,skeleton




def df_cytoskeleton_preprocessing(CYTO_df):
    TextureDF = pd.DataFrame(columns=['Name','Index','Label','Skeleton'])
    
    if len(np.unique([x.shape for x in CYTO_df['Image']])) == 2:
        for index,row in CYTO_df.iterrows():
            if type(index) == int:
                # Cytoskeleton Preprocessing
                texture,skeleton = cytoskeleton_preprocessing([row['Image'],index],  algorithm='soraia', parameters=[0.03,500],plot=False)

                # Add to DataFrame
                new       = pd.DataFrame(data={'Name': [row['Name']], 'Index': [index], 'Label': [label_image_soraia(index)], 'Skeleton': [skeleton*1]}, index = [index])
                TextureDF = pd.concat([TextureDF, new],ignore_index=False)
                
                print(">>> Image " + str(index) + " done.")

            
    return TextureDF