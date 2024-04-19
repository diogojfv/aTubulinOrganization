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
from framework.processing_DCF import *
from framework.Functions import cv2toski,pylsdtoski,polar_to_cartesian, remove_not1D, quantitative_analysis,hist_bin,hist_lim,branch,graphAnalysis
from framework.importing import *
#from framework.PreProcessingCYTO import cytoskeleton_preprocessing, df_cytoskeleton_preprocessing
from framework.processing import process3Dnuclei,analyze_cell
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


def nuclei_segmentation(rowNUCL,algorithm,algorithm_specs,plot,save):
    from skimage.filters import threshold_otsu
    from tifffile import imread,imwrite
    
    idx = rowNUCL.index
    path            = rowNUCL['Path']
    image           = rowNUCL['Image']
    name            = rowNUCL['Name']
    
    if len(image.shape) == 2:
        dim = 2
    if len(image.shape) == 3:
        dim = 3

    global final
    
    def plot_centroids_from_binmask(ax,nucmask):
        for mask in np.unique(nucmask)[1:]:
            # Isolate nucleus
            nucleus = nucmask == mask
            nucleus = nucleus*nucmask
            bin_nuc = np.where(nucleus>0.5, 1, 0) 

            # Get and plot centroid
            props = regionprops(bin_nuc, bin_nuc)
            ax.plot(props[0].centroid[1],props[0].centroid[0],'o',color='red',markersize=7,zorder=5)
    
    # ALGORITHM = STARDIST
    if algorithm == "stardist":
        from csbdeep.utils import Path, normalize
        from csbdeep.io import save_tiff_imagej_compatible
        from skimage.exposure import equalize_adapthist
        from scipy import ndimage
            
        # IMAGE SIZE = 2D
        if dim == 2:   
            # Define model
            from stardist.models import StarDist2D
            model = StarDist2D.from_pretrained('2D_versatile_fluo')
            
            # Normalize image
            img = normalize(image.astype(np.uint16), 1,99.8, axis=(0,1))

            # Apply model
            labels, details = model.predict_instances(img,verbose=True)
            
            # Remove outliers
            labels_ = copy.deepcopy(labels)
            N_excluded = 0
            for vol_id in np.unique(labels_)[1:]:
                u = labels_ == vol_id
                u = u*vol_id
                if len(np.where(u == vol_id)[0]) < algorithm_specs[1] or excludeborder(u) == True:
                    labels_ = labels_ - u
                    N_excluded += 1
            print("Number of excluded nuclei: " + str(N_excluded))
            
            final = labels_
            
        # IMAGE SIZE = 3D 
        if dim == 3:
            # Define model
            from stardist.models import StarDist3D
            model = StarDist3D.from_pretrained('3D_demo')

            # Normalize image
            img = normalize(image.astype(np.uint16), 1,99.8, axis=(0,1,2))

            # Apply model
            labels, details = model.predict_instances(img,prob_thresh=0.75,nms_thresh=0.3,verbose=True)

            # Remove outliers
            #labels_ = copy.deepcopy(labels)
            labels_ = remove_small_objects(labels, min_size=algorithm_specs[1])
            
#             N_excluded = 0
#             for vol_id in np.unique(labels_)[1:]:
#                 u = labels_ == vol_id
#                 u = u*vol_id
#                 if len(np.where(u == vol_id)[0]) < algorithm_specs[1] or excludeborder(u) == True:
#                     labels_ = labels_ - u
#                     N_excluded += 1
#             print("Number of excluded nuclei: " + str(N_excluded))

            final = labels_
                    
                    
                    
    # ALGORITHM = CONTOUR
    if algorithm == "contour": 
        from skimage.measure import label
        from skimage.morphology import remove_small_objects
        from scipy import ndimage
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
        
        if dim == 2:
            # Image > Otsu
            thresh_ori = threshold_otsu(image)
            binary_ori = image > thresh_ori*algorithm_specs[0]
            image_     = binary_ori * image
            

            # Get nuclei contours
            contours, hierarchy = cv2.findContours(image_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            # Draw contours and 
            copi = copy.deepcopy(image)
            contour_int = np.zeros_like(image)
            cv2.fillPoly(contour_int, pts = contours, color=(255,255,255))
            copi = contour_int
            copi = ndimage.binary_fill_holes(copi)
            
            labels__ = label(copi)
            labels__ = remove_small_objects(labels__, min_size=algorithm_specs[1])
            
            # Watershed segmentation
            distance = ndimage.distance_transform_edt((labels__!=0)*1)
            local_maxi = peak_local_max(distance, footprint=np.array(np.ones((7, 7)),dtype=np.float64),min_distance=25,  threshold_rel=0.3, labels=labels__.astype(np.int32))
            mask = np.zeros(distance.shape, dtype=bool)
            mask[tuple(local_maxi.T)] = True
            markers, _ = ndimage.label(mask)
            labels = watershed(-distance, markers, mask=binary_ori)
            
            #labels = label(copi)
            
            # Remove outliers
            labels_ = remove_small_objects(labels, min_size=algorithm_specs[1])

            
#             labels_ = copy.deepcopy(labels)
#             N_excluded = 0
#             for vol_id in np.unique(labels_)[1:]:
#                 u = labels == vol_id
#                 u = u*vol_id
#                 if len(np.where(u == vol_id)[0]) < algorithm_specs[1] or excludeborder(u) == True:
#                     N_excluded += 1
#                     labels_ = labels_ - u
#             print("Number of excluded nuclei: " + str(N_excluded))
            
            final = np.array(labels_,dtype=np.uint16)
            
#             # Analyse each contour
#             add_contours = []
#             dil_contours = []
#             centroids    = []
#             for cnt in contours:
#                 cr = cnt.reshape((cnt.shape[0],cnt.shape[2]))

#                 if excludeborder(cr) == True:
#                     continue

#                 # Size filter
#                 if cv2.contourArea(cnt) >= algorithm_specs[1]:
#                     # Create image with contour
#                     fill_temp = np.zeros_like(image)
#                     cv2.fillPoly(fill_temp, pts = [cnt], color=(255,255,255))

#                     # Dilate previous image and get contour
#                     dil_fill = binary_dilation(fill_temp, disk(thr[2], dtype=bool))
#                     contour_dil, hierarchy_dil = cv2.findContours((dil_fill*1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#                     # Get centroid of contour
#                     props    = regionprops(dil_fill*1,dil_fill*1)
#                     centroid = props[0].centroid 
#                     centroids.append(centroid)

#                     # Add contours to list
#                     add_contours.append(cnt)
#                     dil_contours.append(contour_dil[0])

#             # Get nuclei masks
#             fillcontours = np.zeros_like(image)
#             cv2.fillPoly(fillcontours, pts = add_contours, color=(255,255,255))
#             dilcontours = np.zeros_like(image)
#             cv2.fillPoly(dilcontours, pts = dil_contours, color=(255,255,255))

#             # Result 
#             res = (dilcontours/255).astype(np.uint8) * image

#             if plot:
#                 plt.figure(figsize=(12,12))
#                 plt.imshow(image,cmap='gray')
#                 plt.show()
#                 plt.figure(figsize=(12,12))
#                 plt.imshow(fillcontours,cmap='gray')
#                 plt.show()
#                 plt.figure(figsize=(12,12))
#                 plt.imshow(dilcontours,cmap='gray')
#                 plt.show()
#                 plt.figure(figsize=(12,12))
#                 plt.imshow(res,cmap='gray')
#                 for cx,cy in centroids:
#                     plt.plot(cy,cx,'o',color='red',markersize=7)

#                 plt.show()

            #return dil_contours,centroids,res
        
        if dim == 3:
            from skimage.segmentation import clear_border
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max
            from scipy import ndimage as ndi
            
            temp = image
            thresh = threshold_otsu(temp)
            binary = temp > thresh*algorithm_specs[0]
            imagem = binary * temp

            copi = copy.deepcopy(imagem)

            # For each slice
            for sl in range(len(imagem)):
                # Contour Detection of slice sl
                contours, hierarchy = cv2.findContours(imagem[sl].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  

                # Draw contours and 
                contour_int = np.zeros_like(imagem[sl])
                cv2.fillPoly(contour_int, pts = contours, color=(255,255,255))
                copi[sl] = contour_int
                copi[sl] = clear_border(copi[sl])
                
            copi_ = copy.deepcopy(copi)
#             for slc in range(len(copi2)):
#                 copi_[slc] = clear_border(copi2[slc])
            
            res = label(copi_)
            
            copi2 = remove_small_objects(res, min_size=algorithm_specs[1])
            
            labels_ = copy.deepcopy(copi2)
            
#             N_excluded = 0
#             for vol_id in np.unique(labels_)[1:]:
#                 u = res == vol_id
#                 u = u*vol_id
#                 if len(np.where(u == vol_id)[0]) < algorithm_specs[1]:
#                     N_excluded += 1
#                     labels_ = labels_ - u
#             print("Number of excluded nuclei: " + str(N_excluded))
            
#             dist_transform = cv2.distanceTransform(labels_,cv2.DIST_L2,5)
#             ret2, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
            
#             distance = ndi.distance_transform_edt(labels_)
#             distance = gaussian_filter(distance, sigma=3)
#             coords = peak_local_max(distance, footprint=np.ones((1000, 1000, 1000)), labels=labels_)
#             mask = np.zeros(distance.shape, dtype=bool)
#             mask[tuple(coords.T)] = True
#             markers, _ = ndi.label(mask)
#             labels_ = watershed(-distance, markers, mask=labels_)
    
            final = np.array(labels_,dtype=np.uint16)
    
    
    if plot and dim == 2:
        # ORIGINAL
        fig,ax = plt.subplots(figsize=(15,15))
        ax.imshow(image,cmap='gray')
        ax.axis('off')
        #ax.axis('equal')
        plt.show()
        
        # MASKS
        fig,ax = plt.subplots(figsize=(15,15))
        ax.imshow(final,cmap='viridis')
        ax.axis('off')
        #ax.axis('equal')
        plot_centroids_from_binmask(ax,final)
        plt.show()
    
    if plot and dim == 3:
        pass
    
    # Write in desired path
    if save:
        #print(np.unique(final))
        imwrite(os.path.join(save, name), final,photometric='minisblack')
    
    
def nuclei_preprocessing(rowNUCL,dir_masks,plot,save):
    from tifffile import imread
    from skimage.morphology import binary_dilation,ball,disk
    global NUCL_PRE,contour
    
    idx   = rowNUCL.name
    name  = rowNUCL['Name']
    image = rowNUCL['Image']
    label = rowNUCL['Label']
    
    if len(image.shape) == 2:
        dim = 2
    elif len(image.shape) == 3 and name.split('_')[-1] == 'ch00.tif':
        dim = 3
    else:
        print(">>>>>> PREPROCESSING: Image " + str(name) + " error.")
        return NUCL_PRE
    
    print(">>>>>> PREPROCESSING: Image " + str(name))

    path = dir_masks + str("\\") + name
    nuc_mask = imread(path)
    print(path)
#     if dim == 2:
#         try:
#             idx = name.split('_')[1] # DATASET SORAIA
#             #idx = path.split('/')[1].split('_')[1]
#         except:
#             idx = path.split('/')[-1].split('_')[2]
#     if dim == 3:
#         #idx = path.split('/')[1].split('_')[0]
#         idx = path.split('\\')[-1].split('_')[0]
        
     
    # Open figure
    if plot:
        if dim == 2:
            fig,ax = plt.subplots(figsize=(8,8))
            plt.axis('off')
            plt.imshow(nuc_mask,cmap='gray')
          
                                      
    # Obtain isolated nucleus
    nuc = 0
    
    for nucleus in np.unique(nuc_mask)[1:]:
        # Obtain isolated nucleus
        aux  = nuc_mask == nucleus
        mask_w_id = aux*nuc_mask
        bin_nuclei = np.where(mask_w_id>0.5, 1, 0) 



        # Get contour
        if dim == 2:
            contour, hierarchy = cv2.findContours((bin_nuclei*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #dil = binary_dilation(bin_nuclei,footprint=disk(3))
        if dim == 3:
            contour = 0
            dil = binary_dilation(bin_nuclei,footprint=ball(3))
        

        imgg = bin_nuclei * (image / np.max(image))
        imgg = imgg / np.max(imgg)


        if dim == 2:
            xp,yp  = np.where(bin_nuclei>0)
#             imgg   = imgg[min(xp):max(xp),min(yp):max(yp)]
            pixels = [xp,yp]

        if dim == 3:
            xp,yp,zp = np.where(dil>0)
            #print(min(xp),max(xp),min(yp),max(yp),min(zp),max(zp))
            #imgg     = imgg[min(xp):max(xp),min(yp):max(yp),min(zp):max(zp)]
            pixels   = [xp,yp,zp]

        imgg = (imgg *255).astype(np.uint8)
        #print(imgg.shape,np.unique(imgg))
        props = regionprops((imgg!=0)*1, imgg)


        if dim == 2:
            #cent = round(props[0].centroid[0] + min(xp),3), round(props[0].centroid[1] + min(yp),3)
            cent = round(props[0].centroid[0],3), round(props[0].centroid[1],3)
        if dim == 3:
            #cent = round(props[0].centroid[2] + min(zp),3), round(props[0].centroid[0] + min(xp),3), round(props[0].centroid[1] + min(yp),3)
            cent = props[0].centroid[0], props[0].centroid[1], props[0].centroid[2]


        # Add to dataframe
        global new
        if 'NUCL_PRE' not in globals():
            NUCL_PRE = pd.DataFrame(columns = ['Img Index'] + ['Label'] + ['Nucleus Mask'] + ['Centroid'] + ['Contour']) 
        new   = pd.Series([idx] + [label] + [pixels] + [cent] + [contour[0]],index=NUCL_PRE.columns)
        NUCL_PRE = pd.concat([NUCL_PRE,new.to_frame().T],axis=0,ignore_index=True)
        
        
        # Plot
        if plot:
            if dim == 2:
                #plt.plot(cent[1] + min(yp),cent[0] + min(xp),'o',color='red',markersize=7)
                plt.plot(cent[1],cent[0],'o',color='red',markersize=7)
                
                cr = contour[0].reshape((contour[0].shape[0],contour[0].shape[2]))
                ax.plot(cr[:,0],cr[:,1],'--',color='#6495ED',zorder=11,linewidth=3)
#                 if dim == 3:
#                     test = label2rgb(nuc_mask, image=image, bg_label=0)
#                     sl = 1
#                     for im in test:
#                         plt.figure(figsize=(15,15))
#                         plt.axis('off')
#                         plt.imshow(im)
#                         plt.plot(cent[1],cent[0],'o',color='red',markersize=7)
#                         #plt.savefig("..//tempp//id70_slice" + str(sl) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=200)
#                         plt.show()  
#                         sl += 1



        # Print progress
        nuc += 1
        print('>>> Preprocessing: ' + str(100*nuc / len(np.unique(nuc_mask))))
            
    if plot and dim == 3:
        test = label2rgb(nuc_mask, image=image, bg_label=0)
        dat = NUCL_PRE[NUCL_PRE['Img Index'] == int(idx)]
        sl = 1
        for im in test:
            plt.figure(figsize=(15,15))
            plt.axis('off')
            plt.imshow(im)
            for centr in dat['Centroid']:
                plt.plot(centr[2],centr[1],'o',color='red',markersize=7,zorder=10)
            #plt.savefig("..//tempp//id70_slice" + str(sl) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=200)
            plt.show()  
            sl += 1 
    
    # Save figure
    if plot:
        if save != False:
            plt.savefig(os.getcwd() + str("\\Datasets\\") + str(name) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=500)
        plt.show()
        
#         sl = 1
#         for im in test:
#             plt.figure(figsize=(15,15))
#             plt.axis('off')
#             plt.imshow(im)
#             #plt.savefig("..//tempp//id70_slice" + str(sl) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=200)
#             plt.show()  
#             sl += 1
        
        
    return NUCL_PRE


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
        nuclei_segmentation(row['Path'],row['Image'],row['Name'],dir_nucldec, dir_masks,algorithm,[algorithm_specs[0][otsu_count],algorithm_specs[1]])
        
        if type(algorithm_specs[0]) == list and row['Name'].split('_')[-1] == 'ch00.tif':
            otsu_count +=1

    # Nuclei preprocessing
    for index,row in NUCL_df.iterrows():
        NUCL_PRE = nuclei_preprocessing(row['Path'],row['Image'],row['Name'],dir_masks,plot,save)
        
    return NUCL_PRE