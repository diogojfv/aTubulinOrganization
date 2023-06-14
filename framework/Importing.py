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
from roipoly import RoiPoly

# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as pltc
import matplotlib.colors as colors

# Feature Extraction (.py files by Teresa Parreira)
# from CytoSkeletonPropsMorph import CytoSkeletonPropsMorph
# from CytoSkeletonRegionPropsInt import RegionPropsInt
# from FreqAnalysis import FreqAnalysis
# from GLCM import GLCM

# Graph

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

def label_image_soraia(number):
    if number <= 3:
        label = 'WT'
    elif number >= 4 and number <= 6:
        label = 'Mock'
    elif number >= 7 and number <= 9:
        label = '1901'
    elif number >= 10 and number <= 12:
        label = '2245'
    elif number >= 13 and number <= 15:
        label = '2494'
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
        DeconvDF = pd.DataFrame(columns=['Path','Name','Label','Image'])
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
                new      = pd.DataFrame(data={'Path': [path],'Name': [img], 'Label': [label_image_soraia(img_id)], 'Image': [image]}, index = [img_id])
                DeconvDF = pd.concat([DeconvDF, new], axis=0,ignore_index=False)
        res["CYTO_DECONV"] = DeconvDF
        print(">>> [CYTO_DECONV] added.")
        
    # GRAY-SCALE (2D) DECONVOLUTED NUCLEI IMAGES
    if "NUCL_DECONV" in options:
        NucleiDeconvDF = pd.DataFrame(columns=['Path','Name','Label','Image'])
        for img in os.listdir(folder + "\\NUCL_DECONV"):
            path           = folder + "\\NUCL_DECONV\\" + img
            #image          = nuclei_preprocessing(cv2.imread(path,-1))
            image          = cv2.imread(path,-1)
            img_id         = int(img.split('_')[1])
            new            = pd.DataFrame(data={'Path': [path], 'Name': [img], 'Label': [label_image_soraia(img_id)], 'Image': [image]}, index = [img_id])
            NucleiDeconvDF = pd.concat([NucleiDeconvDF, new], axis=0,ignore_index=False)
        res["NUCL_DECONV"] = NucleiDeconvDF
        print(">>> [NUCL_DECONV] added.")
        
    # 3D GRAY-SCALE SEPARATED RGB CHANNELS
    if "3D" in options:
        TenDF = pd.DataFrame(columns=['Path','Name','Channel','Label','Image'])
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
                new     = pd.DataFrame(data={'Path': [path], 'Name': [img], 'Channel': int(img.split('_')[-1][3]), 'Label': [label_image_soraia(img_id)], 'Image': [image]}, index = [img_id])
            except:
                new     = pd.DataFrame(data={'Path': [path], 'Name': [img], 'Channel': int(img.split('_')[-2][3]), 'Label': [label_image_soraia(img_id)], 'Image': [image]}, index = [img_id])
            TenDF   = pd.concat([TenDF, new], axis=0,ignore_index=False)
        res["3D"] = TenDF
        print(">>> [3D] added.")
        
    return res