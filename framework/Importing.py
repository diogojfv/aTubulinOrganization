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


# Plotting
import matplotlib.pyplot as plt
import matplotlib.cm as pltc
import matplotlib.colors as colors


import tifffile as tiffio
import scipy.misc


def label_tubulin(name):
    number = int(name.split('_')[1])
    
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
    return number,label

def label_tubulin3D(name):
    number = int(name.split('_')[0])
    
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
    return number,label

def label_SPOCC(name):
    hour = name.split("_")[1]
    return hour

def label_soraia(name):
    number = int(name.split("_")[1])
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
    return number,label


def init_import(folder, options, denominator):
    res = OrderedDict()
    
    # RGB NUCLEI + TUBULIN IMAGES
    if "RGB" in options: 
        OriginalDF = pd.DataFrame(columns=['Name','Label','Image'])
        for img in os.listdir(folder + "\\RGB"):
            path       = folder + "\\RGB\\" + img
            image      = cv2.imread(path,cv2.IMREAD_COLOR)  # Size: (1040, 1388, 3)
            img_id     = int(img.split('_')[0])+1 # add 1 to keep the same id as deconvoluted imgs
            new        = pd.DataFrame(data={'Name': [img], 'Label': [denominator(img)], 'Image': [image]}, index = [img])
            OriginalDF = pd.concat([OriginalDF, new], axis=0,ignore_index=False)
        res["RGB"] = OriginalDF
        print(">>> [RGB] added.")
        
    # GRAY-SCALE (2D) DECONVOLUTED CYTOSKELETON IMAGES
    if "CYTO" in options:
        DeconvDF = pd.DataFrame(columns=['Path','Name','Label','Image'])
        for img in os.listdir(folder + "\\CYTO"):
            path     = folder + "\\CYTO\\" + img
            image    = cv2.imread(path,-1)  # Size: (1040,1388)
            new      = pd.DataFrame(data={'Path': [path],'Name': [img], 'Label': [denominator(img)[1]], 'Image': [image]}, index = [denominator(img)[0]])
            DeconvDF = pd.concat([DeconvDF, new], axis=0,ignore_index=False)
        res["CYTO"] = DeconvDF
        print(">>> [CYTO] added.")
        
    # GRAY-SCALE (2D) DECONVOLUTED NUCLEI IMAGES
    if "NUCL" in options:
        NucleiDeconvDF = pd.DataFrame(columns=['Path','Name','Label','Image'])
        for img in os.listdir(folder + "\\NUCL"):
            path           = folder + "\\NUCL\\" + img
            image          = cv2.imread(path,-1)
            new            = pd.DataFrame(data={'Path': [path], 'Name': [img], 'Label': [denominator(img)[1]], 'Image': [image]}, index = [denominator(img)[0]])
            NucleiDeconvDF = pd.concat([NucleiDeconvDF, new], axis=0,ignore_index=False)
        res["NUCL"] = NucleiDeconvDF
        print(">>> [NUCL] added.")
        
    # 3D GRAY-SCALE SEPARATED RGB CHANNELS
    if "CYTO3D" in options:
        import tifffile as tiffio
        TenDF = pd.DataFrame(columns=['Path','Name','Label','Image'])
        for img in os.listdir(folder + "\\CYTO3D"):
            path    = folder + "\\CYTO3D\\" + img
            image   = tiffio.imread(path)
            new      = pd.DataFrame(data={'Path': [path],'Name': [img], 'Label': [denominator(img)[1]], 'Image': [image]}, index = [denominator(img)[0]])
            TenDF   = pd.concat([TenDF, new], axis=0,ignore_index=False)
        res["CYTO3D"] = TenDF
        print(">>> [CYTO3D] added.")
        
    if "NUCL3D" in options:
        import tifffile as tiffio
        TenDF = pd.DataFrame(columns=['Path','Name','Label','Image'])
        for img in os.listdir(folder + "\\NUCL3D"):
            path    = folder + "\\NUCL3D\\" + img
            image   = tiffio.imread(path)
            new      = pd.DataFrame(data={'Path': [path],'Name': [img], 'Label': [denominator(img)[1]], 'Image': [image]}, index = [denominator(img)[0]])
            TenDF   = pd.concat([TenDF, new], axis=0,ignore_index=False)
        res["NUCL3D"] = TenDF
        print(">>> [NUCL3D] added.")
        
    return res



def widget_import_dataset():
    import os
    import ipywidgets as widgets
    from IPython.display import display

    datasets_path = "..\\Datasets"
    folder_names = [name for name in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, name))]
    dropdown = widgets.Dropdown(options=folder_names, description='Select Folder:', disabled=False)

    def imp_datasets(change):
        selected_folder = change.new

        print(f"Selected folder: {selected_folder}")
        if selected_folder == "Set 1-a-tubulin_Sofia":
            folder    = os.path.dirname(os.getcwd()) + "\\Datasets\\Set 1-a-tubulin_Sofia"
            options   = ["CYTO","NUCL"]
            denominator = label_tubulin

        if selected_folder == "Set 3D":
            folder      = os.path.dirname(os.getcwd()) + "\\Datasets\\Set 3D"
            options     = ["CYTO3D","NUCL3D"]
            denominator = label_tubulin

        if selected_folder == "Soraia":
            folder      = os.path.dirname(os.getcwd()) + "\\Datasets\\Soraia"
            options     = ["CYTO","NUCL"]
            denominator = label_soraia

        if selected_folder == "SPOCC":
            folder      = os.path.dirname(os.getcwd()) + "\\Datasets\\SPOCC2022"
            options     = ["CYTO"]
            denominator = label_SPOCC


        data = init_import(folder,options,denominator)
        
        return data

    dropdown.observe(imp_datasets, names='value')
    display(dropdown)


