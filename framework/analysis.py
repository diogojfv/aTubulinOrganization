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

#from framework.Functions import  cv2toski,pylsdtoski,polar_to_cartesian, truncate_colormap, plot_hist, plot_pie, remove_not1D, quantitative_analysis,hist_bin,hist_lim,create_separate_DFs,branch,graphAnalysis
from framework.importing import *
from framework.preprocessingCYTO import *
from framework.preprocessingNUCL import *
from framework.processing import *
import statannot

#from fractal_dimension import fractal_dimension
#from fractal_analysis_fxns import boxcount,boxcount_grayscale,fractal_dimension,fractal_dimension_grayscale,fractal_dimension_grayscale_DBC

def plot_barplot(data):
    # SHOULD BE CHANGED
    cols     = [x for x in data.columns if x.startswith("LSF2D") or x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]
    #feats_1D = [x for x in df.columns if x.startswith("LSF2D")]
    #feats    = [x for x in df.columns if x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]

    # 4
    colors   = ["#2ECC71","#FFA500","#E74C3C","#BC544B"]
    labels   = ['WT','NP','P1','P2']
    pairs    = [ (('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394')), (('Dup41_46', 'Del38_46')), (('Dup41_46', 'Mut394')),]  

    # 6
    #     colors   = ["#2ECC71","#DECF77","#5AB7BD","#FFA500","#E74C3C","#BC544B",]
    #     labels   = ['WT','No transfection','Mock','Dup41_46','Del38_46','Mut394']
    #     pairs    = [(('WT', 'No transfection')),(('WT', 'Mock')),(('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394'))] 
    


    for f in cols:
        try:
            data[f] = data[f].astype(float)
        except: 
            continue

        print(ResultsDF.groupby(['Label']).describe()[[(f,'mean'),(f,'std')]])



        fig,ax = plt.subplots()
        sns.set_theme(style="white")
        # 4
        sns.barplot(x="Label", y=f, data=data,order=['WT','Dup41_46','Del38_46','Mut394'],capsize=.1,errorbar=('ci', 95),edgecolor=colors,fill=False,linewidth=2)
        

        
        #cis = [container.get_yerr()[1]/2 for container in containers]
        
        ax.set_xticks(ax.get_xticks(),labels,font='arial',color='k')
        ax.set_yticks(ax.get_yticks(),font='arial',color='k')
        



        patches = ax.patches
        lines_per_err = 3

        for i, line in enumerate(ax.get_lines()):
            newcolor = patches[i // lines_per_err].get_edgecolor()
            line.set_color(newcolor)

        ax.set_xlabel(None)
        plt.rcParams['axes.linewidth'] = 0.75  
        statannot.add_stat_annotation(plt.gca(),data=data,x="Label",y=f, order=['WT','Dup41_46','Del38_46','Mut394'],box_pairs=pairs,test='t-test_ind',text_format="star",loc="outside",fontsize='small',comparisons_correction=None,color='k',linewidth=0.75)

        sns.despine(left=True)
        plt.grid(alpha=0.2,axis='y')




        plt.show()
        
        
        
def plot_generalized(LinesDF):
    labels  = list(np.unique(LinesDF['Label']))
    try:
        labels.remove('Synthetic')
    except:
        pass
    feats   = list(LinesDF.columns[6:])
    colors  = ['#3498DB','#E74C3C','#95A5A6','#ABE6FF','#CA6F1E','#2ECC71']
    markers = ["-o","-v","-^","-x","-+","-s"]
    cmap = pltc.Reds
    #cmaps = [pltc.Blues, pltc.Reds, pltc.Greys, truncate_colormap(pltc.BuGn, 0, 0.5, 300), truncate_colormap(pltc.YlOrBr, 0, 0.8, 300), pltc.Greens]

    global feat
    for feat in feats: 
        # HISTOGRAM
        plt.figure(figsize=(8,5))
        
        # Boxplot data
        global box
        box = pd.DataFrame(columns = labels) 
        for label in labels:
            # Isolate subdataset 
            dat = LinesDF[LinesDF['Label'] == label]    # For histograms
            
            # Concatenate data
            global feat_concat
            feat_concat = []
            for x in dat[feat]:
                if type(eval(x)) == list:
                    feat_concat += eval(x)
                else:
                    feat_concat += [eval(x)]
            feat_concat = pd.Series(feat_concat)
            feat_concat = feat_concat[~feat_concat.isnull()]
            
            # Add to aux
            box[label] = feat_concat
            
            # Get histogram
            #histog = np.histogram(np.array(feat_concat[~np.isnan(feat_concat)]),bins=10)
            if feat == 'LSF2D:Angles':
                histog = np.histogram(np.array(feat_concat),bins=10,density=True)
            
            elif feat == 'LSF2D:Distances to Centroid':
                histog = np.histogram(np.array(feat_concat),bins=np.arange(0, 100 + 5, 5),density=True)
            elif feat == 'LSF2D:Triangle Areas':
                histog = np.histogram(np.array(feat_concat),bins=np.arange(0, 250 + 5, 5),density=True)
            elif feat == 'LSF2D:Line Lengths':
                histog = np.histogram(np.array(feat_concat),bins=np.arange(0, 15 + 1, 1),density=True)
            else:
                histog = np.histogram(np.array(feat_concat),bins=np.arange(0, 100 + 5, 5),density=True)
            
            # Plot histogram
            plt.plot(histog[1][:-1],histog[0],markers[labels.index(label)],color=colors[labels.index(label)],linewidth=4,label = label,alpha=0.7)

        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylabel('Probability',fontsize=16)
        if feats.index(feat) == 0:
            plt.xlabel('Angles (º)')
        if feats.index(feat) == 1:
            plt.xlabel('Pixels')
        if feats.index(feat) == 2:
            plt.xlabel('Pixels')
        plt.title('Histogram - ' + feat,fontsize=20)

        #plt.title('Probability Density Function - ' + feat,fontsize=20)
        #plt.savefig(".//ResultsGenAnalysis//Hough26Mar//" + str(feat) + "_Histogram.png",format='png',transparent=True,bbox_inches='tight')
        plt.show()
        
        # BOXPLOT
        #plt.figure(figsize=(10,5))
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.title('Boxplot - ' + feat,fontsize=20)
        
        boxhelp = []
        for label in labels:
            temp = box[label]
            boxhelp += [temp[~np.isnan(temp)]]
        bplot = ax.boxplot(np.transpose(boxhelp),patch_artist=True,notch=True,labels=labels)
            
        for patch,color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        plt.ylabel(feat,fontsize=16)
        ax.yaxis.grid(True)
        #plt.savefig(".//ResultsGenAnalysis//Hough26Mar//" + str(feat) + "_Boxplot.png",format='png',transparent=True,bbox_inches='tight')
        plt.show()
        
        
    # HEATMAP 
    i = 0
    global hists,angss,distss
    hists = []
    angss = []
    distss = []
    for label in labels:
        fig = plt.figure(figsize=(5,5)) 
        ax = fig.add_subplot(111, polar=True)
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        
        dat = LinesDF[LinesDF['Label'] == label]
        
        feat_concat_ang = []
        for x in dat['Angles']:
            if type(x) == list:
                feat_concat_ang += x
            else:
                feat_concat_ang += [x]
        feat_concat_ang = pd.Series(feat_concat_ang)
        
        feat_concat_dists = []
        for x in dat['Distances to Centroid']:
            if type(x) == list:
                feat_concat_dists += x
            else:
                feat_concat_dists += [x]
        feat_concat_dists = pd.Series(feat_concat_dists)
        
        print(max(feat_concat_dists))
        
        global histog_
        histog_, angs, dists = np.histogram2d(feat_concat_ang,feat_concat_dists, bins=[np.linspace(0,90,91), np.arange(0, 130 + 5, 1)],density=True)
        histog_, angs, dists = np.histogram2d(feat_concat_ang,feat_concat_dists, bins=[np.linspace(0,90,91), np.arange(0, 130 + 5, 1)],density=True)
        hists += [[histog_]]
        angss += [[angs]]
        distss += [[dists]]
        #print(histog_.max()) 
        print(np.max(histog_))
        
        angs, dists = np.meshgrid(np.linspace(0,np.pi/2,91), np.arange(0, 130 + 5, 1))

        #ax.pcolormesh(angs, dists, histog_, label = label,alpha=0.3,color=colors[labels.index(label)])
        image = ax.pcolormesh(angs, dists, histog_.T,cmap=cmap,vmax=0.0012)

        
        #-----
        #hi, xedges, yedges, image = plt.hist2d(feat_concat_ang,feat_concat_dists,bins=[np.linspace(0,90,10), np.arange(0, 50 + 5, 5)],cmap=cmap,alpha=0.7,density=True,vmax=0.0004)   
        #ax.set_xticks(xedges)
        #ax.set_yticks(np.linspace(0,yedges[-1],10,endpoint=True,dtype=int))

        #ax.set_facecolor(cmap(0))
        plt.ylabel('Angle (º)')
        plt.xlabel('Distance to Centroid (pixels)')
        plt.title(str(label)+' - Line Segment Angles vs. Distances to Centroid')
#         cbar = fig.colorbar(image, ax=ax)
#         cbar.set_label("Relative Frequency")
        #plt.savefig(".//ResultsGenAnalysis//Hough26Mar//" + str(label) + "_HEATMAP.png",format='png',transparent=True,bbox_inches='tight')
        plt.show()  
        
        i = i+1
        
        
def stackedbarplots(ResultsDF):
    labels = ['WT','Dup41_46','Del38_46','Mut394']
    colors = ['#6E7E85','#B7CECE','#BBBAC6','#744253']
    f1,f2,f3,f4 = 'SKNW:Ratio of Endpoint-to-endpoint (isolated branch)','SKNW:Ratio of Junction-to-endpoints','SKNW:Ratio of Junction-to-junctions','SKNW:Ratio of Isolated cycles'
    l1 = 'endpoint-to-endpoint'
    l2 = 'junction-to-endpoint'
    l3 = 'junction-to-junction'
    l4 = 'isolated cycle'
    means_1 = []
    means_2 = []
    means_3 = []
    means_4 = []
    stds_1  = []
    stds_2  = []
    stds_3  = []
    stds_4  = []
    for l in labels:
        aux  = ResultsDF[ResultsDF['Label'] == l]
        
        means_1 += [np.mean(aux[f1])]
        means_2 += [np.mean(aux[f2])]
        means_3 += [np.mean(aux[f3])]
        means_4 += [np.mean(aux[f4])]
        stds_1  += [np.std(aux[f1])]
        stds_2  += [np.std(aux[f2])]
        stds_3  += [np.std(aux[f3])]
        stds_4  += [np.std(aux[f4])]
    

    width = 0.35       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()
    ax.bar(labels, means_1, width, yerr=stds_1, label=l1,color=colors[0])
    ax.bar(labels, means_2, width, yerr=stds_2, bottom=means_1,label=l2,color=colors[1])
    ax.bar(labels, means_3, width, yerr=stds_3, bottom=np.array(means_1) + np.array(means_2),label=l3,color=colors[2])
    ax.bar(labels, means_4, width, yerr=stds_4, bottom=np.array(means_1) + np.array(means_2) + np.array(means_3),label=l4,color=colors[3])

    ax.set_xticks(labels,font='arial')
    ax.set_ylabel('% of branch connectivity type',font='arial')
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1],[0,20,40,60,80,100])
    #ax.set_title('Scores by group and gender')
    leg = ax.legend(framealpha=1,loc=(1.04,0.7),labelcolor='linecolor',edgecolor='k',fancybox=False)
    plt.setp(leg.texts, family='arial')
    leg.get_frame().set_linewidth(1.4)
    plt.savefig(".//output.png",format='png',transparent=True,bbox_inches='tight',dpi=300)
    plt.show()
    




        

def plot_displot(df,ecdf,save):
    global LSF,DCF,DNF,FULL,data,c,ax
    LSF,DCF,DNF,SKNW,OTHERS,FULL = create_separate_DFs(df)
    data = pd.concat([df[df.columns[:7]] , FULL],axis=1)
    
    cols     = [x for x in data.columns if x.startswith("LSF2D") or x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]
    
    # 4
    colors   = ["#2ECC71","#FFA500","#E74C3C","#BC544B"]
    labels   = ['WT','Dup41_46','Del38_46','Mut394']
    pairs    = [(('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394'))]  
    
    # 6
#     colors   = ["#2ECC71","#FFA500","#E74C3C","#BC544B","#5AB7BD","#DECF77"]
#     labels   = ['WT','Dup41_46','Del38_46','Mut394','Mock','No transfection']
#     pairs    = [(('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394')),(('WT', 'Mock')),(('WT', 'No transfection'))] 
    
    for f in cols:
        #fig,ax = plt.subplots(figsize=(8,5))
        sns.set_theme(style="white")
        
        try:
            fig,ax = plt.subplots()
            
            sns.kdeplot(data=data, x=f,hue="Label",hue_order=['WT','Dup41_46','Del38_46','Mut394'],palette=colors,linewidth=3,bw_adjust=.8,common_norm=False,ax=ax)
            

            # AX2
            if ecdf:
                ax2 = ax.twinx()  
                sns.set_theme(style="white")
                sns.ecdfplot(data=data,x=f,hue="Label",hue_order=['WT','Dup41_46','Del38_46','Mut394'],palette=colors,linewidth=1,ax=ax2)
                ax2.legend([],[], frameon=False)

            plt.grid(alpha=0.2)


        except:
            print('Triggered exception')
            data[f] = data[f].astype('float64')
            ax = sns.kdeplot(data=data, x=f,hue="Label",hue_order=['WT','Dup41_46','Del38_46','Mut394'],palette=colors,linewidth=3,bw_adjust=.8,common_norm=False)


        #statannot.add_stat_annotation(plt.gca(),data=data,x="Label",y=f,order=['WT','Dup41_46','Del38_46','Mut394'],box_pairs=pairs,test='t-test_ind',text_format="star",loc="outside",fontsize='small',comparisons_correction=None)
        sns.despine(left=True)
        if save:
            plt.savefig(".//ResultsGenAnalysis_9June//" + str(f.split(':')[0]) + "-"  + str(f.split(':')[1]) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=200)
            print('o')
        
        plt.show()
        

def plot_violinplot(df,save):
    global LSF,DCF,DNF,FULL,data
    LSF,DCF,DNF,SKNW,OTHERS,FULL = create_separate_DFs(df)
    data = pd.concat([df[df.columns[:7]] , FULL],axis=1)
    
    cols     = [x for x in FULL.columns if x.startswith("LSF2D") or x.startswith("LSF1D") or x.startswith("DCF") or x.startswith("DNF") or x.startswith("SKNW") or x.startswith("OTHERS")]
    colors   = ["#2ECC71","#FFA500","#E74C3C","#BC544B"]
    labels   = ['WT','Dup41_46','Del38_46','Mut394']
    pairs    = [(('WT', 'Dup41_46')), (('WT', 'Del38_46')), (('WT', 'Mut394'))]  
    
    for f in cols:
        sns.set_theme(style="whitegrid")
        try:
            sns.violinplot(x="Label", y=f, data=data,order=['WT','Dup41_46','Del38_46','Mut394'],palette=colors,inner="quartiles",scale_hue=True)
        except:
            data[f] = data[f].astype('float64')
            sns.violinplot(x="Label", y=f, data=data,order=['WT','Dup41_46','Del38_46','Mut394'],palette=colors,inner="quartiles",scale_hue=True)
        statannot.add_stat_annotation(plt.gca(),data=data,x="Label",y=f,order=['WT','Dup41_46','Del38_46','Mut394'],box_pairs=pairs,test='t-test_ind',text_format="star",loc="outside",fontsize='small',comparisons_correction=None)
        sns.despine(left=True)
        if save:
            plt.savefig(".//ResultsGenAnalysis_9June//" + str(f.split(':')[0]) + "-"  + str(f.split(':')[1]) + ".png",format='png',transparent=True,bbox_inches='tight',dpi=200)
            print('o')
        
        plt.show()
        
        
        
        
        
        
###### OTHER PLOTS (PRECISA DE REVISÃO)
# Quantitative Analysis
def quantitative_analysis(ImageLinesDF):
    feats = list(ImageLinesDF.columns[3:])
    cmap = pltc.Reds
    global new_cmap
    new_cmap = truncate_colormap(cmap, 0.3, 1, 300)
    
    #histog = np.histogram(np.array(feat_concat[~np.isnan(feat_concat)]),bins=30)
    # plt.plot(histog[1][:-1],histog[0]/np.sum(histog[0]),color=colors[labels.index(label)],linewidth=4,label = label,alpha=0.7)
    flag = False
    global feat
    for feat in feats:

        if feat == 'Angles':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=180)
            #plot_hist(feat=feat,bins=np.linspace(0,90,91))
            plot_hist(ImageLinesDF=ImageLinesDF,feat=feat,bins=hist_bin(feat))

        elif feat == 'Distances to Centroid':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=30)
            plot_hist(ImageLinesDF=ImageLinesDF,feat=feat,bins=hist_bin(feat))
            #aux = []
            #for i in range(len(histog[1])-1):
            #    aux += [np.pi*(histog[1][i+1])**2 - np.pi*(histog[1][i])**2]
            #ax2.plot(histog[1][:-1],(histog[0]/np.trapz(histog[0],x=histog[1][:-1]))/aux,'--',label='uga',alpha=0)
 
        elif feat == 'Triangle Areas':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=30)
            plot_hist(ImageLinesDF=ImageLinesDF,feat=feat,bins=hist_bin(feat))
        
        elif feat == 'Line Lengths':
            #histog = np.histogram(ImageLinesDF[feat][ImageLinesDF[feat].index[0]],bins=30)
            plot_hist(ImageLinesDF=ImageLinesDF,feat=feat,bins=hist_bin(feat))
            
        elif feat == 'Theta':
            plot_hist(ImageLinesDF=ImageLinesDF,feat=feat,bins=hist_bin(feat))
            
        elif feat == 'Angle Difference':
            plot_hist(ImageLinesDF=ImageLinesDF,feat=feat,bins=hist_bin(feat))
            
        elif feat == 'Number of Lines':
            plot_pie(ImageLinesDF=ImageLinesDF,feat=feat,Max=400)
            #flag = True
            
        else:
        #    #print("Fractal Dimension: " + str(ImageLinesDF[feat][ImageLinesDF[feat].index[0]]))
            print(str(feat) + ": " + str(ImageLinesDF.tail(1)[feat][ImageLinesDF.tail(1).index[0]]))
    
    global comb
    comb = combinations(range(len(feats[:6])), 2)
    data = ImageLinesDF.tail(1)
    for f in list(comb):
        fig, ax = plt.subplots()
        f1 = feats[f[0]]
        f2 = feats[f[1]]
        if f1 == 'Angles' or f1 == 'Theta' or f1 == 'Angle Difference':
            bin1 = np.linspace(0,90,10)
        else:
            bin1 = 15
        if f2 == 'Angles' or f2 == 'Theta' or f2 == 'Angle Difference':
            bin2 = np.linspace(0,90,10)
        else:
            bin2 = 15
        
        #plt.figure(figsize=(15,15))
        hi, xedges, yedges, image = plt.hist2d(data[feats[f[0]]][data.index[0]],data[feats[f[1]]][data.index[0]],bins=[bin1,bin2],cmap=cmap,alpha=0.7)   
        ax.set_xticks(xedges)
        ax.set_yticks(np.linspace(0,yedges[-1],10,endpoint=True,dtype=int))
        ax.set_facecolor(cmap(0))
        plt.xlabel(feats[f[0]])
        plt.ylabel(feats[f[1]])
        #plt.title('Line Segment Angles vs. Distances to Centroid')
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Absolute Frequency")
        plt.show()
        
#     # HEATMAP 1
#     fig, ax = plt.subplots()
#     data = ImageLinesDF.tail(1)
#     hi, xedges, yedges, image = plt.hist2d(data['Angles'][data.index[0]],data['Distances to Centroid'][data.index[0]],bins=[np.linspace(0,90,10), np.arange(0, 280 + 5, 5)],cmap=cmap,alpha=0.7)   
#     ax.set_facecolor(cmap(0))
#     plt.xlabel('Angle (º)')
#     plt.ylabel('Distance to Centroid (pixels)')
#     plt.title('Line Segment Angles vs. Distances to Centroid')
#     cbar = fig.colorbar(image, ax=ax)
#     cbar.set_label("Absolute Frequency")
#     plt.show()
    
#     # HEATMAP 2
#     fig, ax = plt.subplots()
#     data = ImageLinesDF.tail(1)
#     hi, xedges, yedges, image = plt.hist2d(data['Line Lengths'][data.index[0]],data['Distances to Centroid'][data.index[0]],bins=[30,30],cmap=cmap,alpha=0.7)   
#     plt.xlabel('Line Lengths (pixels)')
#     plt.ylabel('Distance to Centroid (pixels)')
#     plt.title('Line Segment Lengths vs. Distances to Centroid')
#     cbar = fig.colorbar(image, ax=ax)
#     cbar.set_label("Absolute Frequency")
#     plt.show()


def plot_hist(ImageLinesDF,feat,bins):
    cmap = pltc.Reds
    global new_cmap
    new_cmap = truncate_colormap(cmap, 0.3, 1, 300)
    
    global data
    data = ImageLinesDF.tail(1)
    global histog,bin_edges
    
#     if feat == 'Distances to Centroid':
#         bins = np.arange(0, 280 + 5, 5)
#     if feat == 'Triangle Areas':
#         bins = np.arange(0, 4600 + 5, 5)
#     if feat == 'Line Lengths':
#         # max(data[feat][data.index[0]])
#         bins = np.arange(0, 220 + 5, 5)
#     if feat == 'Theta':
#         bins = np.arange
#     if feat == 'Angle Difference':
#         bins = np.arange
    
    histo = np.histogram(data[feat][data.index[0]],bins=bins[0])
    
    # AX1
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.set_ylabel('Absolute Frequency')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.tick_params(axis='y', colors=pltc.Reds_r(0))

    
    global colours
    #colours = pltc.Reds(plt.Normalize(0, max(histog[0]))(histog[0]),alpha=0.7)
    #colours = new_cmap(plt.Normalize(0, max(histo[0]))(histo[0]),alpha=0.7)

    #ax1.bar(histog[1][:-1],histog[0],color=colours,zorder=5)
    histog, bin_edges, patches = ax1.hist(data[feat][data.index[0]], bins=bins[0],color='k',alpha=0.7,edgecolor = "white")
    #ax1.color = new_cmap(plt.Normalize(0, max(histog))(histog),alpha=0.7)
    for c, p in zip(histog, patches):
        plt.setp(p, 'facecolor', new_cmap(plt.Normalize(0, max(histog))(c), alpha = 0.7))
    
    # AX2
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Relative Frequency')  
    _ = ax2.hist(data[feat][data.index[0]], bins=bins[0], density=True,color='k',alpha=0.7,fill=False,histtype='step',ls='dashed')
    #ax2.plot(histog[1][:-1],(histog[0]/np.trapz(histog[0],x=histog[1][:-1])),'--',alpha=0,zorder=1)
    plt.grid(alpha=0.3)
    
    if feat == 'Angles':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Angle between Centroid and Line Segment',fontsize=12)
        ax1.set_xticks(bin_edges)
    if feat == 'Distances to Centroid':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Distance between Centroid and Line Segment',fontsize=12)
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        #ax1.set_xticks(np.linspace(0,130,10,endpoint=True,dtype=int))
        
    if feat == 'Triangle Areas':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Triangle Areas between Centroid and Line Segment',fontsize=12)
        #ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        
    if feat == 'Line Lengths':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Line Length',fontsize=12)
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        #ax1.set_xticks(np.linspace(0,170,10,endpoint=True,dtype=int))
        
    if feat == 'Theta':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Line Segment Angle',fontsize=12)
        ax1.set_xticks(bin_edges)
        
    if feat == 'Angle Difference':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Angle Difference (º)',fontsize=12)
        ax1.set_xticks(bin_edges)
    
    ax1.set_xticks(hist_bin(feat)[1])
    ax1.set_xticklabels(hist_bin(feat)[1],rotation=45, ha='right', rotation_mode='anchor')
            
    #plt.savefig(".//CellVSCellAnalysis//" + str(feat) + ".png",format='png',transparent=True,bbox_inches='tight')
    plt.show()
    
#OU

def plot_hist(ImageLinesDF,feat,bins):
    cmap = pltc.Reds
    global new_cmap
    new_cmap = truncate_colormap(cmap, 0.3, 1, 300)
    
    global data
    data = ImageLinesDF.tail(1)
    global histog,bin_edges
    
#     if feat == 'Distances to Centroid':
#         bins = np.arange(0, 280 + 5, 5)
#     if feat == 'Triangle Areas':
#         bins = np.arange(0, 4600 + 5, 5)
#     if feat == 'Line Lengths':
#         # max(data[feat][data.index[0]])
#         bins = np.arange(0, 220 + 5, 5)
#     if feat == 'Theta':
#         bins = np.arange
#     if feat == 'Angle Difference':
#         bins = np.arange
    
    histo = np.histogram(data[feat][data.index[0]],bins=bins)
    
    # AX1
    plt.figure()
    ax1 = plt.subplot(1,1,1)
    ax1.set_ylabel('Absolute Frequency')
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    global colours
    #colours = pltc.Reds(plt.Normalize(0, max(histog[0]))(histog[0]),alpha=0.7)
    #colours = new_cmap(plt.Normalize(0, max(histo[0]))(histo[0]),alpha=0.7)

    #ax1.bar(histog[1][:-1],histog[0],color=colours,zorder=5)
    histog, bin_edges, patches = ax1.hist(data[feat][data.index[0]], bins=bins,color='k',alpha=0.7)
    #ax1.color = new_cmap(plt.Normalize(0, max(histog))(histog),alpha=0.7)
    for c, p in zip(histog, patches):
        plt.setp(p, 'facecolor', new_cmap(plt.Normalize(0, max(histog))(c), alpha = 0.7))
    
    # AX2
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Relative Frequency')  
    _ = ax2.hist(data[feat][data.index[0]], bins=bins, density=True, alpha=0)
    #ax2.plot(histog[1][:-1],(histog[0]/np.trapz(histog[0],x=histog[1][:-1])),'--',alpha=0,zorder=1)
    plt.grid(alpha=0.3)
    
    if feat == 'Angles':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Angle between Centroid and Line Segment',fontsize=12)
        ax1.set_xticks(bin_edges)
    if feat == 'Distances to Centroid':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Distance between Centroid and Line Segment',fontsize=12)
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        #ax1.set_xticks(np.linspace(0,130,10,endpoint=True,dtype=int))
        
    if feat == 'Triangle Areas':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Triangle Areas between Centroid and Line Segment',fontsize=12)
        #ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        
    if feat == 'Line Lengths':
        ax1.set_xlabel('Pixels')
        ax1.set_title('Line Length',fontsize=12)
        ax1.set_xticks(np.linspace(0,bin_edges[-1],10,endpoint=True,dtype=int))
        #ax1.set_xticks(np.linspace(0,170,10,endpoint=True,dtype=int))
        
    if feat == 'Theta':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Line Segment Angle',fontsize=12)
        ax1.set_xticks(bin_edges)
        
    if feat == 'Angle Difference':
        ax1.set_xlabel('Degrees (º)')
        ax1.set_title('Angle Difference (º)',fontsize=12)
        ax1.set_xticks(bin_edges)
    
    #plt.savefig(".//CellVSCellAnalysis//" + str(data.index) + str("_") + str(feat) + ".png",format='png',transparent=True,bbox_inches='tight')
    plt.show()