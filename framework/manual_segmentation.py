import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly
import cv2


def retrieve_mask(coordinates, dims):
    mask = np.zeros(dims)
    mask[coordinates] = 1
    return mask

def roi_selector(data,img_id,ROIsDF):
    global ROI
    import copy
    print('🔎')
    
    if type(ROIsDF) != pd.core.frame.DataFrame:
        ROIsDF    = pd.DataFrame(columns = ['Name','Index','Label','Image Size','ROImask'])
       
    ROIsDF_ = copy.deepcopy(ROIsDF)
    
    # ROI loop
    try:
        while 1:
            # Handle figure
            #plt.close('all')
            fig,ax = plt.subplots(figsize=(30,30))
            plt.imshow(data['CYTO'].loc[img_id]['Image'],cmap='gray')
            plt.axis('off')
            
            # Handle variables
            ROI     = RoiPoly(fig=fig, ax=ax, color='r')
            mask    = ROI.get_mask(data['CYTO'].loc[img_id]['Image'])
            non_zero_indices = np.where((mask*1)!=0)
            #mask_coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
            new     = pd.DataFrame(data = {'Name': [data['CYTO']['Name'][img_id]],'Index': [img_id], 'Label': [data['CYTO']['Label'][img_id]], 'Image Size': [mask.shape] ,'ROImask': [non_zero_indices]})
            ROIsDF_ = pd.concat([ROIsDF_, new], axis=0,ignore_index=True)

    except Exception as e:
        print('Window closed')
        print(e)
        return ROIsDF_
    
def plot_selected_ROIs(ROIs2,img_id):
    df = ROIs2[ROIs2['Index']==img_id]

    i = 0
    for index,row in df.iterrows():
        if i == 0:
            auxx = retrieve_mask(row['ROImask'],row['Image Size'])
            i = 1
        else:
            auxx = auxx + retrieve_mask(row['ROImask'],row['Image Size'])

    fig,ax = plt.subplots(figsize=(10,10))
    plt.imshow(auxx,cmap='viridis')
    plt.axis('off')

    plt.show()

# def roi_selector(data,img_id,ROIsDF):
#     global ROI
#     import copy
#     print('🔎')
    
#     if type(ROIsDF) != pd.core.frame.DataFrame:
#         ROIsDF    = pd.DataFrame(columns = ['Name','Index','Label','ROImask'])
       
#     ROIsDF_ = copy.deepcopy(ROIsDF)
    
#     # Get image
#     if '3D' in data.keys():
#         flag = '3D'
#         mult = np.stack([1.8*data['3D']['Image'][12]/np.max(data['3D']['Image'][12]),1.8*data['3D']['Image'][11]/np.max(data['3D']['Image'][11]),1*data['3D']['Image'][10]/np.max(data['3D']['Image'][10])],axis=2)
    
#     if 'CYTO_DECONV' in data.keys():
#         flag = 'RGB'
#         img        = data['RGB']['Image'][img_id]
#         tmp        = copy.deepcopy(img)
#         tmp[:,:,0] = 0
#         grey       = cv2.cvtColor(tmp,cv2.COLOR_RGB2GRAY)

#         #mult = np.stack([1.3*(grey / np.max(grey)),1.3*sk*(grey / np.max(grey)), np.zeros_like(sk)],axis=2)
#         #mult = np.stack([0.9*(grey / np.max(grey)),sk*(grey / np.max(grey)), 0.5 * (OriginalDF['Image'][img_id][:,:,0] / np.max(OriginalDF['Image'][img_id][:,:,0]))],axis=2)
#         #mult = np.stack([1.5*(grey / np.max(grey)),sk, 0.2 * (data['RGB']['Image'][img_id][:,:,0] / np.max(data['RGB']['Image'][img_id][:,:,0]))],axis=2)
#         mult = img

        
        
    
#     while 1:
#         try:
#             plt.close('all')

#             # Select ROI QT
#             #%matplotlib qt
#             fig,ax = plt.subplots(figsize=(30,30))
#             plt.imshow(mult)
#             plt.axis('off')
#             # Plot Contours
#             #plot_nuclei_contours(CentroidsDF=Centroids,imgIndex=img_id,ax=ax)
#             # Define ROI

#             ROI = RoiPoly(color='r')
#             #ROI.display_roi()
#             #global mask,roi_coordinates
#             #roi_coordinates = ROI.get_roi_coordinates()
#             mask = ROI.get_mask(img)

#             # Save ROI
#             new = pd.DataFrame(data = {'Name': [data[flag]['Name'][img_id]],'Index': [img_id], 'Label': [data[flag]['Label'][img_id]], 'ROImask': [mask]})
#             ROIsDF_ = pd.concat([ROIsDF_, new], axis=0,ignore_index=True)


#             plt.show()
#         except Exception as e:
#             print('Window closed')
#             print(e)
#             return ROIsDF_
            
        
        
        
##############################################################
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon
# from matplotlib.widgets import Button
# from roipoly import RoiPoly


# class ROISelector:
#     def __init__(self, img, img_id, df):
#         self.img = img
#         self.img_id = img_id
#         self.df = df
#         self.fig, self.ax = plt.subplots(figsize=(10, 10))
#         self.roi = None
#         self.roi_poly = None
#         self.polygons = []
#         self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
#         self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

#     def on_press(self, event):
#         if event.button == 1:
#             if self.roi_poly is None:
#                 self.roi_poly = RoiPoly(color='r')
#                 self.polygons.append(self.roi_poly)
#             else:
#                 if self.roi_poly.currently_drawing:
#                     self.roi_poly.currently_drawing = False
#                     self.plot_roi()
#                     self.roi_poly = None

#     def on_key(self, event):
#         if event.key == 'enter':
#             self.finish_roi_selection()

#     def plot_roi(self):
#         x, y = self.roi_poly.get_coordinates().T
#         self.ax.plot(x, y, '-r')
#         self.fig.canvas.draw()

#     def finish_roi_selection(self):
#         if self.roi_poly is not None:
#             self.polygons.append(self.roi_poly)

#         self.fig.canvas.mpl_disconnect(self.cid_press)
#         self.fig.canvas.mpl_disconnect(self.cid_key)
#         plt.close()

#         # Store the ROI information in the dataframe
#         ROIs = pd.DataFrame(columns=['Name', 'Index', 'Label', 'ROImask'])
#         for i, polygon in enumerate(self.polygons):
#             mask = polygon.get_mask(self.img)
#             new = pd.DataFrame(data={'Name': [self.df['Name'][self.img_id]],
#                                      'Index': [self.img_id],
#                                      'Label': [self.df['Label'][self.img_id]],
#                                      'ROImask': [mask]})
#             ROIs = ROIs.append(new, ignore_index=True)

#         print("ROIs:", ROIs)

#         # TODO: Add your desired processing or saving logic for the ROIs

#         # Example: Display the selected ROIs
#         fig, ax = plt.subplots(figsize=(10, 10))
#         ax.imshow(self.img)
#         for polygon in self.polygons:
#             x, y = polygon.get_coordinates().T
#             ax.add_patch(Polygon(np.column_stack((x, y)), edgecolor='r', linewidth=2, fill=False))
#         plt.show()


# # Example usage
# img_id = 0  # Index of the image in your dataframe
# img = df['Image'][img_id]
# roi_selector = ROISelector(img, img_id, df)
# plt.imshow(img)
# plt.show()
