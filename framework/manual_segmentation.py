import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from roipoly import RoiPoly

def roi_selector(data,img_id,ROIsDF):
    global ROI
    import copy
    print('🔎')
    
    if type(ROIsDF) != pd.core.frame.DataFrame:
        ROIsDF    = pd.DataFrame(columns = ['Name','Index','Label','ROImask'])
       
    ROIsDF_ = copy.deepcopy(ROIsDF)
    
    while 1:
        try:
            plt.close('all')

            # Original Image
            img = data['Image'][img_id]
            mult = np.stack([1.8*data['Image'][12]/np.max(data['Image'][12]),1.8*data['Image'][11]/np.max(data['Image'][11]),1*data['Image'][10]/np.max(data['Image'][10])],axis=2)


            # Select ROI QT
            #%matplotlib qt
            fig,ax = plt.subplots(figsize=(30,30))
            plt.imshow(mult)
            plt.axis('off')
            # Plot Contours
            #plot_nuclei_contours(CentroidsDF=Centroids,imgIndex=img_id,ax=ax)
            # Define ROI

            ROI = RoiPoly(color='r')
            #ROI.display_roi()
            #global mask,roi_coordinates
            #roi_coordinates = ROI.get_roi_coordinates()
            mask = ROI.get_mask(img)

            # Save ROI
            new = pd.DataFrame(data = {'Name': [data['Name'][img_id]],'Index': [img_id], 'Label': [data['Label'][img_id]], 'ROImask': [mask]})
            ROIsDF_ = pd.concat([ROIsDF_, new], axis=0,ignore_index=True)


            plt.show()
        except Exception as e:
            print('Window closed')
            print(e)
            return ROIsDF_
            
        
        
        
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
