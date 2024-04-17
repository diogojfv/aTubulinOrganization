import numpy as np
import math
from skimage.measure import regionprops,shannon_entropy,marching_cubes,mesh_surface_area
from skimage.filters import gabor_kernel
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage as ndi
from scipy import signal
from statistics import mean 
from scipy.stats import skew, kurtosis
import cv2

def wavelets(ResultsRow,wavelet,level):
    import pywt
    
    patch = ResultsRow['Patches'][1]
    coeffs = pywt.wavedec2(patch, wavelet, level=level)

    EL  = np.sum(np.square(coeffs[0]))     # Energy of the low-frequency channel
    EH1 = np.sum(np.square(coeffs[1][0]))  # Energy of the First high-frequency channel
    EH2 = np.sum(np.square(coeffs[1][1]))  # Energy of the Second high-frequency channel
    EH3 = np.sum(np.square(coeffs[1][2]))  # Energy of the Third high-frequency channel

    return EL,EH1,EH2,EH3

# ResultsDF[["DCF:Wavelets - EL", "DCF:Wavelets - EH1", "DCF:Wavelets - EH2", "DCF:Wavelets - EH3"]] = ResultsDF.apply(lambda row: wavelets(row, 'haar', 3), axis=1, result_type='expand')



def est_area(ResultsRow):
    from skimage.filters import apply_hysteresis_threshold
    
    img = data['CYTO']['Image'][ResultsRow['Img Index']] * ResultsRow['Mask']
    res = apply_hysteresis_threshold(img,threshold_otsu(img)*0.6,threshold_otsu(img))
    
    #img_nucl_ = ResultsRow['Mask'] * (data['NUCL']['Image'][ResultsRow['Img Index']] / np.max(data['NUCL']['Image'][ResultsRow['Img Index']]))
    #img_nucl  = apply_hysteresis_threshold(img_nucl_,threshold_otsu(img_nucl_)*0.6,threshold_otsu(img_nucl_))
    # Find centroid:
    x_,y_   = np.where((ResultsRow['Mask']*1) != 0)
    imgIndex = ResultsRow['Img Index']
    
    for index,row in CentroidsDF[imgIndex].iterrows():
        if (round(row['Centroid'][0]),round(row['Centroid'][1])) in list(zip(x_,y_)):
            centroid = (row['Centroid'][1],row['Centroid'][0])
            contour = row['Contour']
            idx = index
            break
    try:
        centroid
    except:
        centroid = (0,0)
        contour = CentroidsDF[imgIndex].loc[0]['Contour']
        idx = 0
        print(index,'centroid not found. set to (0,0)')
    

    #img_nucl_ = ResultsRow['Mask'] * (data['NUCL']['Image'][ResultsRow['Img Index']] / np.max(data['NUCL']['Image'][ResultsRow['Img Index']]))
    #img_nucl  = apply_hysteresis_threshold(img_nucl_,threshold_otsu(img_nucl_)*0.6,threshold_otsu(img_nucl_))
    img_nucl = cv2.fillPoly(np.zeros((1040,1388)),pts=CentroidsDF[imgIndex].loc[idx]['Contour'],color=(255,255,255))
    
    
    
    
    rprops_cyto,rprops_nucl = regionprops((res!=0)*1,res),regionprops((img_nucl!=0)*1,img_nucl)
    centro_cyto,centro_nucl = rprops_cyto[0].centroid,rprops_nucl[0].centroid
    w_centro_cyto,w_centro_nucl = rprops_cyto[0].weighted_centroid,rprops_nucl[0].weighted_centroid

    return rprops_cyto[0].area*(0.16125**2),rprops_nucl[0].area*(0.16125**2),rprops_nucl[0].area/rprops_cyto[0].area

# ResultsDF[['DCF:Area (scaled)', 'DNF:Area (scaled)', 'DCF:AreaRatio 6']] = ResultsDF.apply(lambda row: est_area(row), axis=1, result_type='expand')




def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

#I = scipy.misc.imread("sierpinski.png")/256.0
#print("Minkowski–Bouligand dimension (computed): ", fractal_dimension(I))
#print("Haussdorf dimension (theoretical):        ", (np.log(3)/np.log(2)))


def boxcount(Z, k):
    """
    returns a count of squares of size kxk in which there are both colours (black and white), ie. the sum of numbers
    in those squares is not 0 or k^2
    Z: np.array, matrix to be checked, needs to be 2D
    k: int, size of a square
    """
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)  # jumps by powers of 2 squares

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k * k))[0])


def boxcount_grayscale(Z, k):
    """
    find min and max intensity in the box and return their difference
    Z - np.array, array to find difference in intensities in
    k - int, size of a box
    """
    S_min = np.fmin.reduceat(
        np.fmin.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    S_max = np.fmax.reduceat(
        np.fmax.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    return S_max - S_min


def fractal_dimension(Z, threshold=0.9):
    """
    calculate fractal dimension of an object in an array defined to be above certain threshold as a count of squares
    with both black and white pixels for a sequence of square sizes. The dimension is the a coefficient to a poly fit
    to log(count) vs log(size) as defined in the sources.
    :param Z: np.array, must be 2D
    :param threshold: float, a thr to distinguish background from foreground and pick up the shape, originally from
    (0, 1) for a scaled arr but can be any number, generates boolean array
    :return: coefficients to the poly fit, fractal dimension of a shape in the given arr
    """
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def wavelets(ResultsRow,wavelet,level):
    import pywt
    
    patch = ResultsRow['Patches'][1]
    coeffs = pywt.wavedec2(patch, wavelet, level=level)

    EL  = np.sum(np.square(coeffs[0]))     # Energy of the low-frequency channel
    EH1 = np.sum(np.square(coeffs[1][0]))  # Energy of the First high-frequency channel
    EH2 = np.sum(np.square(coeffs[1][1]))  # Energy of the Second high-frequency channel
    EH3 = np.sum(np.square(coeffs[1][2]))  # Energy of the Third high-frequency channel

    return EL,EH1,EH2,EH3

# ResultsDF[["DCF:Wavelets - EL", "DCF:Wavelets - EH1", "DCF:Wavelets - EH2", "DCF:Wavelets - EH3"]] = ResultsDF.apply(lambda row: wavelets(row, 'haar', 3), axis=1, result_type='expand')




def hcorr(idd):
    if idd == 9:
        z = 8.7466670
    if idd == 11: 
        z = 10.2433330
    if idd == 16:
        z = 3.6255560
    if idd == 18:
        z = 3.5422220
    if idd == 20:
        z = 3.5422220
    if idd == 30:
        z = 2.7800000
    if idd == 34:
        z = 5.1844440
    if idd == 36:
        z = 2.8955560
    if idd == 38:
        z = 2.8955560
    if idd == 40:
        z = 4.4233330
    if idd == 42:
        z = 3.8311110
    if idd == 44:
        z = 2.6900000
    if idd == 59:
        z = 4.9966670
    if idd == 63:
        z = 5.8877780
    if idd == 66:
        z = 5.8877780
    if idd == 70:
        z = 6.9944440
    if idd == 72:
        z = 6.6288890
    if idd == 74:
        z = 7.5977780
        
    return z


def fractal_dimension_grayscale(Z):
    """
    works the same as fractal_dimension() just does not look at counts and does not require a binary arr rather is looks
    at intensities (hence can be used for a grayscale image) and returns fractal dimensions D_B and D_M (based on sums
    and means), as described in https://imagej.nih.gov/ij/plugins/fraclac/FLHelp/Glossary.htm#grayscale
    :param Z: np. array, must be 2D
    :return: D_B and D_M fractal dimensions based on polyfit to log(sum) or log(mean) resp. vs log(sizes)
    """
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    i_difference = []
    for size in sizes:
        i_difference.append(boxcount_grayscale(Z, size))

    # D_B
    d_b = [np.sum(x) for x in i_difference]

    # D_M
    d_m = [np.mean(x) for x in i_difference]

    # Fit the successive log(sizes) with log (sum)
    coeffs_db = np.polyfit(np.log(sizes), np.log(d_b), 1)
    # Fit the successive log(sizes) with log (mean)
    coeffs_dm = np.polyfit(np.log(sizes), np.log(d_m), 1)

    return -coeffs_db[0], -coeffs_dm[0]


def fractal_dimension_grayscale_DBC(Z):
    """
    Differential box counting method with implementation of appropriate box height selection.
    :param Z: 2D np.array
    :return: fd for a grayscale image
    """
    # Only for 2d image
    assert (len(Z.shape) == 2)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # get the mean and standard deviation of the image intensity to rescale r
    mu = np.mean(Z)
    sigma = np.std(Z)

    # total number of gray levels, used for computing s'
    G = len(np.unique(Z))

    # scaled scale of each block, r=size(s)
    # TODO -- when to rescale, what should a be
    # when to rescale -- either always or when the pixels in the selected box don't fall in +- 1 std, so far always
    a = 1
    r_prime = sizes / (1 + 2 * a * sigma)

    # Actual box counting with decreasing size
    i_difference = []
    for size in sizes:
        # rescale
        n_r = np.ceil(boxcount_grayscale(Z, size) / r_prime)
        # if max==min per the box, replace the 0 result with 1
        n_r[n_r == 0] = 1
        i_difference.append(n_r)

    # contribution from all boxes
    N_r = [np.sum(x) for x in i_difference]

    # Fit the successive log(sizes) with log (sum)
    coeffs = np.polyfit(np.log(sizes), np.log(N_r), 1)

    return -coeffs[0]


def getvoxelsize(folder):
    import tifffile as tf
    with tf.TiffFile(folder) as tif:
        ij_metadata = tif.imagej_metadata
        num_pixels_x, units_x = tif.pages[0].tags['XResolution'].value
        xres = units_x / num_pixels_x
        num_pixels_y, units_y = tif.pages[0].tags['YResolution'].value
        yres = units_y / num_pixels_y

    if ij_metadata is not None:
        try:
            zres = ij_metadata['spacing']
        except:
            #print('Image is 2D. Using default voxel height (z = 1)')
            return 1,xres,yres
    else:
        print('Using default voxel height (z = 1)')
        zres = 1

    return zres,xres,yres

def getAAI(patch):
    """
    Calculate Average Actin Intensity:
        - patch: image of skeleton with intensities
    """
    
    from scipy.signal import argrelextrema
    
    aga       = np.histogram(patch,bins=256)
    local_min = argrelextrema(aga[0], np.less)[0]
    AAI       = np.sum(aga[0][local_min[0]:local_min[-1]] * aga[1][local_min[0]:local_min[-1]]) / len(np.where(patch != 0)[0])
    
    return AAI


class processingDCF:
    
    def __init__(self, img, mask, skel, listfeats, resolution, original_folder):
        
        # CLASS ATTRIBUTES
        self.img = img
        self.img_8bit = (img*255).astype(np.uint8)
        self.bin_img = mask.astype(np.uint8) # OU self.bin_img = (img!=0)*1
        self.listfeats = listfeats
        self.original_folder = original_folder
        self.dim = len(img.shape)
        if type(skel) == np.ndarray:
            self.skel = skel if skel is not None else (print('erro') or None)
        else:
            self.skel = None
            
        # DEFINE VOXEL SPACING
        if self.dim == 3:
            self.spacing    = (getvoxelsize(original_folder)[0],getvoxelsize(original_folder)[1],getvoxelsize(original_folder)[2])
            self.voxel_size = getvoxelsize(original_folder)[0] * getvoxelsize(original_folder)[1] * getvoxelsize(original_folder)[2]
        if self.dim == 2 and resolution == None:
            siz = getvoxelsize(original_folder)[0],getvoxelsize(original_folder)[1],getvoxelsize(original_folder)[2]
            self.spacing    = (siz[1],siz[2])
            self.voxel_size = siz[1] * siz[2]
        if self.dim == 2 and resolution != None:
            self.spacing    = (resolution[1],resolution[2])
            self.voxel_size = resolution[1] * resolution[2]
        
        # --- INTENSITY
        self.n_level     = (np.max(img) - np.min(img)) + 1
        self.level_min   = 1
        self.level_max   = 256
        self.imgzero     = img[img!=0]
        hist,bin_edges   = np.histogram(self.imgzero, self.level_max,[self.level_min,self.level_max])
        self.hist        = np.array(hist)
        self.bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # --- GLCM
        self.glcm   = graycomatrix(self.img_8bit, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed = True) #or self.img 
        matrix      = np.sum(self.glcm, axis = 3) / 4 #makes the matrix invariant as it sums all the elements for different angles
        self.matrix = np.ndarray((256,256,1,1)) #keeps it 4-dimensional
        self.matrix[:,:,:,0] = matrix
        
        # --- FREQUENCY
        # prepare filter bank kernels
        kernels     = []
        powers      = []
        frequencies = np.array([0.1,0.2,0.3,0.4]) #4 frequencies
        for theta in range(4):
            theta = theta / 4. * np.pi #4 angles
            for frequency in frequencies: #a total of 16 filters
                kernel_temp = gabor_kernel(frequency, theta = theta)
                power_img = self.power_calc(img,kernel_temp)
                powers.append (np.mean(power_img))
                #save only real kernel
                kernels.append(np.real(kernel_temp)) 
        self.powers  = powers
        self.kernels = kernels


        # COMPUTE FEATURES
        self.features = self._calc_features()
        
    def _calc_features(self):
        # FEATURES
        features = {} 
        
        # SKIMAGE PROPS
        props    = regionprops(label_image = self.bin_img, intensity_image = self.img,spacing=self.spacing)
        
        # CCONVEX HULL
        try:
            convimg = props[0].convex_image
            if len(np.unique(convimg)) == 1:
                print('could not get convex hull')
                return features
        except:
            return features
        
        # --- MORPHOLOGY 
        if any('DCF:Number of Pixels' in string for string in self.listfeats):
            features['DCF:Number of Pixels'] = props[0].num_pixels
        
        # AREA AND VOLUME
        if any(('DCF:Area' in string or 'DCF:Area convex' in string or 'DCF:Volume' in string or 'DCF:Volume convex') for string in self.listfeats):
            if self.dim == 2:
                features['DCF:Area']                   = props[0].area 
                features['DCF:Area convex']            = props[0].area_convex 
            if self.dim == 3:
                features['DCF:Volume']                 = props[0].area
                features['DCF:Volume convex']          = props[0].area_convex 

        # BOUNDING BOX AREA
        if any('DCF:BB Area' in string for string in self.listfeats):
            features['DCF:BB Area']             = props[0].bbox_area
        
        # PERIMETER
        if any('DCF:Perimeter' in string for string in self.listfeats):
            if self.dim == 2:
                features['DCF:Perimeter']           = props[0].perimeter 
            if self.dim == 3:
                test = np.pad(self.bin_img,((1,1),(0,0),(0,0)))
                #test = self.bin_img.transpose(1, 2, 0)
                verts, faces, normals, values   = marching_cubes(test,level=0)
                peri                            = mesh_surface_area(verts,faces)
                features['DCF:Perimeter']       = peri

                verts, faces, normals, values   = marching_cubes(test, level = 0, spacing=getvoxelsize(self.original_folder))
                peri                            = mesh_surface_area(verts,faces)
                features['DCF:Perimeter (micron)']  = peri
            
            
        # CENTROID
        if any('DCF:Centroid' in string for string in self.listfeats):
            features['DCF:Centroid']              = [round(x,3) for x in props[0].centroid] 
        
        # WEIGHTED CENTROID
        if any('DCF:Weighted Centroid' in string for string in self.listfeats):
            features['DCF:Weighted Centroid']     = [round(x,3) for x in props[0].weighted_centroid]  
        
        # CENTROID DIVERGENCE
        if any('DCF:Centroid Divergence' in string for string in self.listfeats): 
            if self.dim == 2:
                features['DCF:Centroid Divergence']   = self.spacing[-1] * (np.linalg.norm(np.array(features['DCF:Centroid']) - np.array(features['DCF:Weighted Centroid'])))
        
        # EQUIVALENT DIAMETER
        if any('DCF:Equivalent Diameter' in string for string in self.listfeats):  
            #The diameter of a circle with the same area as the region.
            features['DCF:Equivalent Diameter']   = props[0].equivalent_diameter 
        
        # EXTENT
        if any('DCF:Extent' in string for string in self.listfeats):  
            features['Extent']                = props[0].extent
            
        # MAJOR AXIS LENGTH
        if any('DCF:Major Axis Length' in string for string in self.listfeats):  
            features['Major Axis Length']     = props[0].major_axis_length 
            
        # MINOR AXIS LENGTH
        if any('DCF:Minor Axis Length' in string for string in self.listfeats):  
            features['Minor Axis Length']     = props[0].minor_axis_length  
            
#         if self.dim == 3:
#             features['Height']                = getvoxelsize(self.original_folder)[0] * (max(np.where(self.bin_img*1>0)[0])-min(np.where(self.bin_img*1>0)[0]))
#             features['XComp']                 = getvoxelsize(self.original_folder)[1] * (max(np.where(self.bin_img*1>0)[1])-min(np.where(self.bin_img*1>0)[1]))
#             features['YComp']                 = getvoxelsize(self.original_folder)[2] * (max(np.where(self.bin_img*1>0)[2])-min(np.where(self.bin_img*1>0)[2]))
#         if self.dim == 2:
#             features['XComp']                 = getvoxelsize(self.original_folder)[0] * (max(np.where(self.bin_img*1>0)[0])-min(np.where(self.bin_img*1>0)[0]))
#             features['YComp']                 = getvoxelsize(self.original_folder)[1] * (max(np.where(self.bin_img*1>0)[1])-min(np.where(self.bin_img*1>0)[1]))
            
        # EULER NUMBER
        if any('DCF:Euler Number' in string for string in self.listfeats):  
            features['DCF:Euler Number']          = props[0].euler_number
        
        # ECCENTRICITY
        if any('DCF:Eccentricity' in string for string in self.listfeats):  
            try:
                features['DCF:Eccentricity']        = props[0].eccentricity #Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points).
            except:
                features['DCF:Eccentricity']        = props[0].minor_axis_length / props[0].major_axis_length
        
        # CIRCULARITY and ROUNDNESS
        if any('DCF:Extent' in string for string in self.listfeats):  
            if self.dim == 2:
                #Circularity that specifies the roundness of objects.
                features['DCF:Circularity']         = (4*features['DCF:Area']*math.pi)/(features['DCF:Perimeter']**2) 

                #Like circularity, but does not depend on perimeter/roughness.
                features['DCF:Roundness']           = (4*features['DCF:Area'])/(np.pi* props[0].major_axis_length**2)
            if self.dim == 3:
                #Circularity that specifies the roundness of objects.
                features['DCF:Circularity']         = (4*features['DCF:Volume']*math.pi)/(features['DCF:Perimeter']**2) 

                #Like circularity, but does not depend on perimeter/roughness.
                features['DCF:Roundness']           = (4*features['DCF:Volume'])/(np.pi* props[0].major_axis_length**2)
        
        # ASPECT RATIO
        if any('DCF:Aspect Ratio' in string for string in self.listfeats):  
            features['DCF:Aspect Ratio']       = (props[0].major_axis_length)/ props[0].minor_axis_length #Aspect ratio.
        
        # ORIENTATION
        try:
            features['DCF:Orientation']         = props[0].orientation 
        except:
            features['DCF:Orientation']         = 0
        
        # SOLIDITY
        if any('DCF:Solidity' in string for string in self.listfeats):  
            features['DCF:Solidity']            = props[0].solidity #Ratio of pixels in the region to pixels of the convex hull image.  
        
        # ROUGHNESS
        if any('DCF:Roughness' in string for string in self.listfeats):  
            try:
                imconv = props[0].convex_image.astype(np.uint8)
                vertsh, facesh, normalsh, valuesh = marching_cubes(imconv, spacing=getvoxelsize(self.original_folder))
                perihull = mesh_surface_area(vertsh,facesh)

                features['DCF:Roughness']           = features['DCF:Perimeter']/perihull # get convex image of roi and compute perimeter as Ratio of perimeter of region to perimeter of the convex hull image.
            except:
                features['DCF:Roughness']           = 0
            
#         try:
#             features['Hu Moment #1']        = props[0].moments_hu[0] #tuple - Hu moments (translation, scale and rotation invariant) of intensity image.
#             features['Hu Moment #2']        = props[0].moments_hu[1] 
#             features['Hu Moment #3']        = props[0].moments_hu[2]

#             features['Weighted Hu Moment #1'] = props[0].moments_weighted_hu[0]  
#             features['Weighted Hu Moment #2'] = props[0].moments_weighted_hu[1] 
#             features['Weighted Hu Moment #3'] = props[0].moments_weighted_hu[2] 
#         except:
#             features['Hu Moment #1']        = 0
#             features['Hu Moment #2']        = 0 
#             features['Hu Moment #3']        = 0
#             features['Weighted Hu Moment #1'] = 0  
#             features['Weighted Hu Moment #2'] = 0 
#             features['Weighted Hu Moment #3'] = 0
 

        #features['Zernike Moments']    = zernike_moments(self.img,self.img[0].shape[0]/2)   # Zernike Moments of Region [diam = self.img[0].shape[0]/2, maxradius = diam/2]
#         try:
#             features['Feret Diameter Max']  = props[0].feret_diameter_max
#         except:
#             features['Feret Diameter Max']  = 0

        features['Crofton Perimeter']   = props[0].perimeter_crofton
        
        # --- INTENSITY
        features['DCF:Mean Intensity'] = np.mean(self.imgzero) 
        features['DCF:Std']            = np.std(self.imgzero) 
        features['DCF:Variance']       = np.var(self.imgzero) 
        features['DCF:Skewness']       = skew(self.imgzero) 
        features['DCF:Kurtosis']       = kurtosis(self.imgzero) 
        features['DCF:Contrast']       = np.std(self.hist) #contrast can be defined as std of histogram of intensity
        features['DCF:Max Intensity']  = props[0].max_intensity #Value with the greatest intensity in the region.
        features['DCF:Min Intensity']  = props[0].min_intensity #Value with the greatest intensity in the region.
        features['DCF:Entropy']        = shannon_entropy(self.img, base=2) 
        features['DCF:CV']             = features['DCF:Std'] / features['DCF:Mean Intensity']
        
        features['DCF:Inertia Tensor Highest Eigenvalue'] = props[0].inertia_tensor_eigvals[0]
        features['DCF:Inertia Tensor Lowest Eigenvalue'] = props[0].inertia_tensor_eigvals[1]
        

        if any(('DCF:AAI' in string or 'DCF:AMI' in string or 'DCF:AFI' in string ) for string in self.listfeats):
            if type(self.skel) != str:
                features['DCF:AAI'] = getAAI(self.skel)

    

        # --- GLCM
#     try:
        unif   = []
        ent    = []
        matrix = self.matrix[:,:,0,0]
        for i in np.arange(self.glcm.shape[3]):
            mat          = self.glcm[:,:,0,i]
            feature_unif = (mat ** 2).sum()
            unif.append(feature_unif)
            feature_ent  = shannon_entropy(mat)
            ent.append(feature_ent)   

        # UNIFORMITY
        if any('DCF:Uniformity' in string for string in self.listfeats):
            features['DCF:Uniformity']              = list(unif) #uniformity for each matrix/angle
        
        # INVARIANT UNIFORMITY
        if any('DCF:Invariant Uniformity' in string for string in self.listfeats):
            features['DCF:Invariant Uniformity']    = (matrix ** 2).sum()
        
        # GLCM ENTROPY
        if any('DCF:GLCM Entropy' in string for string in self.listfeats):
            features['DCF:GLCM Entropy']            = list(ent) #same, but for entropy
        
        # GLCM INVARIANT ENTROPY
        if any('DCF:GLCM Invariant Entropy' in string for string in self.listfeats):
            features['DCF:GLCM Invariant Entropy']  = shannon_entropy(matrix)
        
        # CORRELATION
        if any('DCF:Correlation' in string for string in self.listfeats):
            features['DCF:Correlation']             = graycoprops(self.glcm, 'correlation')[0]
        
        # INVARIANT CORRELATION
        if any('DCF:Invariant Correlation' in string for string in self.listfeats):
            features['DCF:Invariant Correlation']   = float(graycoprops(self.matrix, 'correlation')[0][0])
        
        # DISSIMILARITY
        if any('DCF:Dissimilarity' in string for string in self.listfeats):
            features['DCF:Dissimilarity']           = graycoprops(self.glcm, 'dissimilarity')[0]
        
        # INVARIANT DISSIMILARITY
        if any('DCF:Invariant Dissimilarity' in string for string in self.listfeats):
            features['DCF:Invariant Dissimilarity'] = float(graycoprops(self.matrix, 'dissimilarity')[0][0])   
        
        # CONTRAST
        if any('DCF:Contrast' in string for string in self.listfeats):
            features['DCF:Contrast']                = graycoprops(self.glcm, 'contrast')[0]
        
        # INVARIANT CONTRAST
        if any('DCF:Invariant Contrast' in string for string in self.listfeats):
            features['DCF:Invariant Contrast']      = float(graycoprops(self.matrix, 'contrast')[0][0])
        
        # HOMOGENEITY
        if any('DCF:Homogeneity' in string for string in self.listfeats):
            features['DCF:Homogeneity']             = graycoprops(self.glcm, 'homogeneity')[0]
        
        # INVARIANT HOMOGENEITY
        if any('DCF:Invariant Homogeneity' in string for string in self.listfeats):
            features['DCF:Invariant Homogeneity']   = float(graycoprops(self.matrix, 'homogeneity')[0][0])
        
        # ENERGY
        if any('DCF:Energy' in string for string in self.listfeats):
            features['DCF:Energy']                  = graycoprops(self.glcm, 'energy')[0]
        
        # INVARIANT ENERGY
        if any('DCF:Invariant Energy' in string for string in self.listfeats):
            features['DCF:Invariant Energy']        = float(graycoprops(self.matrix, 'energy')[0][0])


        
        
        # --- Freq
        # Gabor
#         try:
        features['DCF:Mean Gabor Power'] = mean(self.powers)

        f_var,f_mean,f_energy,f_ent    = [],[],[],[]
        for k, kernel in enumerate(self.kernels):
            filtered = ndi.convolve(self.img, kernel, mode='wrap')
            f_mean.append(np.mean(filtered))
            f_var.append(np.var(filtered))
            f_energy.append(np.sum(np.power(filtered.ravel(),2))/len(filtered.ravel()))
            f_ent.append(shannon_entropy(filtered))

        features['DCF:Gabor Variance'] = mean(f_var)
        features['DCF:Gabor Mean']     = mean(f_mean)
        features['DCF:Gabor Energy']   = mean(f_energy)
        features['DCF:Gabor Entropy']  = mean(f_ent)

        # --- FFT
        f                  = np.fft.fft2(self.img)
        fshift             = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        features['DCF:Mean Spectral Magnitude'] = np.mean(magnitude_spectrum) #in dB

        # --- Welch
        f_psd, Pxx = signal.welch(self.img)
        sum_fp     = np.multiply(f_psd, Pxx)
        sum_int    = np.sum(Pxx)
        mean_pxx   = np.sum(sum_fp)/sum_int

        features['DCF:Mean Spectral Power'] = mean_pxx

#             for k in features.keys():
#                 try:
#                     features[k] = round(features[k],3)
#                 except:
#                     pass
#         except:
#             pass
        
        
        # Fractal Dimension
        # Deconvoluted
#         FD_b,FD_d = fractal_dimension_grayscale(self.img)
#         fractal_values_n_b = [round(FD_b)]
#         features['Fractal Dimension B Deconvoluted'] = fractal_values_n_b
#         fractal_values_n_d = [round(FD_d)] 
#         features['Fractal Dimension D Deconvoluted'] = fractal_values_n_d
        
        # Skeleton
        if any('DCF:Fractal Dimension Skeleton' in string for string in self.listfeats):
            if type(self.skel) != str:
                FD_skel = fractal_dimension((self.skel>0)*1)
                features['DCF:Fractal Dimension Skeleton'] = FD_skel
    #         except:
    #             pass
        


        if self.dim == 3:
            if 100*(features['DCF:Volume convex'] - features['DCF:Volume'])/features['DCF:Volume'] > 250:
                print('Volume convex exceeded 200%')
                features = {}
                return features
        if self.dim == 2:
            if 100*(features['DCF:Area convex'] - features['DCF:Area'])/features['DCF:Area'] > 250:
                print('Area convex exceeded 200%, precisely: ' + str(100*(features['DCF:Area convex'] - features['DCF:Area'])/features['DCF:Area']))
                features = {}
                return features
            
        
        
        
        return features
        
    
    def power_calc(self,image,kernel):
        # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        #convolves images with filter
        return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 + ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
    


    
    def print_features(self, print_values = False):
        feature_labels = []
        feature_values = []
        for key in self.features.keys():
            if print_values:
                print("{}: {}".format(key, self.features[key]))
            feature_labels.append(key)
            feature_values.append(self.features[key])
            
        return feature_labels, feature_values
