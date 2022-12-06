import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import matplotlib.pyplot as plt
import os
from pydicom.pixel_data_handlers.util import apply_color_lut
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from scipy.interpolate import interpn
np.set_printoptions(threshold=np.inf)



def load_scan(path):
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
    slices = sorted(slices, key = lambda s: s.SliceLocation)
    return slices

def load_scan_(path):
    print("Loading scan", path)
    slices = [dcmread(path + '/' + s) for s in os.listdir(path)]
    #slices = [dcmread(path + '/' + iteration) for iteration in os.listdir(path) if os.path.isfile(iteration)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1
            slice_num = int(len(slices) / sec_num)
            slices.sort(key = lambda x:float(x.InstanceNumber))
            slices = slices[0:slice_num]
            slices.sort(key = lambda x:float(x.ImagePositionPatient[2]))
    try:
            slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
            slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
            s.SliceThickness = slice_thickness
        
    slices = sorted(slices, key = lambda s: s.SliceLocation)
    return slices # list of DICOM 



def transferFunction(x):
       
    r = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    g = 1.0*np.exp( -(x - 9.0)**2/1.0 ) +  1.0*np.exp( -(x - 3.0)**2/0.1 ) +  0.1*np.exp( -(x - -3.0)**2/0.5 )
    b = 0.1*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) +  1.0*np.exp( -(x - -3.0)**2/0.5 )
    a = 0.6*np.exp( -(x - 9.0)**2/1.0 ) +  0.1*np.exp( -(x - 3.0)**2/0.1 ) + 0.01*np.exp( -(x - -3.0)**2/0.5 )
       
    return r,g,b,a

def get_pixel_array(slices):
    slices_pixel_array = []
    for iteration in slices:
        slices_pixel_array.append(iteration.pixel_array)
    return slices_pixel_array


def get_pixels_hu(slices):
    
    hu_images = np.stack([iteration.pixel_array for iteration in slices])
    hu_images = hu_images.astype(np.int16)

    #convert to HU units in m * SV + b
    for slice in range(len(slices)):
        #Rescale intercept - b, rescale slope - m, SV - stored values in m * SV + b
        intercept = slices[slice].RescaleIntercept
        slope = slices[slice].RescaleSlope
        if slope != 1:
            hu_images[slice] = slope * hu_images[slice].astype(np.float64)
            hu_images[slice] = hu_images[slice].astype(np.int16)

        hu_images[slice] += np.int16(intercept)

    case_pixels = np.array(hu_images, dtype = np.int16)
    pixel_spacing = np.array([slices[0].SliceThickness, slices[0].PixelSpacing[0], slices[0].PixelSpacing[1]], dtype = np.float32)
    
    return case_pixels, pixel_spacing

direct_path = '/media/sapere-aude/D/PROJECTS/PRJ/VOLUME_RENDER/converter/data/PAT031/D0001.dcm'
relative_path = 'data/PAT031/D0001.dcm'
path = "data/PAT031"


#dicom_list = [dcmread(path + '/' + s) for s in os.listdir(path)]
dicom_list_parced = []
dicom_list_parced = load_scan_(path)

#print(dicom_list_parced)
def pixel_list_dicom(slices):

    pixel_list_dicom = []

    #pixel_array - method from dicom to pixel
    for iteration in dicom_list_parced:
        pixel_numpy = iteration.pixel_array
        pixel_list_dicom.append(pixel_numpy)
        
    return pixel_list_dicom

def three_dimension_array(slices):
    dimension = (slices[0].pixel_array[0].shape, slices[0].pixel_array[1].shape, len(slices))
    return dimension




#print(pixel_list_dicom)

#data_pixels_hu, data_pixel_spacing = get_pixels_hu(dicom_list_parced)
#plt.imshow(data_pixels_hu[1], cmap=plt.cm.gray)
#plt.show()
data_pixel_array = get_pixel_array(dicom_list_parced)
#plt.imshow(data_pixel_array[1], cmap = 'viridis')
#plt.show()
#print(data_pixel_array[0].shape)
#print(type(data_pixel_array[0].shape))
#print(len(data_pixel_array))
#print(len(dicom_list_parced))

def three_dimension_graphic(dicom_list_parced):
    ps = dicom_list_parced[0].PixelSpacing
    ss = dicom_list_parced[0].SliceThickness
    ax_aspect = ps[1]/ps[0]
    sag_aspect = ps[1]/ss
    cor_aspect = ss/ps[0]
    dimension = list(dicom_list_parced[0].pixel_array.shape)
    dimension.append(len(dicom_list_parced))
    img3d = np.zeros(dimension)
    for i, j in enumerate(dicom_list_parced):
        img2d = j.pixel_array
        img3d[:, :, i] = img2d
    a1 = plt.subplot(2, 2, 1)
    plt.imshow(img3d[:, :, dimension[2]//2])
    a1.set_aspect(ax_aspect)
    z = dimension[2]
 

    a2 = plt.subplot(2, 2, 2)
    plt.imshow(img3d[:, dimension[1]//2, :])
    a2.set_aspect(sag_aspect)
    y = dimension[1]
  

    a3 = plt.subplot(2, 2, 3)
    plt.imshow(img3d[dimension[0]//2, :, :].T)
    a3.set_aspect(cor_aspect)
    x = dimension[0]
    #plt.text(x,y,z)

    return plt.show()

three_dimension_graphic(dicom_list_parced)
dimension = list(dicom_list_parced[0].pixel_array.shape)
dimension.append(len(dicom_list_parced))
x = dimension[0]
y = dimension[1]
z = dimension[2]
print(x, y, z)



def volume_render(slices):
    dimension = list(slices[0].pixel_array.shape)
    dimension.append(len(slices))
    datacube = np.zeros(dimension)
    for i, j in enumerate(slices):
        pixels_2d = j.pixel_array
        datacube[:, :, i] = pixels_2d
    Nx, Ny, Nz = dimension
    print(dimension)
    print(Nx, Ny, Nz)
    x = np.linspace(-Nx/2, Nx/2, Nx)    
    y = np.linspace(-Ny/2, Ny/2, Ny)
    z = np.linspace(-Nz/2, Nz/2, Nz)
    points = (x, y, z)
    print(points)
    Nangles = 10
    for i in range(Nangles):
        print('Rendering Scene ' + str(i+1) + ' of ' + str(Nangles) + '.\n')
       
        # Camera Grid / Query Points -- rotate camera view

        angle = np.pi/2 * i / Nangles

        N = 180
        c = np.linspace(-N/2, N/2, N)
        qx, qy, qz = np.meshgrid(c,c,c)
        qxR = qx
        qyR = qy * np.cos(angle) - qz * np.sin(angle) 
        qzR = qy * np.sin(angle) + qz * np.cos(angle)
        qi = np.array([qxR.ravel(), qyR.ravel(), qzR.ravel()]).T

        # Interpolate onto Camera Grid
        camera_grid = interpn(points, datacube, qi, method='nearest').reshape((N,N,N))

        # Do Volume Rendering
        image = np.zeros((camera_grid.shape[1],camera_grid.shape[2],3))
       
        for dataslice in camera_grid:
            r,g,b,a = transferFunction(np.log(dataslice))
            image[:,:,0] = a*r + (1-a)*image[:,:,0]
            image[:,:,1] = a*g + (1-a)*image[:,:,1]
            image[:,:,2] = a*b + (1-a)*image[:,:,2]

        #image = np.clip(image,0.0,1.0)

        # Plot Volume Rendering
        plt.figure(figsize=(4,4), dpi=300)
        plt.imshow(image)
        plt.axis('off')
        print(type(image))
        print(type(camera_grid))

        # Save figure
        plt.savefig('volumerender' + str(i) + '.png',dpi=300,  bbox_inches='tight', pad_inches = 0)

       
       
       
    # Plot Simple Projection -- for Comparison
    plt.figure(figsize=(4,4), dpi=300)
       
    plt.imshow(np.log(np.mean(datacube,0)), cmap = 'viridis')
    plt.clim(-5, 5)
    plt.axis('off')
       
    # Save figure
    plt.savefig('projection.png',dpi=300,  bbox_inches='tight', pad_inches = 0)
    plt.show()
    return 0

volume_render(dicom_list_parced)
