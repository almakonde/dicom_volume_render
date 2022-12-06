import numpy as np
import pydicom
from pydicom.data import get_testdata_file
from pydicom import dcmread
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_color_lut

np.set_printoptions(threshold=np.inf)

pat31Path = './data/PAT031/D0002.dcm'
#DICOM_file = dcmread(pat31Path)
#testData = get_testdata_file(pat31Path)
DICOM_file = dcmread(pat31Path)
#DICOM_test = dcmread(testData)
print(DICOM_file)
#DICOM_file.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
pixelarray = DICOM_file.pixel_array
print(pixelarray)
print("class object from pydicom ->", type(DICOM_file))
print("class object from pydicom pixelarray ->", type(pixelarray))
plt.imshow(DICOM_file.pixel_array, cmap=None)
plt.show()
rgb = apply_color_lut(pixelarray, palette='PET')
print(rgb)
plt.imshow(rgb)
plt.show()
print(DICOM_file)

