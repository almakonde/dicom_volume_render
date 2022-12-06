from pydicom import dcmread

from pydicom.data import get_testdata_file

filename = get_testdata_file('/media/sapere-aude/D/PROJECTS/PRJ/VOLUME_RENDER/converter/data/PAT031/D0254.dcm')

ds = dcmread(filename)

ds.PixelData 