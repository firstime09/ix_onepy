import glob, os
import numpy as np
from osgeo import gdal

def readRaster(path, dem):
    # Read Data Landsat 8 and Sentinel 2
    raster_list = glob.glob(path + '*.TIF')
    read = []
    for i in raster_list:
        band = gdal.Open(i)
        read.append(band.GetRasterBand(1).ReadAsArray().astype(float))
    filename = []
    for a in [os.path.basename(x) for x in glob.glob(path + '*.TIF')]:
        p = os.path.splitext(a)[0]
        filename.append(p)
    my_dict = dict(zip(filename, read))
    # Read Data Sentinel DEM
    raster_list_dem = glob.glob(dem + '*.TIF')
    read_dem = []
    for j in raster_list_dem:
        band_dem = gdal.Open(j)
        read_dem.append(band_dem.GetRasterBand(1).ReadAsArray().astype(float))
    return(my_dict, read_dem)
