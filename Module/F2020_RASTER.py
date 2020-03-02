import glob, os
import numpy as np
from osgeo import gdal

class Raster_Func():

    def stack_data(self, input_path, name):
        path = input_path
        file_layer = glob.glob(path + "/*.tif")
        file_vrt = path + "/Stacked.vrt"
        file_tif = path + (name + ".tif")
        vrt = gdal.BuildVRT(file_vrt, file_layer, separate=True)
        stack_layer = gdal.Translate(file_tif, vrt)
        # outData = output_path + stack_layer
        return stack_layer

    def saved_data_TIF(out_path1, pred_model, name, ras):
        ## Make data prediction to TIF file
        saved_data = (name + "F2020.TIF")
        output_path = (out_path1 + saved_data)
        # raster = in_path1 + '/CIDANAU_STACK_13052019.tif'
        raster = ras
        in_path = gdal.Open(raster)
        in_array = pred_model
        ## global proj, geotrans, row, col
        proj = in_path.GetProjection()
        geotrans = in_path.GetGeoTransform()
        row = in_path.RasterYSize
        col = in_path.RasterXSize
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(output_path, col, row, 1, gdal.GDT_Float32)
        # outdata = driver.Create(output_path, col, row, 1, gdal.GDT_Byte)
        outband = outdata.GetRasterBand(1)
        outband.SetNoDataValue(-9999)
        outband.WriteArray(in_array)
        outdata.SetGeoTransform(geotrans)  # Georeference the image
        outdata.SetProjection(proj)  # Write projection information
        outdata.FlushCache()
        outdata = None
        return outdata

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
        return (my_dict, read_dem)
