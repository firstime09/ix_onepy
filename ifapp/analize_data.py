import glob, os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class func_raster():

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
    
class func_Mlearn():
    
    def Model_RMSE(actual, predic):
        rmse = np.sqrt(((actual - predic)**2).mean())
        return rmse
    
    def Model_R2(actual, predic):
        rsqrt = (1 - sum((actual - predic)**2) / sum((actual - actual.mean(axis=0))**2))
        return rsqrt
    
    def RFR_Model(dataX, dataY, tsize, rstate):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        ###
        best_score = 0
        for n_estimate in [150, 200, 250, 300, 350, 400, 450, 500]:
            clfRFR = RandomForestRegressor(n_estimators=n_estimate, random_state=rstate)
            clfRFR.fit(X_train, y_train)
            score = clfRFR.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_esti = n_estimate
        clfRFR_Model = RandomForestRegressor(n_estimators=best_esti, random_state=rstate)
        clfRFR_Model.fit(X_train, y_train)
        return(clfRFR_Model)
    
    def SVR_Model(dataX, dataY, tsize, rstate, ker):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        ###
        best_score = 0
        for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gamma in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]:
                for epsilon in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    clfSVR = SVR(kernel=ker, C=C, gamma=gamma, epsilon=epsilon)
                    clfSVR.fit(X_train, y_train)
                    score = clfSVR.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_C = C
                        best_gam = gamma
                        best_eps = epsilon
        clfSVR_Model = SVR(kernel=ker, C=best_C, gamma=best_gam, epsilon=best_eps)
        clfSVR_Model.fit(X_train, y_train)
        return(clfSVR_Model)
    
    def plot_data(DataY, DataX):
        """Data Visualization 2D 16/01-2019"""
        plt.plot(DataX, DataY)
        plt.scatter(DataX, DataY, edgecolors='none', s=30, label='Data')
        plt.title('Data Visualization')
        plt.xlabel('Data X')
        plt.ylabel('Data Y')
        plt.legend()
        plt.show()