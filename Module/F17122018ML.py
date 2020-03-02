import math, numpy, pandas, os, glob
import joblib as jb
from osgeo import gdal
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def save_model(model, path):
    with open(path, "wb") as mdl:
        return joblib.dump(model, mdl)
    
def load_model(path):
    with open(path, "rb") as mdl:
        return joblib.load(mdl)

class F2020ML:

    def plot_data(DataY, DataX):
        """Data Visualization 2D 16/01-2019"""
        plt.plot(DataX, DataY)
        plt.scatter(DataX, DataY, edgecolors='none', s=30, label='Data')
        plt.title('Data Visualization')
        plt.xlabel('Data X')
        plt.ylabel('Data Y')
        plt.legend()
        plt.show()

    def export_array(in_path, in_array, output_path):
        """
        This function is used to produce output of array as a map.
        Ket. For map of array from Mr. Sahid (13/12-2018)
        """
        global proj, geotrans, row, col
        proj        = in_path.GetProjection()
        geotrans    = in_path.GetGeoTransform()
        row         = in_path.RasterYSize
        col         = in_path.RasterXSize
        driver      = gdal.GetDriverByName("GTiff")
        outdata     = driver.Create(output_path, col, row, 1, gdal.GDT_CFloat32)
        outband     = outdata.GetRasterBand(1)
        outband.SetNoDataValue(-9999)
        outband.WriteArray(in_array)
        outdata.SetGeoTransform(geotrans) # Georeference the image
        outdata.SetProjection(proj) # Write projection information
        outdata.FlushCache()
        outdata = None
        return outdata

    def readRaster(path, dem):
        """"Read Data Landsat 8 dan Sentinel 2"""
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

    def F2020_DF(dframe):
        # Load_class = 'Class'
        # dclass = numpy.asarray(dframe[Load_class])
        dclass = numpy.asarray(dframe)
        df1 = pandas.Series(dclass).value_counts().reset_index().sort_values('index').reset_index(drop=True)
        df1.columns = ['Class', 'Frequency']
        a = df1.min()
        return a

    def F2020_Class_Min(dataframe):
        """ 08/11-2019
        Select minimum class from data, so you must to have column class in your data frame
        """
        df1 = pandas.Series(dataframe).value_counts().reset_index().sort_values('index').reset_index(drop=True)
        df_min = df1.min()
        df_min = numpy.asarray(df_min[1])
        return df_min

    def F2020_Class_data(self, data):
        return self.sample(data)

    def F2020_RSQRT(ActualY, PredictY):
        """R Squared function 04/12-2018"""
        rScores = (1 - sum((ActualY - PredictY)**2) / sum((ActualY - ActualY.mean(axis=0))**2))
        return rScores

    def F2020_RMSE(ActualY, PredictY):
        """Root Mean Squared Error (RMSE) 04/12-2018"""
        rootMSE = (math.sqrt(sum((ActualY - PredictY)**2) / ActualY.shape[0]))
        return rootMSE

    def F2020_RMSError(Actual, Prediction):
        """Root Mean Squared Error (RMSE) 10/12-2019"""
        rMSE = numpy.sqrt(((Prediction - Actual)**2).mean())
        return rMSE

    def F2020_MinMax(data):
        """Normalization using Min-Max function"""
        Norm_MinMax = (data - numpy.min(data)) / (numpy.max(data) - numpy.min(data))
        return Norm_MinMax

    def F2020_ZScore(data):
        """Normalization using Z Score function"""
        Norm_ZScore = (data - numpy.mean(data)) / (numpy.std(data))
        return Norm_ZScore

    """Support Vector Regression Model"""
    def F2020_SVR_TEST_SC(dataX, dataY, tsize, rstate, k, c, g, e):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        minMax = MinMaxScaler()
        minMax.fit(X_train)
        X_train = minMax.fit_transform(X_train)
        X_test = minMax.transform(X_test)
        SVRModel = SVR(kernel=k, C=c, gamma=g, epsilon=e)
        SVRModel.fit(X_train, y_train)
        return(SVRModel)

    def F2020_SVR_TEST_NSC(dataX, dataY, tsize, rstate, k, c, g, e):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        SVRModel = SVR(kernel=k, C=c, gamma=g, epsilon=e)
        SVRModel.fit(X_train, y_train)
        return(SVRModel)

    def F2020_SVR_SC(dataX, dataY, tsize, rstate):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        sc = MinMaxScaler()
        sc.fit(X_train)
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        best_score = 0
        for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gamma in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
                for epsilon in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    # Train Model SVR
                    clfSVR = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                    clfSVR.fit(X_train, y_train)
                    score = clfSVR.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_parameters = {'C': C, 'gamma': gamma, 'epsilon': epsilon}
        # return(best_score, best_parameters)
        return(clfSVR)

    def SVR_Model(dataX, dataY, test_size, r_state):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=test_size, random_state=r_state)
        # sc = MinMaxScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        best_score = 0
        for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gamma in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1]:
                for epsilon in [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]:
                    calfSVR = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                    calfSVR.fit(X_train, y_train)
                    score = calfSVR.score(X_test, y_test)
                    if score > best_score:
                        best_score = score
                        best_C = C
                        best_gam = gamma
                        best_eps = epsilon
                        # best_parm = {'C': best_C, 'Gamma': best_gam, 'Epsilon': best_eps}

        calfSVR_Model = SVR(kernel='rbf', C=best_C, epsilon=best_eps, gamma=best_gam)
        calfSVR_Model.fit(X_train, y_train)
        # calfSVR_Score = calfSVR_Model.score(X_test, y_test)
        # y_pred = calfSVR_Model.predict(X_test)
        # RMSE_Model = F2020ML.F2020_RMSE(y_test, y_pred)
        # R2_Model = F2020ML.F2020_RSQRT(y_test, y_pred)
        # Model = {'RMSE': RMSE_Model, 'R^2': R2_Model}
        return(calfSVR_Model)

    """Random Forests Regression Model"""
    def F2020_RFR_SC(dataX, dataY, tsize, rstate): #--- Random Forest Regressor Model
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        best_score = 0
        for n_esti in [50, 100, 150, 200, 400, 450]:
            """Train Random Forests Model"""
            clfRFR = RandomForestRegressor(n_estimators=n_esti, random_state=rstate)
            clfRFR.fit(X_train, y_train)
            score = clfRFR.score(X_test, y_test)
            if score > best_score:
                best_score = score
                total_tree = {'n_estimators': n_esti}
        # return(best_score, total_tree)
        return(clfRFR)

    def RFR_Model(dataX, dataY, tsize, rstate):
        X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=tsize, random_state=rstate)
        # sc = MinMaxScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        best_score = 0
        for n_esti in [50, 150, 200, 250, 300, 350, 400, 450, 500]:
            clfRFR = RandomForestRegressor(n_estimators=n_esti, random_state=rstate)
            clfRFR.fit(X_train, y_train)
            score = clfRFR.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_esti = n_esti

        clfRFR_Model = RandomForestRegressor(n_estimators=best_esti, random_state=rstate)
        clfRFR_Model.fit(X_train, y_train)
        return(clfRFR_Model)
