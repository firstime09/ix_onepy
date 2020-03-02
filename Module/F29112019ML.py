import os, glob, joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def ix_model_save(model, path):
    with open(path, "wb") as mdl:
        return joblib.dump(model, mdl)

def ix_model_load(path):
    with open(path, "rb") as mdl:
        return joblib.load(mdl)

class JuPyter_ML:
    def SVC_Model_NScale(dtX, dtY, test_z, r_state, kernel):
        """Model Support Vector Classification"""
        X_train, X_test, y_train, y_test = train_test_split(dtX, dtY, test_size=test_z, random_state=r_state)
        # sc = StandardScaler()
        # sc.fit(X_train)
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)
        best_score = 0
        for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gamma in [0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1]:
                clfSVC = SVC(kernel=kernel, C=C, gamma=gamma)
                clfSVC.fit(X_train, y_train)
                score = clfSVC.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_C = C
                    best_gamma = gamma

        clfSVC_use = SVC(kernel=kernel, C=best_C, gamma=best_gamma)
        clfSVC_use.fit(X_train, y_train)
        return(clfSVC_use)
    
    def plot_data(dataY, dataX):
        plt.plot(dataX, dataY)
        plt.scatter(dataX, dataY, edgecolors='none', s=30, label='Data')
        plt.title('Data Visualization')
        plt.xlabel('Data X')
        plt.ylabel('Data Y')
        plt.legend()
        plt.show()
        
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