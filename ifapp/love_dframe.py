import joblib
import pandas as pd
import numpy as np

def pkl_model_save(model, path):
    with open(path, "wb") as mdl:
        return joblib.dump(model, mdl)

def pkl_model_load(path):
    with open(path, "rb") as mdl:
        return joblib.load(mdl)
    
class ifapp_df():
#     def min_frequence(dframe): """this code is not finished"""
#         dt = np.asarray(dframe)
#         df = pd.Series(dt).value_counts().reset_index().sort_values('index').reset_index(drop=True)
#         df.columns = ['Class', 'Frequency']
#         data = df.min()
#         return data
    
    def combine_dframe(df1, df2):
        dt1 = pd.DataFrame({"X1": df1})
        dt2 = pd.DataFrame({"X2": df2})
        df_join = pd.concat([dt1, dt2], axis=1)
        return df_join