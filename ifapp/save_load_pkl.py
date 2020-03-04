import joblib

def pkl_model_save(model, path):
    with open(path, "wb") as mdl:
        return joblib.dump(model, mdl)

def pkl_model_load(path):
    with open(path, "rb") as mdl:
        return joblib.load(mdl)