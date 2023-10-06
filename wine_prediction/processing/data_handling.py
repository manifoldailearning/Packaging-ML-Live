import os
from wine_prediction.config import config
import joblib

def load_dataset():
    url = config.DATASET_URL
    _data = pd.read_csv(url, sep=";")
    return _data

# serialization
def save_model(model_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    joblib.dump(model_to_save,save_path)
    print(f"Model has been saved successfully under that path {save_path}")

#deserialization
def load_model(modelname_to_load):
    load_path = os.path.join(config.SAVE_MODEL_PATH,modelname_to_load)
    loaded_model = joblib.load(load_path)
    return loaded_model
