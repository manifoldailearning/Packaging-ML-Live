import numpy as np
import pandas as pd 
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from wine_prediction.eval import evaluation_metrics
from wine_prediction.config import config
from wine_prediction.processing import data_handling,preprocessing




if __name__ == "__main__":
    # csv_url = (
    #     "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    # )
    # data = pd.read_csv(config.DATASET_URL, sep=";")
    data = data_handling.load_dataset()
    X = data.drop(columns=config.TARGET)
    y = data[config.TARGET]
    
    X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)

    alpha=config.ALPHA
    l1_ratio = config.L1_RATIO
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=0)
    lr.fit(X_train,y_train)
    data_handling.save_model(lr) # serialization

    final_model = data_handling.load_model(config.MODEL_NAME) # deserialization
    predictions = final_model.predict(X_test)

    (rmse, mae,  r2) = evaluation_metrics(y_test,predictions)
    print(f"rmse is {rmse}")
    print(f"mae is {mae}")
    print(f"r2 is {r2}")