from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluation_metrics(y_test,y_pred):
    rmse = mean_squared_error(y_test,y_pred,squared=False)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return rmse,mae,r2