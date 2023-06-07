# make logistic regression and xgboost models
import pandas as pd
import numpy as np
import os
import gc
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


if not os.path.exists('sklearn_models'):
    os.makedirs('sklearn_models')

np.random.seed(42)

X = np.load('./data/train_emb.npy')
y = np.load('./data/y_train.npy')
X_test = np.load('./data/test_emb.npy')
y_test = np.load('./data/y_test.npy')
X = X.reshape(-1, 512)
X_test = X_test.reshape(-1, 512)
print(X.shape, y.shape)
print(X_test.shape, y_test.shape)

# create the logistic regression model
linear_model = LinearRegression()
linear_model.fit(X, y)

# calculate the MSE RMSE and MAE and MAPE
y_pred = linear_model.predict(X_test)
scaler = joblib.load('./data/scaler_price.pkl')
y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)


print("Linear Regression")
print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

# save the model
pickle.dump(linear_model, open('./sklearn_models/linear_model.pkl', 'wb'))

# create the xgboost model
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X, y)
y_pred = xgb_model.predict(X_test)

# add dimension to y_pred
y_pred = y_pred.reshape(-1, 1)
y_pred = scaler.inverse_transform(y_pred)
print("XGBoost")
print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))
