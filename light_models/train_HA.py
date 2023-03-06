from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from utils import seed_everything, score
from config import HAConfig
import pandas as pd
import numpy as np
import pickle
import os

# here we compare the performance of linear regression and support vector regression
if __name__ == '__main__':
    # Set device and load config
    device = 'cpu'
    seed_everything()
    config = HAConfig(device)
    # Load data
    df = pd.read_csv(config.data_path + config.dataset)
    years = df.year
    if config.year == 'all':
        # we choose all years and remove duplicate houses
        df_new = df.copy()
        for i in list(set(years)):
            df_year = df[df['year'] == i]
            df_year = df_year.drop_duplicates(subset=['house'], keep='last')
            if i == list(set(df.year))[0]:
                df_new = df_year
            else:
                df_new = pd.concat((df_new, df_year))
        df = df_new
    else:
        # we choose only 1 year
        df = df[df['year'] == config.year]
        # remove duplicate houses
        df = df.drop_duplicates(subset=['house'], keep='last')
    y = df['price'].values
    df = df.drop(['price', 'house'], axis=1)
    X = df.values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-config.train_ratio, \
     random_state=42)
    # fit estimator 1
    est1 = LinearRegression()
    est1.fit(X_train, y_train)
    # predict class labels
    pred1 = est1.predict(X_test)

    # fit estimator 2
    est2 = SVR(C=config.C, epsilon=config.epsilon, kernel=config.kernel, gamma=config.gamma)
    est2.fit(X_train, y_train)
    # predict class labels
    pred2 = est2.predict(X_test)

    pred = (pred1 + pred2) / 2
    # score on test data (accuracy)
    mse, mae, mape, rmse = score(pred, y_test)
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("MAPE: ", mape)
    print("RMSE: ", rmse)
    # save model
    if not os.path.isdir(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    #pickle.dump(est, open(config.ckpt_path + config.model_name, 'wb'))
    

