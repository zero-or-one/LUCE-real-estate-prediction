import numpy as np
import pandas as pd
from sklearn.externals import joblib 


# load scaler
scaler = joblib.load('./data/scaler.pkl')

# load txt file of arrays and convert to numpy array
target = pd.read_csv('./data/target.txt').values[:,0]
prediction = pd.read_csv('./data/valid_pred.txt').values[:,0]


# ['  [-0.70294523]' '  [-0.86014014]' '  [-0.82346135]' ...
#  '  [-0.9732332 ]' '  [-0.9732332 ]' '  [-0.72741985]]]']
# remove all the spaces and brackets
target = np.array([float(i.strip('[] ').strip()) for i in target])
prediction = np.array([float(i.strip('[] ').strip()) for i in prediction])

print(target.shape)
print(target)

# reshape to 2D array
target = target.reshape(-1, 1)
prediction = prediction.reshape(-1, 1)

# inverse transform
target_padding = np.zeros((target.shape[0], 338))
prediction_padding = np.zeros((prediction.shape[0], 338))
target = np.concatenate((target_padding, target), axis=1)
prediction = np.concatenate((prediction_padding, prediction), axis=1)


target = scaler.inverse_transform(target)[:, -1]
prediction = scaler.inverse_transform(prediction)[:, -1]

# save to txt
np.savetxt('./data/inv_target.txt', target)
np.savetxt('./data/inv_valid_pred.txt', prediction)