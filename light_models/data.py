import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class Simple_Dataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, y

class DNN_Dataset(Dataset):
    def __init__(self, input, output, number_of_features):
        # Feature selection
        self.sfs = SequentialFeatureSelector(LogisticRegression(),
                                k_features=number_of_features,
                                forward=True,
                                scoring='accuracy',
                                cv=None)
        self.pca = PCA(n_components=number_of_features)
        self.output = output
        self.selected_data = self.sfs.fit_transform(input, output)
        self.transformed_data = self.pca.fit_transform(input)
        self.input = np.concatenate((self.selected_features, self.transformed_data), axis=1)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        x = self.input[idx]
        y = self.output[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, y

    
class KNSHS_Dataset(Dataset):
    def __init__(self, input, output, lon, lat, K, distance_limit):
        self.output = output
        self.knn = KNSHS(input, lon, lat, K, distance_limit)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        x = self.knn.apply_knn(idx).transpose(1,0)
        y = self.output[idx]
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return x, y

class KNSHS:
    def __init__(self, input, lon, lat, K, distance_limit):
        self.K = K
        self.input = input
        self.lon = lon
        self.lat = lat
        self.distance_limit = distance_limit
    
    def distance(self, lon1, lat1, lon2, lat2):
        # lon1, lat1: longitude and latitude of the first point
        # lon2, lat2: longitude and latitude of the second point
        # return: orthodromic distance between the two points
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) * np.sin(dlat / 2) + \
            np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
        c = 2 * np.arcsin(np.sqrt(a))
        d = R * c
        return d
    
    def apply_knn(self, idx):
        # x: input data
        # K: number of nearest neighbors
        # return: x with K nearest neighbors
        closest = []
        for i in range(len(self.input)):
            if i == idx:
                continue
            dist = self.distance(self.lon[idx], self.lat[idx], self.lon[i], self.lat[i])
            if dist <= self.distance_limit:
                closest.append((i, dist))
        closest.sort(key=lambda x: x[1])
        closest = closest[:self.K-1]

        x = np.expand_dims(self.input[idx], axis=1)
        for i in range(len(closest)):
            x = np.append(x, np.expand_dims(self.input[closest[i][0]], axis=1), axis=1)
        if x.shape[1] < self.K:
            x = np.append(x, (-1)*np.ones((x.shape[0], self.K - x.shape[1])), axis=1)
        # put the input data at the center of the array
        # flipping is for better feature extraction
        x = np.append(np.flip(x[:, self.K//2+2:]), x[:, :self.K//2+2], axis=1)
        #print(sum(x[:, self.K//2+1]==0))
        #exit()
        return x
