import numpy as np
import pandas as pd



class MugRepDataset:
    def __init__(self, df, target, idxs):
        self.df = df
        self.target = target
        self.idxs = idxs

    def __len__(self):
        return len(target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.df.iloc[idx]
        y = self.target.iloc[idx]
        loc = self.idxs[idx]
        return x, y, loc

