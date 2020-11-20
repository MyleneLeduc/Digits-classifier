from scipy.io import loadmat
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from os.path import dirname, join as pjoin
import scipy.io as sio
from sklearn.model_selection import train_test_split


class DigitsDataset(Dataset):
    def __init__(self, file_name):
        mat_contents = sio.loadmat(file_name)
        X = mat_contents['X']
        y = mat_contents['y']
        for i in range(len(y)):
            if y[i] == 10:
                y[i] = 0
        y = np.ravel(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.y_train = torch.FloatTensor(y_train)
        self.y_test = torch.FloatTensor(y_test)
    def __len__(self):
        return(len(self.X))
    def __getitem__(self, i):
        return self.X[i], self.y[i]