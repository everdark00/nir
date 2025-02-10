import re
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

import xgboost as xgb
from tqdm import tqdm

from .tde import TreeDrivenEncoder


class DeepTLF(BaseEstimator):
    """
    A deep learning model based on XGBoost and a custom neural network.

    Parameters
    ----------
    n_est : int, optional
        Number of estimators for XGBoost model, default is 23.
    max_depth : int, optional
        Max depth for each tree in XGBoost, default is 4.
    xgb_lr : float, optional
        Learning rate for XGBoost model, default is 0.5.

    Attributes
    ----------
    xgb_model : XGBClassifier or XGBRegressor
        Fitted XGBoost model.
    TDE_encoder : TreeDrivenEncoder
        Fitted Tree-Driven Encoder.
    device : torch.device
        Device used for computations ('cuda' or 'cpu').
    """

    def __init__(
        self,
        n_est=23,
        max_depth=4,
        xgb_lr=0.5,
        min_freq=2
    ):
        self.n_est = n_est
        self.max_depth = max_depth
        self.xgb_lr = xgb_lr
        self.xgb_model = None
        self.TDE_encoder = TreeDrivenEncoder(min_freq)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train):
        self.fit_xgb(X_train)
        trees = self.xgb_model.get_booster().get_dump(with_stats=False)
        self.TDE_encoder.fit(trees)
        return self.TDE_encoder.all_conditions

    def transform(self, X):
        return self.TDE_encoder.transform(X)

    def fit_xgb(self, X_train):
        # Using XGBRegressor for self-supervised learning.
        self.xgb_model = xgb.XGBRegressor(
            learning_rate=self.xgb_lr,
            n_jobs=-1,
            max_depth=self.max_depth,
            n_estimators=self.n_est,
        )
        # Using X_train as target for self-supervised learning.
        self.xgb_model.fit(X_train, X_train)
