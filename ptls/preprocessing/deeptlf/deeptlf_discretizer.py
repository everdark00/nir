
from .src import DeepTLF
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import pandas as pd
import os


class DeepTLFDisc():
    def __init__(self, params):
        self.params = params
        self.tree_encoder = DeepTLF(n_est=self.params["n_est"], min_freq=self.params["min_freq"])
        self.emb_tresholds = dict([(fe, []) for fe in self.params["features"]])
        self.tresholds = dict([(fe, []) for fe in self.params["features"]])
        self.k_bins = []

    def decrease_n_bins(self, tresholds, k_bins_required):
        bins_gap = pd.DataFrame({'bn' : tresholds[:-1],
                                'gap' : [tresholds[i] - tresholds[i - 1] for i in range(1, len(tresholds))]})
        bins_gap = bins_gap.sort_values(by='gap', ascending=True)

        for j in range(len(tresholds) + 1 - k_bins_required):
            bins_gap.iloc[j, :] = np.nan
        bins_gap = bins_gap.dropna()
        return list(sorted(bins_gap.bn.values)) + [tresholds[-1]]


    def fit(self, X):
        split_conditions = self.tree_encoder.fit(X[self.params["features"]])
        for cond in split_conditions:
            self.tresholds[cond["feature"]].append(cond["threshold"])

        for fn in self.params["features_to_split"]:
            self.k_bins.append(len(self.tresholds[fn]))
            if "emb_size" in self.params:
                if len(self.tresholds[fn]) < self.params["emb_size"] + 1:
                    raise Exception(f"too few bins in {fn} discretization, raise k_bins or lower pruning rate")
                else:
                    self.emb_tresholds[fn] = self.decrease_n_bins(self.tresholds[fn], self.params["emb_size"])

    def fit_transform(self, X, to_embeds=False):
        self.fit(X)
        return self.transform(X, to_embeds)

    def transform(self, X, to_embeds=False):
        if to_embeds:
            for i, fn in enumerate(self.params["features_to_split"]):
                self.emb_tresholds[fn] = [X[fn].min()] + self.emb_tresholds[fn] + [X[fn].max()]
                passed_idxs = []
                position = []
                value = []
                for j in range(1, len(self.emb_tresholds[fn])):
                    gap_vals = X.loc[(X[fn] > self.emb_tresholds[fn][j - 1]) * (X[fn] <= self.emb_tresholds[fn][j]), fn]
                    value.append((gap_vals.values - self.emb_tresholds[fn][j - 1])/(self.emb_tresholds[fn][j] + self.emb_tresholds[fn][j - 1]))
                    position.append((np.ones(value[-1].shape[0]) * (j)).astype(int))
                    passed_idxs.append(gap_vals.index.values.astype(int))
                X = X.drop(columns=[fn])
                X = X.join(
                    pd.DataFrame({
                        f'{fn}_pos' : np.concatenate(position),
                        f'{fn}_val' : np.concatenate(value)
                    }).set_index(np.concatenate(passed_idxs))
                )
        else:
            for fe in self.params["features_to_split"]:
                ts = sorted(self.tresholds[fe])
                X.loc[X[fe] < ts[0], fe] = 0
                for i in range(1, len(ts)):
                    X.loc[(X[fe] > ts[i - 1]) & (X[fe] <= ts[i]), fe] = i
                X.loc[X[fe] > ts[len(ts) -1], fe] = len(ts)
                X[fe] = X[fe].astype("int64")
        return X