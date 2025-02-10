from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

class KDiscretizer():
    def __init__(self, f_names, k_bins, d_type='quantile', emb_sz=None):
        '''
        d_type should be one of {'quantile', 'uniform', ‘kmeans’}
        '''
        self.emb_sz = emb_sz
        self.f_names = f_names
        self.k_bins = k_bins
        self.disc = KBinsDiscretizer(
            n_bins=k_bins, encode='ordinal', strategy=d_type)
        self.emb_tresholds = {fn : [] for fn in self.f_names}
        self.tresholds = {}

    def decrease_n_bins(self, tresholds, k_bins_required):
        bins_gap = pd.DataFrame({'bn' : tresholds[:-1], 
                                'gap' : [tresholds[i] - tresholds[i - 1] for i in range(1, len(tresholds))]})
        bins_gap = bins_gap.sort_values(by='gap', ascending=True)

        for j in range(len(tresholds) + 1 - k_bins_required):
            bins_gap.iloc[j, :] = np.nan
        bins_gap = bins_gap.dropna()
        return list(sorted(bins_gap.bn.values)) + [tresholds[-1]]

    def fit(self, X):
        self.disc.fit(X.loc[:, self.f_names])
        self.tresholds = {self.disc.feature_names_in_[i] : self.disc.bin_edges_[i] for i in range(len(self.f_names))}
        for fn in self.f_names:
            if self.emb_sz is not None:
                if len(self.tresholds[fn]) < self.emb_sz + 1:
                    raise Exception(f"too few bins in {fn} discretization, raise k_bins or lower pruning rate")
                else:
                    self.emb_tresholds[fn] = self.decrease_n_bins(self.tresholds[fn], self.emb_sz + 2)

    def fit_transform(self, X, to_embeds=False):
        self.disc.fit(X.loc[:, self.f_names])
        self.tresholds = {self.disc.feature_names_in_[i] : self.disc.bin_edges_[i] for i in range(len(self.f_names))}
        for fn in self.f_names:
            if self.emb_sz is not None:
                if len(self.tresholds[fn]) < self.emb_sz + 1:
                    raise Exception(f"too few bins in {fn} discretization, raise k_bins or lower pruning rate")
                else:
                    self.emb_tresholds[fn] = self.decrease_n_bins(self.tresholds[fn], self.emb_sz + 2)
        return self.transform(X, to_embeds)

    def transform(self, X, to_embeds=False):
        if to_embeds:
            for i, fn in enumerate(self.f_names):
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
            X.loc[:, self.f_names] = self.disc.transform(X.loc[:, self.f_names]).astype(int)
        return X