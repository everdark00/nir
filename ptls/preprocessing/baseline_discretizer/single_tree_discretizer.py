from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

class SingleTreeDiscretizer():
    def __init__(self, f_names, target_name, target_type, k_bins=None, pruning_rates=None, emb_sz=None):
        '''
        k_bins, pruning_rates - optional
        '''
        self.target_name = target_name
        self.target_type = target_type
        self.pruning_rates = pruning_rates
        self.f_names = f_names
        self.k_bins = k_bins
        self.tresholds = {fn : [] for fn in self.f_names}
        self.emb_tresholds = {fn : [] for fn in self.f_names}
        self.emb_sz = emb_sz

    def decrease_n_bins(self, tresholds, k_bins_required):
        bins_gap = pd.DataFrame({'bn' : tresholds[:-1], 
                                'gap' : [tresholds[i] - tresholds[i - 1] for i in range(1, len(tresholds))]})
        bins_gap = bins_gap.sort_values(by='gap', ascending=True)

        for j in range(len(tresholds) + 1 - k_bins_required):
            bins_gap.iloc[j, :] = np.nan
        bins_gap = bins_gap.dropna()
        return list(sorted(bins_gap.bn.values)) + [tresholds[-1]]

    def fit(self, X):
        for i, fn in enumerate(self.f_names): 
            if self.k_bins is not None:
                ccp_alpha_min = 1e-8
                ccp_alpha_max = 1e-3

                if self.emb_sz is None:
                    k_bins_f = self.k_bins[i]
                else:
                    k_bins_f = max(self.k_bins[i], self.emb_sz + 2)
    
                tresholds = None
                for i in tqdm(range(15), desc=fn):
                    curr_ccp_alpha = (ccp_alpha_max + ccp_alpha_min) / 2.0
                    if self.target_type == 'classification':
                        disc = DecisionTreeClassifier(
                            criterion='gini',
                            ccp_alpha=curr_ccp_alpha,
                        )
                    else:
                        disc = DecisionTreeRegressor(
                            criterion='gini',
                            ccp_alpha=curr_ccp_alpha,
                        )
                    disc.fit(X=X.loc[:, [fn]], y=X[self.target_name])
                    curr_k_bins = disc.tree_.threshold[disc.tree_.threshold != -2].shape[0] + 1
                    tresholds = sorted(disc.tree_.threshold[disc.tree_.threshold != -2])

                    if curr_k_bins < k_bins_f:
                        ccp_alpha_max = curr_ccp_alpha
                    elif curr_k_bins > k_bins_f + 0.5 * k_bins_f:
                        ccp_alpha_min = curr_ccp_alpha
                    else:
                        break
                
                self.tresholds[fn] = self.decrease_n_bins(tresholds, k_bins_f)


            else:
                if self.pruning_rates is not None:
                    ccp_alpha = self.pruning_rates[i]
                else:
                    ccp_alpha = 1e-4
                
                if self.target_type == 'classification':
                    disc = DecisionTreeClassifier(
                        criterion='gini',
                        ccp_alpha=ccp_alpha,
                    )
                else:
                    disc = DecisionTreeRegressor(
                        criterion='gini',
                        ccp_alpha=ccp_alpha,
                    )
                disc.fit(X=X.loc[:, [fn]], y=X[self.target_name])
                self.tresholds[fn] = list(sorted(disc.tree_.threshold[disc.tree_.threshold != -2]))
            if self.emb_sz is not None:
                if len(self.tresholds[fn]) < self.emb_sz + 1:
                    raise Exception(f"too few bins in {fn} discretization, raise k_bins or lower pruning rate")
                else:
                    self.emb_tresholds[fn] = self.decrease_n_bins(self.tresholds[fn], self.emb_sz + 2)

    def fit_transform(self, X, to_embeds=False):
        self.fit(X)
        return self.transform(X, to_embeds)

    def transform(self, X, to_embeds=False):
        for i, fn in enumerate(self.f_names):
            if to_embeds:
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
                X.loc[(X[fn] <= self.tresholds[fn][0]), fn] = 0
                for j in range(1, len(self.tresholds[fn])):
                    X.loc[(X[fn] > self.tresholds[fn][j - 1]) * (X[fn] <= self.tresholds[fn][j]), fn] = j
                X.loc[(X[fn] > self.tresholds[fn][-1]), fn] = len(self.tresholds[fn])
                X[fn] = X[fn].astype("int64")
    
        return X

