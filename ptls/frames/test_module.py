import pandas as pd
import pytorch_lightning as pl 
import torch
import numpy as np 
from ptls.data_load.padded_batch import PaddedBatch 

class TestModule(pl.LightningModule):
    def __init__(self, model, metrics):
        super().__init__()

        self.metrics = metrics
        self.model = model

    def forward(self, batch):
        x, y = batch
        out = self.model(x)

        for m_name in self.metrics.keys():
            self.metrics[m_name].update(out, y)


    def get_metrics(self):
        ds_metrics = {}
        for m_name in self.metrics.keys():
            ds_metrics[m_name] = (self.metrics[m_name].mean_value / self.metrics[m_name].weight).item()
            self.metrics[m_name].mean_value = torch.tensor(0.)
            self.metrics[m_name].weight = torch.tensor(0.)
        return ds_metrics