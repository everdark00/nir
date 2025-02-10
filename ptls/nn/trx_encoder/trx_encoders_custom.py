from collections import OrderedDict

import torch
from torch import nn as nn

from ptls.data_load.padded_batch import PaddedBatch
from .custom_embeddings import GloveEmbedding, TransEmbedding
from ptls.nn.trx_encoder.noisy_embedding import NoisyEmbedding
from ptls.nn.trx_encoder.trx_encoder_base import TrxEncoderBase


class TrxEncoderGlove(nn.Module):
    def __init__(self,  
                 glove_embedding : GloveEmbedding = None,
                 agg_type="cat",
                 numeric_separate = False,
                 numeric_features=[]
                 ):
        super().__init__()
        self.numeric_separate = numeric_separate
        self.numeric_features = numeric_features
        self.agg_type = agg_type
        self.feature_names = glove_embedding.feature_names
        self.embedding_vectors = glove_embedding.get_vectors(agg_type="mean")
                
    

    def forward(self, x: PaddedBatch):
        if self.agg_type == "cat":
            out = []
            for fe in self.feature_names:
                out.append(self.embedding_vectors(x.payload[fe]))
            if self.numeric_separate:
                for fe in self.numeric_features:
                    out.append(x.payload[fe])
            out = torch.cat(out, dim=2)
            return PaddedBatch(out, x.seq_lens)
        else:
            if self.numeric_separate:
                raise Exception("Only cat agg alowed with numeric separate")
            out = self.embedding_vectors(x.payload[self.feature_names[0]])
            for fe in self.feature_names[1:]:
                out += self.embedding_vectors(x.payload[fe])
            if self.agg_type == "sum":
                return PaddedBatch(out, x.seq_lens)
            else:
                return PaddedBatch(out/len(self.feature_names), x.seq_lens)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.agg_type == "cat":
            return self.embedding_vectors.weight.shape[1] * len(self.feature_names)  + (0 if not self.numeric_separate else len(self.numeric_features))
        else:
            return self.embedding_vectors.weight.shape[1]

class TrxEncoderCat(TrxEncoderBase):
    def __init__(self,  
                 embeddings,
                 numeric_separate=False,
                 numeric_features=None,
                 embeddings_noise=0.003,
                 emb_dropout=0,
                 spatial_dropout=False,
                 agg_type: str = "cat",
                 out_of_index: str = 'clip',
                 ):
        super().__init__()

        self.numeric_separate = numeric_separate
        self.numeric_features = numeric_features
        self.device = 'cuda'
        self.esz = 16

        for e in self.embeddings.values():
            self.esz = e.embedding_dim
            break

        noisy_embeddings = {}
        for emb_name, emb_props in embeddings.items():
            if emb_props.get('disabled', False):
                continue
            if emb_props['in'] == 0 or emb_props['out'] == 0:
                continue
            noisy_embeddings[emb_name] = NoisyEmbedding(
                num_embeddings=emb_props['in'],
                embedding_dim=emb_props['out'],
                padding_idx=0,
                max_norm=None,
                noise_scale=embeddings_noise,
                dropout=emb_dropout,
                spatial_dropout=spatial_dropout,
            )

        super().__init__(
            embeddings=noisy_embeddings,
            numeric_values=None,
            custom_embeddings={},
            out_of_index=out_of_index,
        )

        self.agg_type = agg_type  

    def forward(self, x: PaddedBatch):
        processed_embeddings = []

        if self.numeric_separate:
            for fn in self.numeric_features:
                value = x.payload[f'{fn}_val']
                pos = x.payload[f'{fn}_pos'] 
                numeric_embedding = torch.ones((pos.shape[0], pos.shape[1], self.esz)).double().to(self.device)
                zero_mask = torch.ones(pos.shape[0], pos.shape[1]).double().to(self.device)
                for i in range(self.esz):
                    if i > 0:
                        numeric_embedding[:, :, i] *= zero_mask
                    numeric_embedding[:, :, i][pos == i + 1] = value[pos == i + 1]
                    numeric_embedding[:, :, i][pos == 0] = 0
                    zero_mask *= (pos != i + 1)
                processed_embeddings.append(numeric_embedding)

        for field_name in self.embeddings.keys():
            processed_embeddings.append(self.get_category_embeddings(x, field_name))

        if self.agg_type == "cat":
            out = torch.cat(processed_embeddings, dim=2)
        else:
            n_emb = 0
            out = None
            for i, emb in enumerate(processed_embeddings):
                out = emb if i == 0 else out + emb
                n_emb += 1
            if self.agg_type == "mean":
                out = out / n_emb

        return PaddedBatch(out.float(), x.seq_lens)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.agg_type == "cat":
            return self.esz * (len(self.embeddings) + (len(self.numeric_features) if self.numeric_separate else 0))
        else:
            return self.esz

'''
algos

orig: obtain embeddings of size len(feature_names), sum them and sum with raw features 
classic: num discr + different aggregations on obtained embeds
'''
class TrxEncoderTrans(nn.Module):
    def __init__(self,  
                 feature_names, #must be cat or discretized num
                 algo='orig',
                 in_emb_sizes=[],
                 out_emb_size=None,
                 agg_type="cat",
                 numeric_separate=False,
                 numeric_features=[],
                 ):
        super().__init__()

        if 'algo' not in {'orig', 'classic'}:
            raise Exception('algo must be "orig" or "classic"')

        self.numeric_separate = numeric_separate
        self.numeric_features = numeric_features if self.numeric_separate else []

        self.device = 'cpu'
        self.algo = algo
        if self.algo == 'orig':
            self.esz = len(feature_names) + len(numeric_features)
        else:
            self.esz = out_emb_size

        self.agg_type = agg_type
        
        self.feature_names = feature_names
        self.embeddings = TransEmbedding(feature_names, in_emb_sizes, self.esz, self.device, self.algo)
                
    

    def forward(self, x: PaddedBatch):
        if self.algo == 'orig':
            out = self.embeddings(x)

            out += torch.cat([x.payload[i] for i in self.numeric_features + self.feature_names], dim=2)
        else:
            if self.agg_type == "cat":
                out = self.embeddings(x)
                for fe in self.numeric_features:
                    out.append(x.payload[fe])
                out = torch.cat(out, dim=2)
                return PaddedBatch(out, x.seq_lens)
            else:
                if self.numeric_separate:
                    raise Exception('mean and sum agg does not supports using non-disc numeric features')
                out = torch.sum(self.embeddings(x), dim=2)
                
                if self.agg_type == "mean":
                    out = out / len(self.feature_names)
                    
        return PaddedBatch(out, x.seq_lens)

    @property
    def output_size(self):
        """Returns hidden size of output representation
        """
        if self.agg_type == "cat":
            return self.esz * (len(self.embeddings.features) + (len(self.numeric_features) if self.numeric_separate else 0))
        else:
            return self.esz