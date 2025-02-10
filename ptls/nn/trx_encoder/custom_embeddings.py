
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
import pandas as pd
import os

class GloVe(nn.Module):

    def __init__(self, vocab_size, embedding_size, x_max, alpha=0.75):
        super().__init__()
        self.weight = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        )
        self.weight_tilde = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_size,
            sparse=True
        )
        self.bias = nn.Parameter(
            torch.randn(
                vocab_size,
                dtype=torch.float,
            )
        )
        self.bias_tilde = nn.Parameter(
            torch.randn(
                vocab_size,
                dtype=torch.float,
            )
        )
        self.weighting_func = lambda x: (x / x_max).float_power(alpha).clamp(0, 1)
    
    def forward(self, i, j, x):
        loss = torch.mul(self.weight(i), self.weight_tilde(j)).sum(dim=1)
        loss = (loss + self.bias[i] + self.bias_tilde[j] - x.log()).square()
        loss = torch.mul(self.weighting_func(x), loss).mean()
        return loss

class GloveEmbedding():
    def __init__(self, 
                 feature_names,
                 calculate_cooccur=True,
                 embedding_folder="glove_embedding",
                 glove_params={"alpha" : 0.75, "x_max" : 100, "embedding_size" : 16, "num_epochs_train" : 10}
                 ):
        os.makedirs(embedding_folder, exist_ok=True) 

        self.feature_names = feature_names
        
        self.token2cat = {0 : 0}
        self.cat2token = {0 : 0}
        
        self.cooccur_dataset = []
        self.calculate_cooccur = calculate_cooccur
        self.saved_data_path = os.path.join(embedding_folder, f"emb-{glove_params['alpha']}-{glove_params['x_max']}-{glove_params['embedding_size']}")
        os.makedirs(self.saved_data_path, exist_ok=True) 
        
        self.model = []
        self.params = glove_params
        

    def fit(self, data):
        data = data[self.feature_names]
                
        num_features = len(self.feature_names)
        
        #build vocabulary 
        vocabulary = [0]
        for fe in (self.feature_names):
             vocabulary += [f"{fe}_" + str(i) for i in data[fe].unique()]

        token = 1
        for word in vocabulary:
            self.token2cat.update([(token, word)])
            self.cat2token.update([(word, token)])
            token += 1

        with open(os.path.join(self.saved_data_path, 'glove_token2cat.pkl'), 'wb') as f:
            pickle.dump(self.token2cat, f)
        with open(os.path.join(self.saved_data_path, 'glove_cat2token.pkl'), 'wb') as f:
            pickle.dump(self.cat2token, f)
        
        #build coocur dict
        if self.calculate_cooccur:
            cooccur_dict = dict()
            for item in data.values:
                for i in range(num_features):
                    for j in range(i + 1, num_features):
                        t1 = self.cat2token[f"{self.feature_names[i]}_" + str(item[i])]
                        t2 = self.cat2token[f"{self.feature_names[j]}_" + str(item[j])]
                        if cooccur_dict.get((t1, t2)):
                            cooccur_dict[(t1, t2)] += 1
                        elif cooccur_dict.get((t1, t2)):
                            cooccur_dict[(t1, t2)] += 1
                        else:
                            cooccur_dict.update([((t1, t2), 1)])

            self.cooccur_dataset = np.zeros((len(cooccur_dict), 3))
            for idx, ((i, j), count) in enumerate(cooccur_dict.items()):
               self.cooccur_dataset[idx] = (i, j, count)

            del cooccur_dict
            pd.DataFrame(self.cooccur_dataset, columns = ["w1", "w2", "count"]).to_csv(os.path.join(self.saved_data_path, "cooccur_dataset.csv"))
        else:
            self.cooccur_dataset = pd.read_csv(os.path.join(self.saved_data_path, "cooccur_dataset.csv"), index_col=[0]).values
        
                
        #train glove model
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        self.model = GloVe(
            vocab_size=len(vocabulary) + 1,
            embedding_size=self.params["embedding_size"],
            x_max=self.params["x_max"],
            alpha=self.params["alpha"]
        )
        
        dataloader = DataLoader(
            dataset=self.cooccur_dataset,
            batch_size=32,
            shuffle=True
        )
        
        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=7e-2
        )

        self.model.to(device)
        
        print("train started")
        self.model.train()
        losses = []
        for epoch in tqdm(range(self.params["num_epochs_train"])):
            epoch_loss = 0
            for batch in dataloader:
                batch = batch.type(torch.int64)
                batch = batch.to(device)
                loss = self.model(
                    batch[:, 0],
                    batch[:, 1],
                    batch[:, 2]
                )
                epoch_loss += loss.detach().item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
            losses.append(epoch_loss)
            print(f"Epoch {epoch}: loss = {epoch_loss}")
        torch.save(self.model.state_dict(), os.path.join(self.saved_data_path, "model.pth"))

    def load(self, model_path="glove_models/model.pth"):
        with open(os.path.join(self.saved_data_path, "glove_cat2token.pkl"), "rb") as f:
           self.cat2token = pickle.load(f)

        self.model = GloVe(
            vocab_size=len(self.cat2token) + 1,
            embedding_size=self.params["embedding_size"],
            x_max=self.params["x_max"],
            alpha=self.params["alpha"]
        )

        self.model.load_state_dict(torch.load(os.path.join(self.saved_data_path, "model.pth")))

    def tokenize_data(self, data):
        for fe in self.feature_names:
            data[fe] = [fe + "_" + str(i) for i in data[fe].values]

        data[self.feature_names] = np.vectorize(self.cat2token.get)(data[self.feature_names].values)
        return data

    def get_vectors(self, agg_type="sum"):
        if agg_type == "sum":
            vectors = self.model.weight.weight.detach() + self.model.weight_tilde.weight.detach()
            return nn.Embedding.from_pretrained(vectors)
        elif agg_type == "mean":
            vectors = np.mean([self.model.weight.weight.detach().numpy(), self.model.weight_tilde.weight.detach().numpy()], axis=0)
            return nn.Embedding.from_pretrained(torch.Tensor(vectors))
        else:
            return self.model.weight


class TransEmbedding(nn.Module):
    def __init__(self, feature_names, cat_emb_sizes, out_emb_size, device='cpu', dropout=0.2, algo='orig'):
        super(TransEmbedding, self).__init__()
        
        self.cat_table = nn.ModuleDict({col: nn.Embedding(cat_emb_sizes[i] + 1, 
                            out_emb_size).to(device) for i, col in enumerate(feature_names)})

        self.forward_mlp = nn.ModuleList(
            [nn.Linear(out_emb_size, out_emb_size) for i in range(len(feature_names))])
        self.dropout = nn.Dropout(dropout)
        self.features = feature_names
        self.emb_dict = None
        self.algo = algo

    def forward_emb(self, df):
        if self.emb_dict is None:
            self.emb_dict = self.cat_table
        support = {col: self.emb_dict[col](df[col]) for col in self.features}
        
        return support

    def forward(self, df):
        support = self.forward_emb({i : df.payload[i] for i in self.features})
        if self.algo == 'orig':
            output = 0
            for i, k in enumerate(support.keys()):
                support[k] = self.dropout(support[k])
                support[k] = self.forward_mlp[i](support[k])
                output = output + support[k]
        else:
            output = []
            for i, k in enumerate(support.keys()):
                support[k] = self.dropout(support[k])
                support[k] = self.forward_mlp[i](support[k])
                output = output.append(support[k])
        return output

    
        

#https://github.com/AI4Risk/antifraud/blob/main/methods/gtan/gtan_main.py   
        