from numpy.core.fromnumeric import nonzero
from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np

class IncomeDataset(Dataset):
    def __init__(self, df, embed_df, device, dtype):
        X = df.drop(columns=['income>50K']).values
        y = df['income>50K'].values[:, None]
            
        X_embed_df = embed_df.apply(lambda x: x.cat.codes).values

        self.X = torch.from_numpy(X).to(device=device, dtype=dtype)
        self.X_embed = torch.from_numpy(X_embed_df).to(device=device, dtype=torch.int64)
        self.y = torch.from_numpy(y).to(device=device, dtype=dtype)        
        self.shape = self.X.shape


    def get_embed_sizes(self):
        return self.embed_sizes


    def get_balanced_weights(self):
        nonzero_idx = torch.nonzero(self.y)
        nonzero_pct = nonzero_idx.shape[0] / self.y.shape[0]
        weights = torch.zeros_like(self.y.flatten())
        weights[nonzero_idx] = 1 - nonzero_pct
        weights[weights==0] = nonzero_pct
        return weights

    def __getitem__(self, index):
        return {'X': self.X[index], 'X_embed': self.X_embed[index], 'y': self.y[index]}


    def __len__(self):
        return self.shape[0]



if __name__ == '__main__':
    df = pd.read_csv('./data/train_final.csv')
    train_len = int(df.shape[0] * .8)
    
    train_df = df.iloc[:train_len, :]
    val_df = df.iloc[train_len:, :]
    train_dataset = IncomeDataset(train_df, 'cpu', torch.float32)
    print(train_dataset[3])


    