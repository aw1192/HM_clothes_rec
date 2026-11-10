# create dataset class

import duckdb
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class TransactionDataset(Dataset):
    # expects integer columns 'user_id' and 'item_id' after Austin's indexing step

    def __init__(self, data):
        if isinstance(data, (str, os.PathLike)): # if data is path
            df = duckdb.read_parquet(os.fspath(data)).select('user_id, item_id').df()
        elif isinstance(data, pd.DataFrame): # if data is pandas df
            df = data[['user_id', 'item_id']]
        else:
            raise TypeError("Data must be a parquet path or pandas DataFrame.")
        
        self.user_ids = torch.as_tensor(df['user_id'].to_numpy(), dtype=torch.long)
        self.item_ids = torch.as_tensor(df['item_id'].to_numpy(), dtype=torch.long)
    
    def __len__(self):
        return len(self.item_ids)
    
    def __getitem__(self, idx):
        return self.item_ids[idx], self.user_ids[idx]