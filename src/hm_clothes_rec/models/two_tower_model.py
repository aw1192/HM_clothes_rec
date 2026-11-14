import json
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

IDX_DIR = Path(__file__).resolve().parents[3] / 'data' / 'indexed'


def emb_dim(n):
    '''
    rule of thumb embedding size for a categorical col with n unique values, capped at 32
    '''
    return min(32, round(1.6 * n ** 0.56))


class TwoTowerRetrieval(nn.Module):
    '''
    Two tower model for retrieval in recommendation systems.
    '''

    def __init__(self):
        super().__init__() # necessary to use nn.Module

        # ========== ITEM TOWER ===========

        # RETURN EMBEDDING WITH 256 DIMENSIONS
        emb_size = 100 # TODO: decide on the input size

        self.item_tower = nn.Sequential(
            nn.LazyLinear(1024), # LazyLinear for now, during prototyping phase
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256)
        )

        # ========== USER TOWER ===========

        # RETURN EMBEDDING WITH 256 DIMENSIONS
        self.user_tower = nn.Sequential(
            nn.LazyLinear(512), # LazyLinear like the item tower, input is the output of user_features (43 wide for now)
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256)
        )
        

        # ========== LOG Q  ===========

        # loaded once, buffer so it moves with the model to the gpu. indexed by article row position
        log_q = pd.read_parquet(IDX_DIR / 'articles_indexed.parquet', columns=['item_log_prob'])['item_log_prob'].to_numpy()
        self.register_buffer('log_q', torch.tensor(log_q, dtype=torch.float32))

        # ===== CATEGORICAL EMBEDDINGS =====

        # one embedding table per categorical col, sizes taken from the lookup tables
        # ModuleDict (not a plain dict) so the weights are registered as parameters
        with open(IDX_DIR / 'item_emb_lookup.json') as f:
            item_lookup = json.load(f)
        with open(IDX_DIR / 'user_emb_lookup.json') as f:
            user_lookup = json.load(f)

        self.item_embs = nn.ModuleDict({
            col: nn.Embedding(len(mapping), emb_dim(len(mapping)))
            for col, mapping in item_lookup.items()
        })
        self.user_embs = nn.ModuleDict({
            col: nn.Embedding(len(mapping), emb_dim(len(mapping)))
            for col, mapping in user_lookup.items()
        })

        # ===== ITEM FEATURE TABLE =====

        # articles_indexed loaded once as buffers, row position = item id
        articles = pd.read_parquet(IDX_DIR / 'articles_indexed.parquet', columns=list(self.item_embs) + ['median_price'])

        # (N_items, n_cat_cols), cols in the same order as self.item_embs
        self.register_buffer('item_cat', torch.tensor(articles[list(self.item_embs)].to_numpy(), dtype=torch.long))

        # standardized log median price, (N_items,)
        log_price = torch.log(torch.tensor(articles['median_price'].to_numpy(), dtype=torch.float32))
        self.register_buffer('item_price', (log_price - log_price.mean()) / log_price.std())

        # ===== USER FEATURE TABLE =====

        # customers_indexed loaded once as a buffer, row position = user id
        customers = pd.read_parquet(IDX_DIR / 'customers_indexed.parquet', columns=list(self.user_embs))

        # (N_users, n_cat_cols), cols in the same order as self.user_embs
        self.register_buffer('user_cat', torch.tensor(customers.to_numpy(), dtype=torch.long))


        # ======= TEXT EMBEDDINGS =======




        # ======= IMAGE EMBEDDINGS ======



    # ========= FORWARD PASS =========
    def item_forward(self, x):
        '''
        x: input
        '''

        return F.normalize(self.item_tower(x)) # item emb, normalized

    def user_forward(self, x):
        '''
        x: input
        '''

        return F.normalize(self.user_tower(x)) # user emb, normalized

    def all_forward(self, user_feat, item_feat):
        '''
        user_feat: input for the user tower
        item_feat: input for the item tower
        '''

        item_emb = self.item_forward(item_feat)
        user_emb = self.user_forward(user_feat)

        # cosine similarity of embeddings
        sim = user_emb @ item_emb.T # (B,B), users = rows, items = cols
        return sim


    # ========= TRANSFORM RAW DATA INTO INPUTS FOR EACH TOWER =========
    def item_features(self, x):
        '''
        x: batch of item_ids (row positions in articles_indexed.parquet), LongTensor (B,)
        returns: FloatTensor (B, sum of item emb dims + 1)
        '''
        rows = self.item_cat[x] # (B, n_cat_cols). the rows of the items in the batch with the categorical columns

        # for each embedding table in the items data, do a weight lookup on the whole batch
        embs = [emb(rows[:, i]) for i, emb in enumerate(self.item_embs.values())]

        return torch.cat(embs + [self.item_price[x].unsqueeze(1)], dim=1) # add on median price as a feature to the items


    def user_features(self, x):
        '''
        x: batch of user_ids (row positions in customers_indexed.parquet), LongTensor (B,)
        returns: FloatTensor (B, sum of user emb dims)
        '''
        rows = self.user_cat[x] # (B, n_cat_cols). the rows of the users in the batch and only the categorical columns

        # for each embedding table in user data, do a weight lookup on the batch
        embs = [emb(rows[:, i]) for i, emb in enumerate(self.user_embs.values())]

        return torch.cat(embs, dim=1)



    def logq_loss(self, sim, T, item_ids):
        '''
        sim: cosine similarity measure, size (B,B)
        T: temperature
        log_q: log q correction
        item_ids: item row positions in articles_indexed.parquet for the current batch, size (B,)
        '''

        logits = sim/T

        logits = logits - self.log_q[item_ids] # (B,B) - (B,), subtracts each item's log q from its column

        # mask duplicate items that aren't the true positive (false negatives)
        same_item = item_ids.unsqueeze(0) == item_ids.unsqueeze(1) # builds the square matrix which checks for duplicates
        eye = torch.eye(len(item_ids), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(same_item & ~eye, float("-inf")) # fill duplicates with -inf so they have no influence on the softmax

        # user i's positive is item i, so the true class for row i is column i (the diagonal)
        labels = torch.arange(logits.size(0), device=logits.device)

        return F.cross_entropy(logits, labels)

    def fit(self, data):
        '''
        data: a Dataset object
        '''
        pass



