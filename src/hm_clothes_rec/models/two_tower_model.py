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

    def forward(self, user_feat, item_feat):
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


    # ========= TRAINING =========
    def batch_loss(self, item_ids, user_ids, T):
        '''
        item_ids, user_ids: LongTensors (B,), already on the model's device
        returns: scalar logq corrected in-batch softmax loss
        '''
        sim = self(self.user_features(user_ids), self.item_features(item_ids)) # (B,B)
        return self.logq_loss(sim, T, item_ids)

    @staticmethod
    def iter_batches(user_ids, item_ids, batch_size, shuffle):
        '''
        user_ids, item_ids: LongTensors (N,), already on the model's device
        yields (item_ids, user_ids) batches of exactly batch_size by slicing, no per row python work
        the last partial batch is dropped so every batch has the same number of in-batch negatives
        '''
        n = len(item_ids)
        order = torch.randperm(n, device=item_ids.device) if shuffle else torch.arange(n, device=item_ids.device)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start:start + batch_size]
            yield item_ids[idx], user_ids[idx]

    def train_epoch(self, user_ids, item_ids, optimizer, T, batch_size):
        '''
        one shuffled pass over the training pairs, returns the mean batch loss
        '''
        self.train() # turn dropout on

        # accumulate on device, calling .item() every step would force a gpu sync per batch
        total = torch.zeros((), device=item_ids.device)
        n_batches = 0
        for b_items, b_users in self.iter_batches(user_ids, item_ids, batch_size, shuffle=True):
            optimizer.zero_grad()
            loss = self.batch_loss(b_items, b_users, T)
            loss.backward()
            optimizer.step()

            total += loss.detach()
            n_batches += 1

        return total.item() / max(n_batches, 1)

    @torch.no_grad()
    def all_item_embs(self, chunk_size=8192):
        '''
        embeds the whole catalog, returns FloatTensor (N_items, 256), row position = item id
        '''
        n_items = self.item_cat.size(0)
        embs = []
        for start in range(0, n_items, chunk_size):
            ids = torch.arange(start, min(start + chunk_size, n_items), device=self.item_cat.device)
            embs.append(self.item_forward(self.item_features(ids)))
        return torch.cat(embs)

    @torch.no_grad()
    def recall_at_k(self, user_ids, item_ids, ks, chunk_size=1024):
        '''
        user_ids, item_ids: LongTensors (N,), the ground truth (user, item) purchase pairs
        ks: cutoffs, e.g. (12, 50, 100)
        returns: dict {'recall@k': float}, same definition as evaluate.recall_at_k:
                 per user |top k ∩ bought| / |bought|, averaged over every user in the pairs
        '''
        self.eval()
        n_items = self.item_cat.size(0)
        item_emb = self.all_item_embs() # (N_items, 256)

        # encode each distinct (user, item) pair as one int so membership is a single isin
        device = self.item_cat.device
        keys = torch.unique(user_ids.to(device) * n_items + item_ids.to(device)) # sorted, duplicates removed

        # distinct users and how many distinct items each bought (keys are sorted by user)
        eval_users, n_bought = torch.unique(keys // n_items, return_counts=True)

        max_k = max(ks)
        totals = dict.fromkeys(ks, 0.0)
        for start in range(0, len(eval_users), chunk_size):
            u = eval_users[start:start + chunk_size] # (C,)
            denom = n_bought[start:start + chunk_size].float() # (C,)

            # no log q here, the correction is only for the sampled training loss
            scores = self.user_forward(self.user_features(u)) @ item_emb.T # (C, N_items)
            topk = scores.topk(max_k, dim=1).indices # (C, max_k), best first

            hits = torch.isin(u.unsqueeze(1) * n_items + topk, keys) # (C, max_k), True if bought
            cum_hits = hits.cumsum(dim=1) # hits within the first j+1 positions

            for k in ks:
                totals[k] += (cum_hits[:, k - 1] / denom).sum().item()

        return {f'recall@{k}': totals[k] / len(eval_users) for k in ks}

    @torch.no_grad()
    def eval_epoch(self, user_ids, item_ids, T, batch_size, ks=(12, 50, 100)):
        '''
        one pass over the validation pairs, no gradient updates
        returns: (mean batch loss, dict of recall@k over the full catalog)
        '''
        self.eval() # turn dropout off

        total = torch.zeros((), device=item_ids.device)
        n_batches = 0
        for b_items, b_users in self.iter_batches(user_ids, item_ids, batch_size, shuffle=False):
            total += self.batch_loss(b_items, b_users, T)
            n_batches += 1

        recall = self.recall_at_k(user_ids, item_ids, ks) # uses every val row, not just the full batches

        return total.item() / max(n_batches, 1), recall

    def fit(self, train_data, val_data, epochs, batch_size=2048, lr=1e-3, weight_decay=0.0, T=0.05, ks=(12, 50, 100), device=None):
        '''
        train_data, val_data: TransactionDataset objects
        epochs: number of passes over the training data
        batch_size: also the number of candidates per user in the loss (B-1 in-batch negatives)
        T: softmax temperature for the logq loss
        ks: cutoffs for val recall@k
        returns: dict with per epoch train loss, val loss and val recall@k
        '''
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # move the id columns to the device once, batches are then just index slices
        train_users, train_items = train_data.user_ids.to(device), train_data.item_ids.to(device)
        val_users, val_items = val_data.user_ids.to(device), val_data.item_ids.to(device)

        # dry run so the LazyLinear layers get their shapes before the optimizer sees the parameters
        with torch.no_grad():
            dummy = torch.zeros(2, dtype=torch.long, device=device)
            self(self.user_features(dummy), self.item_features(dummy))

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {'train_loss': [], 'val_loss': [], **{f'recall@{k}': [] for k in ks}}
        for e in range(epochs):
            train_loss = self.train_epoch(train_users, train_items, optimizer, T, batch_size)
            val_loss, recall = self.eval_epoch(val_users, val_items, T, batch_size, ks)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            for name, value in recall.items():
                history[name].append(value)

            recall_str = ' | '.join(f'{name} {value:.4f}' for name, value in recall.items())
            print(f'epoch {e + 1}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {recall_str}')

        return history


