import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from hm_clothes_rec.paths import INDEXED_DIR as IDX_DIR


def emb_dim(n):
    '''
    rule of thumb embedding size for a categorical col with n unique values, capped at 32
    '''
    return min(32, round(1.6 * n ** 0.56))


class TwoTowerRetrieval(nn.Module):
    '''
    Two tower model for retrieval in recommendation systems.
    '''

    def __init__(self, max_hist=30, use_clip=True, clip_dim=64):
        '''
        max_hist: number of most recent purchases used as a user's history.
                  history is cut per (user, day) pair: only purchases strictly before that day count,
                  so every transaction is both a training pair and history for the user's later pairs
        use_clip: add the frozen FashionCLIP item vectors (clip_item_emb.npy, from image_text_emb.embedder) to the item features
        clip_dim: width the 512 dim CLIP vector is projected down to before it is concatenated
        '''
        super().__init__() # necessary to use nn.Module

        # constructor args, saved with the weights so load() rebuilds the same layer sizes
        self.config = {'max_hist': max_hist, 'use_clip': use_clip, 'clip_dim': clip_dim}

        # data tables (log_q, item/user features, CLIP vectors, purchase history) are registered with persistent=False:
        # they move to the gpu with the model but are not saved, __init__ reloads them from IDX_DIR

        # ========== LOG Q  ===========

        # loaded once, buffer so it moves with the model to the gpu. indexed by article row position
        log_q = pd.read_parquet(IDX_DIR / 'articles_indexed.parquet', columns=['item_log_prob'])['item_log_prob'].to_numpy()
        self.register_buffer('log_q', torch.tensor(log_q, dtype=torch.float32), persistent=False)

        # ===== CATEGORICAL EMBEDDINGS =====

        # read the lookup tables; they contain the mappings of {col: {unique_val: enumerated_idx}}
        with open(IDX_DIR / 'item_emb_lookup.json') as f:
            item_lookup = json.load(f)
        with open(IDX_DIR / 'user_emb_lookup.json') as f:
            user_lookup = json.load(f)
        
        # ModuleDict (not a plain dict) so the weights are registered as parameters
        # makes dictionary where it is col: [num_maps, emb_dim(num_maps)]
        self.item_embs = nn.ModuleDict({
            col: nn.Embedding(len(mapping), emb_dim(len(mapping)))
            for col, mapping in item_lookup.items()
        })
        self.user_embs = nn.ModuleDict({
            col: nn.Embedding(len(mapping), emb_dim(len(mapping)))
            for col, mapping in user_lookup.items()
        })

        # ======================= ITEM FEATURE TABLE =========================

        # read article data, row position = item id
        articles = pd.read_parquet(IDX_DIR / 'articles_indexed.parquet', columns=list(self.item_embs) + ['median_price'])

        # makes categorical columns into a buffer (N_items, n_cat_cols)
        self.register_buffer('item_cat', torch.tensor(articles[list(self.item_embs)].to_numpy(), dtype=torch.long), persistent=False)

        # standardized log median price, (N_items,)
        log_price = torch.log(torch.tensor(articles['median_price'].to_numpy(), dtype=torch.float32))
        self.register_buffer('item_price', (log_price - log_price.mean()) / log_price.std(), persistent=False)

        # ======================== CLIP ITEM VECTORS ==================================

        # frozen FashionCLIP vectors (N_items, 512), row = item id: image emb, or text emb where there is no image.
        # a buffer, so the optimizer never updates them; only the LayerNorm + projection below train
        self.use_clip = use_clip
        self.clip_dim = clip_dim if use_clip else 0
        if use_clip:
            clip = np.load(IDX_DIR / 'clip_item_emb.npy')
            assert clip.shape[0] == len(articles), f'clip_item_emb has {clip.shape[0]} rows, expected {len(articles)} items'
            self.register_buffer('item_clip', torch.from_numpy(clip).float(), persistent=False)
            self.clip_norm = nn.LayerNorm(clip.shape[1])
            self.clip_proj = nn.Linear(clip.shape[1], clip_dim) # keeps CLIP from drowning out the categorical embs

        # ======================== USER FEATURE TABLE ==================================

        # customers_indexed loaded once as a buffer, row position = user id
        customers = pd.read_parquet(IDX_DIR / 'customers_indexed.parquet', columns=list(self.user_embs))

        # register an array (N_users, n_cat_cols) as a buffer
        self.register_buffer('user_cat', torch.tensor(customers.to_numpy(), dtype=torch.long), persistent=False)

        # ===== USER PURCHASE HISTORY =====

        # query the transactions parquet -> select ids and days from 1970
        # -> order by users, day so we get continuous blocks of users ordered by day of purchase
        self.max_hist = max_hist
        tx = duckdb.sql(f'''
            SELECT user_id, item_id, (t_dat - DATE '1970-01-01')::BIGINT AS day
            FROM read_parquet('{IDX_DIR / 'transactions_indexed.parquet'}')
            ORDER BY user_id, day, item_id
        ''').fetchnumpy() # item_id breaks same day ties, otherwise duckdb's order (and so which items make the max_hist cut) changes every build
        tx_user, tx_day = torch.from_numpy(tx['user_id']), torch.from_numpy(tx['day'])

        # create a user-day key; user_id * day_mult determines a unique user, and + tx_day-day0 determines the days from day0.
        self.day0 = int(tx_day.min())
        self.day_mult = int(tx_day.max()) - self.day0 + 2
        self.register_buffer('tx_key', tx_user * self.day_mult + (tx_day - self.day0), persistent=False)

        # item bought in each transaction, int32 to halve memory (31M rows), cast to long at lookup time
        self.register_buffer('tx_item', torch.from_numpy(tx['item_id']).to(torch.int32), persistent=False)

        # SELF ATTENTION
        # self attention over the purchase history, item feature width in and out (202 without CLIP, 266 with clip_dim=64)
        item_feat_dim = sum(e.embedding_dim for e in self.item_embs.values()) + 1 + self.clip_dim # +1 for price
        self.attention = nn.MultiheadAttention(embed_dim=item_feat_dim, num_heads=2, batch_first=True)
        hist_dim = 40
        self.hist_proj = nn.Linear(item_feat_dim, hist_dim) # project pooled history down to balance with the 43 wide user features

        # input widths are fixed by the features above: item = item_features width, user = user cat embs + history
        user_feat_dim = sum(e.embedding_dim for e in self.user_embs.values()) + hist_dim # 43 + 40 = 83

        # ========== ITEM TOWER ===========

        # RETURN EMBEDDING WITH 256 DIMENSIONS
        self.item_tower = nn.Sequential(
            nn.Linear(item_feat_dim, 1024), # 202 without CLIP, 266 with clip_dim=64
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
            nn.Linear(user_feat_dim, 1024), # 202 without CLIP, 266 with clip_dim=64
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256)
        )
        

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
        returns: FloatTensor (B, sum of item emb dims + 1 + clip_dim)
        '''
        rows = self.item_cat[x] # (B, n_cat_cols). chooses items in the item_ids, and only their categorical cols.

        # for each column i, map the index to a embedding vector -> list of (B, emb_i_dims)
        embs = [emb(rows[:, i]) for i, emb in enumerate(self.item_embs.values())]
        feats = embs + [self.item_price[x].unsqueeze(1)] # add on median price as a feature to the items

        # CLIP: normalize + project on the forward pass so the LayerNorm and projection train
        if self.use_clip:
            feats.append(self.clip_proj(self.clip_norm(self.item_clip[x]))) # (B, clip_dim)

        # concatenate all the features into one item input
        return torch.cat(feats, dim=1)

    def user_history(self, x, day):
        '''
        x: batch of user_ids (row positions in customers_indexed.parquet), LongTensor (B,)
        day: days since 1970-01-01, LongTensor (B,) or an int for the whole batch.
        returns: FloatTensor (B, 40), self attention over each user's recent purchases, mean pooled and projected
        '''
        B, L = x.size(0), self.max_hist
        day = torch.as_tensor(day, device=x.device).expand(B) # makes (B, ) for day.

        # where this user's block starts, and where day `day` starts inside it (= one past their last earlier purchase)
        start = torch.searchsorted(self.tx_key, x * self.day_mult) # x * day_mult = smallest possible key for user x (day offset 0) -> index of the user's first purchase

        end = torch.searchsorted(self.tx_key, x * self.day_mult + (day - self.day0).clamp(0, self.day_mult - 1)) # key for (user x, day) -> index of the user's first purchase ON or AFTER day, so everything before it is history

        lens = (end - start).clamp(max=L) # (B,) each entry is the number of purchases made , capped at L. 

        # the L slots ending at `end`, oldest first. slots before the user's block are padding (left padded)
        idx = end.unsqueeze(1) - L + torch.arange(L, device=x.device) # (B, L); gets the user's last L purchases
        pad = torch.arange(L, device=x.device).unsqueeze(0) < (L - lens).unsqueeze(1) # True = padding, marks those as NOT a user's purchases before.
        ids = self.tx_item[idx.clamp(min=0)].long() # gets stuff in the L window slots

        # one item_features call for the whole batch, then back to (B, L, 202)
        h = self.item_features(ids.flatten()).view(B, L, -1)

        # users with no history would have every key masked -> softmax over nothing -> NaN.
        # unmask their last slot so attention is defined, their output is zeroed below anyway
        empty = lens == 0
        pad[:, -1] &= ~empty

        # take h and mask using pad -> attention
        out, _ = self.attention(h, h, h, key_padding_mask=pad, need_weights=False) # (B, L, 202)

        # mean over the real positions only
        keep = (~pad).unsqueeze(2).float() # (B, L, 1)
        pooled = (out * keep).sum(1) / keep.sum(1) # (B, 202)
        pooled = pooled.masked_fill(empty.unsqueeze(1), 0.0) # no history -> zero vector

        # a projected mean pooled attention condenses historical preference of user over time. 
        return self.hist_proj(pooled)

    def user_features(self, x, day):
        '''
        x: batch of user_ids (row positions in customers_indexed.parquet), LongTensor (B,)
        day: days since 1970-01-01 to cut the history at, LongTensor (B,) or int (see user_history)
        returns: FloatTensor (B, sum of user emb dims + 40), categorical embs then the history vector
        '''
        rows = self.user_cat[x] # (B, n_cat_cols). the rows of the users in the batch and only the categorical columns

        # for each embedding table in user data, do a weight lookup on the batch

        embs = [emb(rows[:, i]) for i, emb in enumerate(self.user_embs.values())]

        # concat the purchase history vector onto the user input
        return torch.cat(embs + [self.user_history(x, day)], dim=1)



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
    def batch_loss(self, item_ids, user_ids, days, T):
        '''
        item_ids, user_ids, days: LongTensors (B,), already on the model's device
        returns: scalar logq corrected in-batch softmax loss
        '''
        sim = self(self.user_features(user_ids, days), self.item_features(item_ids)) # (B,B)
        return self.logq_loss(sim, T, item_ids)

    @staticmethod
    def iter_batches(user_ids, item_ids, days, batch_size, shuffle):
        '''
        user_ids, item_ids, days: LongTensors (N,), already on the model's device
        yields (item_ids, user_ids, days) batches of exactly batch_size by slicing, no per row python work
        the last partial batch is dropped so every batch has the same number of in-batch negatives
        '''
        n = len(item_ids)
        order = torch.randperm(n, device=item_ids.device) if shuffle else torch.arange(n, device=item_ids.device)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start:start + batch_size]
            yield item_ids[idx], user_ids[idx], days[idx]

    def train_epoch(self, user_ids, item_ids, days, optimizer, T, batch_size, desc='train', log_every=50):
        '''
        one shuffled pass over the training pairs, returns the mean batch loss
        desc: progress bar label
        log_every: refresh the running loss on the bar every this many batches
        '''
        self.train() # turn dropout on

        # accumulate on device, calling .item() every step would force a gpu sync per batch
        total = torch.zeros((), device=item_ids.device)
        n_batches = 0
        batches = self.iter_batches(user_ids, item_ids, days, batch_size, shuffle=True)
        pbar = tqdm(batches, total=len(item_ids) // batch_size, desc=desc, leave=False) # leave=False: the epoch summary line replaces it
        for b_items, b_users, b_days in pbar:
            optimizer.zero_grad()
            loss = self.batch_loss(b_items, b_users, b_days, T)
            loss.backward()
            optimizer.step()

            total += loss.detach()
            n_batches += 1

            # only sync for the bar every log_every batches
            if n_batches % log_every == 0:
                pbar.set_postfix(loss=f'{total.item() / n_batches:.4f}')

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
    def _topk_chunks(self, user_ids, cutoff_day, k, item_emb, chunk_size):
        '''
        yields (users, scores, item_ids) per chunk of users, all on the model's device, best first:
        users (C,), scores (C, k) cosine similarity, item_ids (C, k)
        no log q here, the correction is only for the sampled training loss
        '''
        for start in range(0, len(user_ids), chunk_size):
            u = user_ids[start:start + chunk_size]
            sim = self.user_forward(self.user_features(u, cutoff_day)) @ item_emb.T # (C, N_items)
            scores, ids = sim.topk(k, dim=1)
            yield u, scores, ids

    @torch.no_grad()
    def predict(self, user_ids, cutoff_day, k=300, chunk_size=1024):
        '''
        top k retrieval candidates per user, for the ranker.
        user_ids: LongTensor (n_users,), row positions in customers_indexed.parquet
        cutoff_day: int, days since 1970-01-01. history is cut here (only purchases strictly before it count),
                    so use the first day of the week being predicted
        returns: (item_ids, scores) numpy arrays on the cpu, both (n_users, k), row i = user_ids[i], best first.
                 item_ids int32 (row positions in articles_indexed.parquet), scores float32 cosine similarity
        '''
        self.eval() # dropout off
        device = self.item_cat.device
        user_ids = torch.as_tensor(user_ids, dtype=torch.long, device=device)
        item_emb = self.all_item_embs() # (N_items, 256), computed once for every chunk

        # move each chunk to the cpu as it is done, (1.4M users, 300) would not fit next to the model on the gpu
        ids_out, scores_out = [], []
        for _, scores, ids in self._topk_chunks(user_ids, cutoff_day, k, item_emb, chunk_size):
            ids_out.append(ids.to(torch.int32).cpu())
            scores_out.append(scores.float().cpu())
        return torch.cat(ids_out).numpy(), torch.cat(scores_out).numpy()

    @torch.no_grad()
    def recall_at_k(self, user_ids, item_ids, ks, cutoff_day, chunk_size=1024):
        '''
        user_ids, item_ids: LongTensors (N,), the ground truth (user, item) purchase pairs
        ks: cutoffs, e.g. (12, 50, 100)
        cutoff_day: int, days since 1970-01-01. every user's history is cut here, normally the first day of the
                    eval window, so the model only sees what it would know when the recommendations are made
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
        chunks = self._topk_chunks(eval_users, cutoff_day, max_k, item_emb, chunk_size)
        for start, (u, _, topk) in zip(range(0, len(eval_users), chunk_size), chunks):
            denom = n_bought[start:start + chunk_size].float() # (C,)

            hits = torch.isin(u.unsqueeze(1) * n_items + topk, keys) # (C, max_k), True if bought
            cum_hits = hits.cumsum(dim=1) # hits within the first j+1 positions

            for k in ks:
                totals[k] += (cum_hits[:, k - 1] / denom).sum().item()

        return {f'recall@{k}': totals[k] / len(eval_users) for k in ks}

    @torch.no_grad()
    def eval_epoch(self, user_ids, item_ids, days, T, batch_size, ks=(12, 50, 100)):
        '''
        one pass over the validation pairs, no gradient updates
        returns: (mean batch loss, dict of recall@k over the full catalog)
        '''
        self.eval() # turn dropout off

        total = torch.zeros((), device=item_ids.device)
        n_batches = 0
        for b_items, b_users, b_days in self.iter_batches(user_ids, item_ids, days, batch_size, shuffle=False):
            total += self.batch_loss(b_items, b_users, b_days, T)
            n_batches += 1

        # recall: history cut at the start of the val window for everyone. uses every val row, not just the full batches
        recall = self.recall_at_k(user_ids, item_ids, ks, cutoff_day=int(days.min()))

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
        train_users, train_items, train_days = train_data.user_ids.to(device), train_data.item_ids.to(device), train_data.days.to(device)
        val_users, val_items, val_days = val_data.user_ids.to(device), val_data.item_ids.to(device), val_data.days.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        history = {'train_loss': [], 'val_loss': [], **{f'recall@{k}': [] for k in ks}}
        for e in range(epochs):
            train_loss = self.train_epoch(train_users, train_items, train_days, optimizer, T, batch_size, desc=f'epoch {e + 1}/{epochs}')
            val_loss, recall = self.eval_epoch(val_users, val_items, val_days, T, batch_size, ks)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            for name, value in recall.items():
                history[name].append(value)

            recall_str = ' | '.join(f'{name} {value:.4f}' for name, value in recall.items())
            print(f'epoch {e + 1}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | {recall_str}')

        return history

    # ========= SAVE / LOAD =========
    def save(self, path):
        '''
        saves the trained weights + constructor args only 
        '''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'config': self.config, 'state_dict': self.state_dict()}, path)

    @classmethod
    def load(cls, path, device=None):
        '''
        rebuilds the model with the saved args (reloading the data tables from IDX_DIR) and loads the weights.
        usage: model = TwoTowerRetrieval.load('models/two_tower.pt')
        '''
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = torch.load(path, map_location='cpu')
        model = cls(**ckpt['config'])
        model.load_state_dict(ckpt['state_dict'])
        return model.to(device).eval()


