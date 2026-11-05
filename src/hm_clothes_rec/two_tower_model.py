import torch
from torch import nn
from torch.nn import functional as F
import timm
import duckdb
import pandas


class TwoTowerRetrieval(nn.Module):
    '''
    Two tower model for retrieval in recommendation systems.
    '''

    def __init__(self):
        super().__init__() # necessary to use nn.Module

        # ========== ITEM TOWER ===========

        # RETURN EMBEDDING WITH 256 DIMENSIONS
        placeholder = 100 # TODO: decide on the input size

        self.item_tower = nn.Sequential(
            nn.Linear(placeholder, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 256)
        )


        # ========== USER TOWER ===========

        # RETURN EMBEDDING WITH 256 DIMENSIONS
        self.user_tower = None # build architecture here



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
        sim = item_emb @ user_emb.T
        return sim


    # ========= TRANSFORM RAW DATA INTO INPUTS FOR EACH TOWER =========
    def item_features():
        pass # fill these in

    def user_features():
        pass # fill these in



    # ====== LOSS FUNCTION =======

    # TODO: finish up loss function
    def logq(self, ):
        pass

    def logq_loss(self, sim, T, log_q):
        '''
        sim: similarity measure
        T: temperature
        log_q: log q correction
        '''

        logits = sim/T
        logits = logits - log_q
        return None 
