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
        self.item_tower = None # build architecture here


        # ========== USER TOWER ===========

        # RETURN EMBEDDING WITH 256 DIMENSIONS
        self.user_tower = None # build architecture here



    # ========= FORWARD PASS =========
    def item_forward(self, x):
        return F.normalize(self.item_tower(x)) # item emb, normalized

    def user_forward(self, x):
        return F.normalize(self.user_tower(x)) # user emb, normalized

    def forward(self, user_feat, item_feat):
        item_emb = self.item_forward(item_feat)
        user_emb = self.user_forward(user_feat)
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

    def logq_loss(self, sim, temp, log_q):
        logits = sim/temp 
        logits = logits - log_q
        return None 
