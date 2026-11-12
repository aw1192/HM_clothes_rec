# run from the repo root: PYTHONPATH=src python -m pytest -q tests/test_user_tower.py
import pytest
import torch

from hm_clothes_rec.models.two_tower_model import IDX_DIR, TwoTowerRetrieval


# user_emb_lookup.json is the last file index_data.py writes, so it implies the other three exist
@pytest.mark.skipif(not (IDX_DIR / 'user_emb_lookup.json').exists(), reason='data/indexed not built')
def test_user_tower_end_to_end():
    torch.manual_seed(0)
    model = TwoTowerRetrieval()
    user_ids = torch.arange(64)
    item_ids = torch.arange(64)

    # ids -> features -> tower -> 256-d unit vectors
    user_emb = model.user_forward(model.user_features(user_ids))
    assert user_emb.shape == (64, 256)
    assert torch.allclose(user_emb.norm(dim=1), torch.ones(64), atol=1e-5)

    # both towers -> (B,B) sim -> loss -> gradients reach the user tower and the user embedding tables
    sim = model.all_forward(model.user_features(user_ids), model.item_features(item_ids))
    assert sim.shape == (64, 64)
    assert sim.abs().max() <= 1 + 1e-5 # cosine similarity, so both sides are normalized
    # unequal batch sizes pin the orientation: users = rows, items = cols
    assert model.all_forward(model.user_features(user_ids[:3]), model.item_features(item_ids)).shape == (3, 64)
    loss = model.logq_loss(sim, 0.05, item_ids)
    assert torch.isfinite(loss)
    loss.backward()
    for p in list(model.user_tower.parameters()) + list(model.user_embs.parameters()):
        assert p.grad is not None and p.grad.abs().sum() > 0
