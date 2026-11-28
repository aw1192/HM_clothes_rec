'''
Image and text embedder.
Encodes every article with FashionCLIP and saves (N_items, 512) float32 arrays whose row i is item_id i,
so the two tower model can register them directly as a frozen buffer.
'''

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from hm_clothes_rec.paths import CLEANED_DIR, IMG_DIR
from hm_clothes_rec.paths import INDEXED_DIR as IDX_DIR # the copy every model reads from

# same weights the fashion-clip package loads; we load them straight from transformers because
# fashion-clip 0.2.2 passes use_auth_token, which transformers 5 removed
MODEL_NAME = 'patrickjohncyh/fashion-clip'


def _features(out):
    # get_*_features returns a tensor in older transformers, an output object (pooler_output) in newer ones
    return out if torch.is_tensor(out) else out.pooler_output


class FashionCLIPEncoder:
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    @torch.no_grad()
    def encode_text(self, texts, batch_size=64):
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc='text'):
            batch = self.processor(text=texts[i:i + batch_size], return_tensors='pt',
                                   padding=True, truncation=True, max_length=77).to(self.device)
            embs.append(_features(self.model.get_text_features(**batch)).float().cpu().numpy())
        return np.concatenate(embs)

    @torch.no_grad()
    def encode_images(self, paths, batch_size=64):
        embs = []
        for i in tqdm(range(0, len(paths), batch_size), desc='image'):
            imgs = [Image.open(p).convert('RGB') for p in paths[i:i + batch_size]]
            batch = self.processor(images=imgs, return_tensors='pt').to(self.device)
            embs.append(_features(self.model.get_image_features(**batch)).float().cpu().numpy())
        return np.concatenate(embs)


def article_text(df):
    '''
    create one caption per article, e.g. "Strap top. Black Vest top. Jersey top with narrow shoulder straps."
    '''
    
    desc = df['detail_desc'].fillna('').replace('unknown', '')
    return (df['prod_name'] + '. ' + df['colour_group_name'] + ' ' + df['product_type_name'] + '. ' + desc).str.strip().tolist()


def l2_normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def embedder(out_dir=IDX_DIR, batch_size=64, overwrite=False):
    '''
    writes to out_dir:
        clip_text_emb.npy  (N, 512) text embedding for every article
        clip_image_emb.npy (N, 512) image embedding, zero row where has_image == 0
        clip_item_emb.npy  (N, 512) image embedding, falling back to the text embedding where there is no image
    all rows are L2-normalized (the CLIP space is cosine), dtype float32, row i = item_id i
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = {name: out_dir / f'clip_{name}_emb.npy' for name in ('text', 'image', 'item')}
    if not overwrite and all(p.exists() for p in outs.values()):
        print('embeddings already exist, pass overwrite=True to recompute')
        return {name: np.load(p) for name, p in outs.items()}

    # indexed articles give item_id order + has_image; cleaned articles still hold the raw strings for the captions
    items = pd.read_parquet(IDX_DIR / 'articles_indexed.parquet', columns=['item_id', 'article_id', 'has_image'])
    text_cols = ['article_id', 'prod_name', 'colour_group_name', 'product_type_name', 'detail_desc']
    raw = pd.read_parquet(CLEANED_DIR / 'articles_clean.parquet', columns=text_cols)
    items = items.merge(raw, on='article_id', how='left', validate='one_to_one').sort_values('item_id')
    assert (items['item_id'].to_numpy() == np.arange(len(items))).all(), 'row position must equal item_id'

    fclip = FashionCLIPEncoder()
    print(f'FashionCLIP on {fclip.device}')

    # TEXT: every article has a caption
    text_emb = l2_normalize(fclip.encode_text(article_text(items), batch_size=batch_size)).astype(np.float32)

    # IMAGE: only articles whose image exists, the rest stay zero
    has_img = items['has_image'].to_numpy().astype(bool)
    paths = [str(IMG_DIR / aid[:3] / f'{aid}.webp') for aid in items.loc[has_img, 'article_id']]
    image_emb = np.zeros_like(text_emb)
    image_emb[has_img] = l2_normalize(fclip.encode_images(paths, batch_size=batch_size))

    # ITEM: image where available, else text (both live in the same CLIP space)
    item_emb = np.where(has_img[:, None], image_emb, text_emb).astype(np.float32)

    result = {'text': text_emb, 'image': image_emb, 'item': item_emb}
    for name, arr in result.items():
        np.save(outs[name], arr)
        print(f'saved {outs[name]} {arr.shape}')
    return result


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    embedder()
