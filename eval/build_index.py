from __future__ import annotations

from typing import Tuple

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.inference_mode()
def encode_dataset(
    model,
    dataset,
    batch_size: int = 512,
    device: torch.device | str = "cuda",
    mode: str = "image",   # "image" | "text"
) -> np.ndarray:
    """
    Encode an _ImageSubset or _TextSubset into a float32 numpy array.

    Embeddings are L2-normalised inside the model, so the returned array
    is already unit-norm — suitable for inner-product (cosine) search.
    """
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    all_feats: list[np.ndarray] = []

    model.eval()
    for batch in tqdm(loader, desc=f"Encoding {mode}s", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        if mode == "image":
            feats = model.encode_image(batch["pixel_values"])
        else:
            feats = model.encode_text(batch["input_ids"], batch["attention_mask"])
        all_feats.append(feats.cpu().float().numpy())

    return np.concatenate(all_feats, axis=0)   # (N, D)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an exact inner-product FAISS index.

    Since embeddings are L2-normalised, inner product == cosine similarity.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index


def build_retrieval_indices(
    model,
    eval_dataset,          # RetrievalDataset (eval split)
    batch_size: int = 512,
    device: torch.device | str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, faiss.IndexFlatIP, faiss.IndexFlatIP]:
    """
    Encode both modalities and return embeddings + FAISS indices.

    Returns:
        img_embs   : (N_images, D)
        txt_embs   : (N_texts,  D)
        img_index  : FAISS index over image embeddings (for t2i)
        txt_index  : FAISS index over text  embeddings (for i2t)
    """
    img_embs = encode_dataset(
        model, eval_dataset.image_dataset, batch_size, device, mode="image"
    )
    txt_embs = encode_dataset(
        model, eval_dataset.text_dataset, batch_size, device, mode="text"
    )
    img_index = build_faiss_index(img_embs)
    txt_index = build_faiss_index(txt_embs)
    return img_embs, txt_embs, img_index, txt_index
