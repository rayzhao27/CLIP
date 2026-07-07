from __future__ import annotations

from typing import Dict, List

import faiss
import numpy as np


def recall_at_k(
    index: faiss.IndexFlatIP,
    query_embs: np.ndarray,
    ground_truth: List[List[int]],
    k: int,
) -> float:
    """
    Compute Recall@k.

    Args:
        index:        FAISS index over the gallery embeddings.
        query_embs:   (N_queries, D) float32, L2-normalised.
        ground_truth: ground_truth[i] = list of gallery indices that are
                      positive matches for query i.
        k:            number of retrieved results to consider.

    Returns:
        Recall@k in [0, 1].
    """
    _, top_k_indices = index.search(query_embs.astype(np.float32), k)  # (N, k)
    hits = 0
    for i, gt in enumerate(ground_truth):
        gt_set = set(gt)
        if gt_set.intersection(top_k_indices[i]):
            hits += 1
    return hits / len(ground_truth)


def compute_retrieval_metrics(
    img_index: faiss.IndexFlatIP,
    txt_index: faiss.IndexFlatIP,
    img_embs: np.ndarray,
    txt_embs: np.ndarray,
    gt_i2t: List[List[int]],
    gt_t2i: List[List[int]],
    top_k: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute bidirectional R@K.

    i2t: each image queries the text gallery.
    t2i: each caption queries the image gallery.

    Returns a flat dict, e.g.:
        {"i2t/R@1": 0.72, "i2t/R@5": 0.92, ..., "t2i/R@1": 0.54, ...}
    """
    metrics: Dict[str, float] = {}
    max_k = max(top_k)

    # i2t
    _, i2t_top = txt_index.search(img_embs.astype(np.float32), max_k)
    for k in top_k:
        hits = sum(
            bool(set(gt_i2t[i]).intersection(i2t_top[i, :k]))
            for i in range(len(img_embs))
        )
        metrics[f"i2t/R@{k}"] = hits / len(img_embs)

    # t2i
    _, t2i_top = img_index.search(txt_embs.astype(np.float32), max_k)
    for k in top_k:
        hits = sum(
            bool(set(gt_t2i[i]).intersection(t2i_top[i, :k]))
            for i in range(len(txt_embs))
        )
        metrics[f"t2i/R@{k}"] = hits / len(txt_embs)

    # Convenience aggregate: mean R@1
    metrics["mean_R@1"] = (metrics["i2t/R@1"] + metrics["t2i/R@1"]) / 2.0

    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    lines = ["Retrieval Results"]
    lines.append(f"  i2t  R@1={metrics.get('i2t/R@1', 0):.4f}  "
                 f"R@5={metrics.get('i2t/R@5', 0):.4f}  "
                 f"R@10={metrics.get('i2t/R@10', 0):.4f}")
    lines.append(f"  t2i  R@1={metrics.get('t2i/R@1', 0):.4f}  "
                 f"R@5={metrics.get('t2i/R@5', 0):.4f}  "
                 f"R@10={metrics.get('t2i/R@10', 0):.4f}")
    lines.append(f"  mean R@1 = {metrics.get('mean_R@1', 0):.4f}")
    return "\n".join(lines)
