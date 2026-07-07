#!/usr/bin/env python3
"""
Evaluate retrieval R@K on the test split.

Zero-shot baseline (no fine-tuning):
    python evaluate.py --config configs/colab.yaml

Fine-tuned adapter:
    python evaluate.py --config configs/colab.yaml --adapter /content/checkpoints/best
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizerFast

from config import Config
from data.dataset import build_dataset
from data.transforms import build_train_transform, build_val_transform
from eval.build_index import build_retrieval_indices
from eval.metrics import compute_retrieval_metrics, format_metrics
from model.clip_lora import CLIPWithLoRA


class ZeroShotCLIP(nn.Module):
    """Plain pretrained CLIP with the same encode interface as CLIPWithLoRA."""

    def __init__(self, backbone: str):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(backbone)

    def encode_image(self, pixel_values, normalize: bool = True):
        out = self.clip.vision_model(pixel_values=pixel_values)
        feats = self.clip.visual_projection(out.pooler_output)
        return F.normalize(feats, dim=-1) if normalize else feats

    def encode_text(self, input_ids, attention_mask, normalize: bool = True):
        out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feats = self.clip.text_projection(out.pooler_output)
        return F.normalize(feats, dim=-1) if normalize else feats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", default=None,
                   help="Path to a saved LoRA adapter. Omit for zero-shot baseline.")
    p.add_argument("--split", default="test", choices=["val", "test"])
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.adapter:
        print(f"Loading fine-tuned adapter: {args.adapter}")
        model = CLIPWithLoRA.load_adapter(cfg.model, args.adapter)
    else:
        print("Zero-shot baseline (no fine-tuning)")
        model = ZeroShotCLIP(cfg.model.backbone)
    model.eval().to(device)

    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.model.backbone)
    ds = build_dataset(
        cfg.data, args.split,
        build_train_transform(cfg.data.image_size),
        build_val_transform(cfg.data.image_size),
        tokenizer,
    )
    print(f"{args.split}: {len(ds.image_paths):,} images / {len(ds.captions):,} captions")

    img_embs, txt_embs, img_idx, txt_idx = build_retrieval_indices(
        model, ds, cfg.eval.batch_size, device
    )
    metrics = compute_retrieval_metrics(
        img_idx, txt_idx, img_embs, txt_embs,
        ds.gt_i2t, ds.gt_t2i, cfg.eval.top_k,
    )
    print()
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
