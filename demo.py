#!/usr/bin/env python3
"""
Gradio demo: type a caption, retrieve the top-k matching images.

    python demo.py --config configs/colab.yaml --adapter /content/checkpoints/best
"""
from __future__ import annotations

import argparse

import faiss
import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizerFast

from config import Config
from data.dataset import build_dataset
from data.transforms import build_train_transform, build_val_transform
from eval.build_index import build_faiss_index, encode_dataset
from model.clip_lora import CLIPWithLoRA


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--share", action="store_true", default=True)
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading adapter: {args.adapter}")
    model = CLIPWithLoRA.load_adapter(cfg.model, args.adapter)
    model.eval().to(device)

    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.model.backbone)
    ds = build_dataset(
        cfg.data, args.split,
        build_train_transform(cfg.data.image_size),
        build_val_transform(cfg.data.image_size),
        tokenizer,
    )

    print(f"Indexing {len(ds.image_paths):,} images from the {args.split} split...")
    img_embs = encode_dataset(model, ds.image_dataset, cfg.eval.batch_size, device, "image")
    index = build_faiss_index(img_embs)
    print("Index ready.")

    @torch.inference_mode()
    def search(query: str):
        enc = tokenizer(
            [query],
            max_length=cfg.data.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        feats = model.encode_text(enc["input_ids"], enc["attention_mask"])
        feats = feats.cpu().float().numpy()
        scores, ids = index.search(feats, args.top_k)
        return [
            (Image.open(ds.image_paths[i]).convert("RGB"), f"{s:.3f}")
            for i, s in zip(ids[0], scores[0])
        ]

    app = gr.Interface(
        fn=search,
        inputs=gr.Textbox(label="Describe an image", placeholder="a dog catching a frisbee on the beach"),
        outputs=gr.Gallery(label="Top matches", columns=4, height="auto"),
        title="CLIP + LoRA Image Search",
        description=f"Text-to-image retrieval over {len(ds.image_paths):,} "
                    f"Flickr30k {args.split} images. Fine-tuned CLIP ViT-B/32 with LoRA.",
        examples=[
            "a dog catching a frisbee on the beach",
            "two people riding bicycles down a city street",
            "a child in a red shirt climbing a tree",
        ],
    )
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
