from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizerFast


# ---------------------------------------------------------------------------
# Helper sub-datasets used during evaluation
# ---------------------------------------------------------------------------

class _ImageSubset(Dataset):
    def __init__(self, paths: List[str], transform: Callable):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        return {"pixel_values": self.transform(img)}


class _TextSubset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


# ---------------------------------------------------------------------------
# Base retrieval dataset (training)
# ---------------------------------------------------------------------------

class RetrievalDataset(Dataset):
    """
    Flat (image_path, caption) pairs for training.

    For evaluation use .image_dataset / .text_dataset / .gt_i2t / .gt_t2i.
    """

    def __init__(
        self,
        image_paths: List[str],
        captions: List[str],
        image_indices: List[int],      # caption_idx -> unique image_idx
        train_transform: Callable,
        val_transform: Callable,
        tokenizer: CLIPTokenizerFast,
        max_text_length: int = 77,
        split: str = "train",
    ):
        self.image_paths = image_paths          # unique images for this split
        self.captions = captions                # all captions for this split
        self.image_indices = image_indices      # len == len(captions)
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.split = split

        # Pre-tokenize once to avoid multiprocessing issues with HF fast tokenizers
        enc = tokenizer(
            captions,
            max_length=max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = enc["input_ids"]           # (N_captions, max_len)
        self.attention_mask = enc["attention_mask"]  # (N_captions, max_len)

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[self.image_indices[idx]]
        img = Image.open(img_path).convert("RGB")
        transform = self.train_transform if self.split == "train" else self.val_transform
        return {
            "pixel_values": transform(img),
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

    # ------------------------------------------------------------------
    # Evaluation interface
    # ------------------------------------------------------------------

    @property
    def image_dataset(self) -> _ImageSubset:
        return _ImageSubset(self.image_paths, self.val_transform)

    @property
    def text_dataset(self) -> _TextSubset:
        return _TextSubset(self.input_ids, self.attention_mask)

    @property
    def gt_i2t(self) -> List[List[int]]:
        """image_idx -> list of matching caption indices."""
        mapping: Dict[int, List[int]] = defaultdict(list)
        for cap_idx, img_idx in enumerate(self.image_indices):
            mapping[img_idx].append(cap_idx)
        return [mapping[i] for i in range(len(self.image_paths))]

    @property
    def gt_t2i(self) -> List[List[int]]:
        """caption_idx -> list of matching image indices (always 1 for standard datasets)."""
        return [[img_idx] for img_idx in self.image_indices]


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------

def _load_karpathy_split(
    ann_file: str,
    root: str,
    split: str,
    image_subdir: Optional[str] = None,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Parse a Karpathy-format annotation JSON.

    Returns (image_paths, captions, image_indices).
    The JSON schema is shared between Flickr30k and COCO.
    """
    with open(ann_file) as f:
        data = json.load(f)

    image_paths: List[str] = []
    captions: List[str] = []
    image_indices: List[int] = []

    for entry in data["images"]:
        if entry["split"] != split:
            continue

        filename = entry["filename"]
        img_path = (
            os.path.join(root, image_subdir, filename)
            if image_subdir
            else os.path.join(root, filename)
        )
        img_idx = len(image_paths)
        image_paths.append(img_path)

        for sent in entry["sentences"]:
            captions.append(sent["raw"])
            image_indices.append(img_idx)

    return image_paths, captions, image_indices


def build_flickr30k(
    root: str,
    ann_file: str,
    split: str,
    train_transform: Callable,
    val_transform: Callable,
    tokenizer: CLIPTokenizerFast,
    max_text_length: int = 77,
) -> RetrievalDataset:
    """
    Flickr30k using the Karpathy split JSON (dataset_flickr30k.json).
    Images are expected flat inside `root`.
    Splits: "train" | "val" | "test"
    """
    image_paths, captions, image_indices = _load_karpathy_split(
        ann_file, root, split, image_subdir=None
    )
    return RetrievalDataset(
        image_paths, captions, image_indices,
        train_transform, val_transform, tokenizer, max_text_length, split,
    )


def build_coco(
    root: str,
    ann_file: str,
    split: str,
    train_transform: Callable,
    val_transform: Callable,
    tokenizer: CLIPTokenizerFast,
    max_text_length: int = 77,
) -> RetrievalDataset:
    """
    COCO Captions using the Karpathy split JSON.
    Images live in `root/train2014/` or `root/val2014/` depending on split.
    Splits: "train" | "restval" | "val" | "test"
    """
    # COCO Karpathy: images reference a subdir via "filepath" field or just filename.
    # We handle both the Karpathy JSON (which includes a "filepath" key) and the
    # flat-directory case by falling back to filename-only lookup.
    with open(ann_file) as f:
        data = json.load(f)

    image_paths: List[str] = []
    captions: List[str] = []
    image_indices: List[int] = []

    for entry in data["images"]:
        entry_split = entry["split"]
        if entry_split == "restval" and split == "train":
            pass  # include restval images in training
        elif entry_split != split:
            continue

        filepath = entry.get("filepath", "")
        filename = entry["filename"]
        img_path = (
            os.path.join(root, filepath, filename)
            if filepath
            else os.path.join(root, filename)
        )
        img_idx = len(image_paths)
        image_paths.append(img_path)

        for sent in entry["sentences"]:
            captions.append(sent["raw"])
            image_indices.append(img_idx)

    return RetrievalDataset(
        image_paths, captions, image_indices,
        train_transform, val_transform, tokenizer, max_text_length,
        split="train" if split in ("train", "restval") else split,
    )


def build_custom(
    csv_file: str,
    train_transform: Callable,
    val_transform: Callable,
    tokenizer: CLIPTokenizerFast,
    max_text_length: int = 77,
    split: str = "train",
    val_frac: float = 0.05,
) -> RetrievalDataset:
    """
    Custom dataset from a CSV with columns: image_path, caption[, split].
    If a 'split' column is absent, the last `val_frac` fraction becomes val/test.
    """
    import csv

    rows: List[Tuple[str, str, str]] = []
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        has_split = "split" in (reader.fieldnames or [])
        for row in reader:
            s = row.get("split", "") if has_split else ""
            rows.append((row["image_path"], row["caption"], s))

    if not has_split:
        n = len(rows)
        val_start = int(n * (1 - val_frac))
        rows = [
            (p, c, "train" if i < val_start else "val")
            for i, (p, c, _) in enumerate(rows)
        ]

    # Group unique images, keep ordering stable
    seen: Dict[str, int] = {}
    image_paths: List[str] = []
    captions: List[str] = []
    image_indices: List[int] = []

    for img_path, caption, row_split in rows:
        if row_split != split:
            continue
        if img_path not in seen:
            seen[img_path] = len(image_paths)
            image_paths.append(img_path)
        captions.append(caption)
        image_indices.append(seen[img_path])

    return RetrievalDataset(
        image_paths, captions, image_indices,
        train_transform, val_transform, tokenizer, max_text_length, split,
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dataset(
    cfg,                   # DataConfig
    split: str,
    train_transform: Callable,
    val_transform: Callable,
    tokenizer: CLIPTokenizerFast,
) -> RetrievalDataset:
    if cfg.dataset == "flickr30k":
        return build_flickr30k(
            cfg.root, cfg.ann_file, split,
            train_transform, val_transform, tokenizer, cfg.max_text_length,
        )
    elif cfg.dataset == "coco":
        return build_coco(
            cfg.root, cfg.ann_file, split,
            train_transform, val_transform, tokenizer, cfg.max_text_length,
        )
    elif cfg.dataset == "custom":
        return build_custom(
            cfg.ann_file, train_transform, val_transform,
            tokenizer, cfg.max_text_length, split,
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r}")
