from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Applies to all q_proj / v_proj in both vision and text encoders
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"


@dataclass
class ModelConfig:
    backbone: str = "openai/clip-vit-base-patch32"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    freeze_logit_scale: bool = False
    unfreeze_projection_heads: bool = True  # visual_projection + text_projection


@dataclass
class DataConfig:
    dataset: str = "flickr30k"   # "flickr30k" | "coco" | "custom"
    root: str = "data/flickr30k"
    ann_file: str = "data/flickr30k/dataset_flickr30k.json"  # Karpathy split JSON
    image_size: int = 224
    batch_size: int = 256
    num_workers: int = 8
    max_text_length: int = 77


@dataclass
class TrainingConfig:
    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_accumulation_steps: int = 1
    mixed_precision: str = "no"        # "bf16"/"fp16" on CUDA; "no" on MPS/CPU
    clip_grad_norm: float = 1.0
    hard_neg_weight: float = 0.0       # 0 = disabled; try 0.2 to enable
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    log_interval: int = 50             # steps
    eval_interval: int = 1             # epochs
    wandb_project: Optional[str] = "clip-lora"
    wandb_run_name: Optional[str] = None
    seed: int = 42


@dataclass
class EvalConfig:
    top_k: List[int] = field(default_factory=lambda: [1, 5, 10])
    batch_size: int = 512


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, path: str) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = cls()
        _merge(cfg, raw)
        return cfg


def _merge(dc, d: dict) -> None:
    """Recursively apply dict overrides onto a dataclass instance."""
    for key, value in d.items():
        if not hasattr(dc, key):
            raise ValueError(f"Unknown config key: {key!r}")
        attr = getattr(dc, key)
        if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
            _merge(attr, value)
        else:
            setattr(dc, key, value)
