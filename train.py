#!/usr/bin/env python3
"""
Fine-tune CLIP with LoRA for image-text retrieval.

Usage:
    python train.py                          # defaults
    python train.py --config configs/my.yaml
    python train.py --config cfg.yaml --training.lr 5e-5 --data.batch_size 128
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import CLIPTokenizerFast, get_cosine_schedule_with_warmup

from config import Config
from data.dataset import build_dataset
from data.transforms import build_train_transform, build_val_transform
from eval.build_index import build_retrieval_indices
from eval.metrics import compute_retrieval_metrics, format_metrics
from model.clip_lora import CLIPWithLoRA
from model.loss import CLIPContrastiveLoss


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Path to a YAML config file")
    # Allow arbitrary key=value overrides, e.g. --training.lr 1e-4
    p.add_argument("overrides", nargs="*",
                   help="Key=value overrides, e.g. training.lr=5e-5")
    return p.parse_args()


def apply_overrides(cfg: Config, overrides: list[str]) -> None:
    from config import _merge
    for item in overrides:
        # Strip leading "--" if present
        item = item.lstrip("-")
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        parts = key.split(".")
        # Convert string value to best-guess Python type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
        # Navigate nested keys
        dc = cfg
        for part in parts[:-1]:
            dc = getattr(dc, part)
        setattr(dc, parts[-1], value)


# ---------------------------------------------------------------------------
# Optimizer param groups
# ---------------------------------------------------------------------------

def build_optimizer(model: CLIPWithLoRA, cfg) -> AdamW:
    lora_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_" in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    return AdamW(
        [
            {"params": lora_params,  "weight_decay": cfg.training.weight_decay},
            {"params": other_params, "weight_decay": 0.0},
        ],
        lr=cfg.training.lr,
        betas=(0.9, 0.98),
        eps=1e-6,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(model, eval_dataset, cfg, accelerator) -> dict:
    device = accelerator.device
    unwrapped = accelerator.unwrap_model(model)
    img_embs, txt_embs, img_idx, txt_idx = build_retrieval_indices(
        unwrapped, eval_dataset, cfg.eval.batch_size, device
    )
    metrics = compute_retrieval_metrics(
        img_idx, txt_idx,
        img_embs, txt_embs,
        eval_dataset.gt_i2t,
        eval_dataset.gt_t2i,
        cfg.eval.top_k,
    )
    return metrics


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(model, cfg, epoch: int, metrics: dict, is_best: bool) -> None:
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, f"epoch_{epoch:03d}")
    accelerator_unwrapped = model  # already unwrapped at call site
    accelerator_unwrapped.save_adapter(ckpt_path)
    if is_best:
        best_path = os.path.join(cfg.training.checkpoint_dir, "best")
        accelerator_unwrapped.save_adapter(best_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.overrides:
        apply_overrides(cfg, args.overrides)

    # ------------------------------------------------------------------
    # Accelerator
    # ------------------------------------------------------------------
    log_with = "wandb" if cfg.training.wandb_project else None
    accelerator = Accelerator(
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.grad_accumulation_steps,
        log_with=log_with,
    )

    set_seed(cfg.training.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
        if log_with == "wandb":
            accelerator.init_trackers(
                project_name=cfg.training.wandb_project,
                config=vars(cfg),
                init_kwargs={"wandb": {"name": cfg.training.wandb_run_name}},
            )

    # ------------------------------------------------------------------
    # Tokenizer + transforms
    # ------------------------------------------------------------------
    tokenizer = CLIPTokenizerFast.from_pretrained(cfg.model.backbone)
    train_tf = build_train_transform(cfg.data.image_size)
    val_tf   = build_val_transform(cfg.data.image_size)

    # ------------------------------------------------------------------
    # Datasets + loaders
    # ------------------------------------------------------------------
    train_ds = build_dataset(cfg.data, "train", train_tf, val_tf, tokenizer)
    val_ds   = build_dataset(cfg.data, "val",   train_tf, val_tf, tokenizer)
    test_ds  = build_dataset(cfg.data, "test",  train_tf, val_tf, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # Val loader is only used if you want a quick sanity loop;
    # eval uses val_ds.image_dataset / val_ds.text_dataset directly.

    # ------------------------------------------------------------------
    # Model + loss + optimizer + scheduler
    # ------------------------------------------------------------------
    model = CLIPWithLoRA(cfg.model)

    if accelerator.is_main_process:
        model.print_trainable_summary()

    criterion = CLIPContrastiveLoss(hard_neg_weight=cfg.training.hard_neg_weight)
    optimizer = build_optimizer(model, cfg)

    num_update_steps = (
        len(train_loader) // cfg.training.grad_accumulation_steps * cfg.training.epochs
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=num_update_steps,
    )

    # Resume
    if cfg.training.resume_from:
        from peft import PeftModel
        accelerator.print(f"Resuming adapter from {cfg.training.resume_from}")
        base_clip = model._clip
        model._peft = PeftModel.from_pretrained(base_clip, cfg.training.resume_from)

    model, optimizer, train_loader = accelerator.prepare(
        model, optimizer, train_loader
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    global_step = 0
    best_mean_r1 = 0.0

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                img_feats, txt_feats, logit_scale = model(
                    batch["pixel_values"],
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                loss, log_dict = criterion(img_feats, txt_feats, logit_scale)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), cfg.training.clip_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1

                epoch_loss += log_dict["loss/total"]

                if global_step % cfg.training.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    log_dict["train/lr"] = lr
                    log_dict["train/step"] = global_step
                    log_dict["train/epoch"] = epoch

                    accelerator.log(log_dict, step=global_step)
                    accelerator.print(
                        f"epoch {epoch:3d}  step {global_step:6d}  "
                        f"loss {log_dict['loss/total']:.4f}  "
                        f"scale {log_dict['logit_scale']:.2f}  "
                        f"lr {lr:.2e}"
                    )

        # ----------------------------------------------------------------
        # Epoch-level evaluation
        # ----------------------------------------------------------------
        if epoch % cfg.training.eval_interval == 0:
            accelerator.print(f"\n--- Epoch {epoch} eval (val) ---")
            metrics = evaluate(model, val_ds, cfg, accelerator)
            accelerator.print(format_metrics(metrics))

            # Prefix metrics for logging
            logged = {f"val/{k}": v for k, v in metrics.items()}
            logged["val/epoch"] = epoch
            accelerator.log(logged, step=global_step)

            # Checkpoint
            mean_r1 = metrics["mean_R@1"]
            is_best = mean_r1 > best_mean_r1
            if is_best:
                best_mean_r1 = mean_r1

            if accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(model), cfg, epoch, metrics, is_best
                )
            if is_best:
                accelerator.print(f"  *** New best mean R@1: {mean_r1:.4f} ***")

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    accelerator.print("\n=== Final test evaluation ===")
    test_metrics = evaluate(model, test_ds, cfg, accelerator)
    accelerator.print(format_metrics(test_metrics))
    logged = {f"test/{k}": v for k, v in test_metrics.items()}
    accelerator.log(logged, step=global_step)

    if log_with:
        accelerator.end_training()


if __name__ == "__main__":
    main()
