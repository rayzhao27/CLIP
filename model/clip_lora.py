from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import CLIPModel

from config import ModelConfig


class CLIPWithLoRA(nn.Module):
    """
    CLIP ViT-B/32 with LoRA applied to attention q_proj/v_proj in both encoders.

    Trainable params:
      - LoRA A/B matrices (via PEFT)
      - logit_scale  (unless freeze_logit_scale=True)
      - visual_projection + text_projection  (unless unfreeze_projection_heads=False)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()

        base = CLIPModel.from_pretrained(cfg.backbone)

        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=cfg.lora.target_modules,
            bias=cfg.lora.bias,
        )
        # PEFT freezes all base weights and injects trainable LoRA matrices
        self._peft = get_peft_model(base, lora_config)

        # Unfreeze extras on top of LoRA
        extras = []
        if not cfg.freeze_logit_scale:
            extras.append("logit_scale")
        if cfg.unfreeze_projection_heads:
            extras += ["visual_projection", "text_projection"]
        for name, param in self._peft.named_parameters():
            if any(k in name for k in extras):
                param.requires_grad_(True)

    # ------------------------------------------------------------------
    # Internal accessor: the original CLIPModel with LoRA layers in-place
    # ------------------------------------------------------------------

    @property
    def _clip(self) -> CLIPModel:
        return self._peft.base_model.model

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @property
    def logit_scale(self) -> torch.Tensor:
        return self._clip.logit_scale.exp().clamp(max=100.0)

    def encode_image(
        self, pixel_values: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        clip = self._peft.base_model.model
        vision_out = clip.vision_model(pixel_values=pixel_values)
        feats = clip.visual_projection(vision_out.pooler_output)
        return F.normalize(feats, dim=-1) if normalize else feats

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        clip = self._peft.base_model.model
        text_out = clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feats = clip.text_projection(text_out.pooler_output)
        return F.normalize(feats, dim=-1) if normalize else feats

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_feats = self.encode_image(pixel_values)
        txt_feats = self.encode_text(input_ids, attention_mask)
        return img_feats, txt_feats, self.logit_scale

    # ------------------------------------------------------------------
    # Persistence — saves only the LoRA adapter weights
    # ------------------------------------------------------------------

    def save_adapter(self, path: str) -> None:
        self._peft.save_pretrained(path)

    @classmethod
    def load_adapter(cls, cfg: ModelConfig, adapter_path: str) -> CLIPWithLoRA:
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        base = CLIPModel.from_pretrained(cfg.backbone)
        instance._peft = PeftModel.from_pretrained(base, adapter_path)
        return instance

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def trainable_parameters(self):
        return [(n, p) for n, p in self.named_parameters() if p.requires_grad]

    def print_trainable_summary(self) -> None:
        self._peft.print_trainable_parameters()
