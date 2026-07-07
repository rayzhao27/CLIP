from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (standard CLIP loss).

    Args:
        img_feats:   (N, D) L2-normalised image embeddings
        txt_feats:   (N, D) L2-normalised text embeddings
        logit_scale: scalar temperature (already exp'd and clamped)

    Returns:
        Scalar loss.
    """
    N = img_feats.shape[0]
    logits = logit_scale * img_feats @ txt_feats.T   # (N, N)
    labels = torch.arange(N, device=img_feats.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2.0


def hard_negative_loss(
    img_feats: torch.Tensor,
    txt_feats: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Pairwise contrastive loss between each positive pair and its hardest
    in-batch negative.

    For each anchor i the loss is:
        -log σ(s_pos_i - s_hardneg_i)
    where σ is the sigmoid, computed symmetrically for i2t and t2i.

    Adding this on top of InfoNCE with a small weight (e.g. 0.2) explicitly
    penalises the most-confused in-batch pairs.
    """
    N = img_feats.shape[0]
    logits = logit_scale * img_feats @ txt_feats.T   # (N, N)
    eye = torch.eye(N, device=logits.device, dtype=torch.bool)

    # i2t: for each image, find the hardest (highest-scoring) negative text
    neg_i2t = logits.masked_fill(eye, float("-inf"))
    hard_neg_i2t = neg_i2t.max(dim=1).values          # (N,)
    pos_i2t = logits.diagonal()                        # (N,)
    loss_i2t = -F.logsigmoid(pos_i2t - hard_neg_i2t).mean()

    # t2i: for each text, find the hardest negative image
    neg_t2i = logits.T.masked_fill(eye, float("-inf"))
    hard_neg_t2i = neg_t2i.max(dim=1).values          # (N,)
    pos_t2i = logits.T.diagonal()                     # (N,)
    loss_t2i = -F.logsigmoid(pos_t2i - hard_neg_t2i).mean()

    return (loss_i2t + loss_t2i) / 2.0


class CLIPContrastiveLoss:
    """
    Combined loss: InfoNCE + optional hard-negative pairwise term.

    Args:
        hard_neg_weight: Weight λ for the hard-negative term (0 = disabled).
    """

    def __init__(self, hard_neg_weight: float = 0.0):
        self.hard_neg_weight = hard_neg_weight

    def __call__(
        self,
        img_feats: torch.Tensor,
        txt_feats: torch.Tensor,
        logit_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        loss = infonce_loss(img_feats, txt_feats, logit_scale)
        log = {"loss/infonce": loss.item()}

        if self.hard_neg_weight > 0.0:
            hn = hard_negative_loss(img_feats, txt_feats, logit_scale)
            loss = loss + self.hard_neg_weight * hn
            log["loss/hard_neg"] = hn.item()

        log["loss/total"] = loss.item()
        log["logit_scale"] = logit_scale.item()
        return loss, log
