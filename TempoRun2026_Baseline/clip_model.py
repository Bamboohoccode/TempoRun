"""Self-contained CLIP wrapper (open_clip) — image AND text encoding, L2-normalized."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class ClipModel(nn.Module):
    def __init__(
        self,
        model_name="PE-Core-bigG-14-448",
        pretrained="meta",
        precision="fp16",
        device=None,
    ):
        super().__init__()

        self.device = torch.device(
            device or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model, self.process_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                precision=precision,
            )
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)


        self.logit_scale = nn.Parameter(
            torch.tensor(
                math.log(1.0 / 0.07),
                dtype=torch.float32,
            ))
        self.to(self.device)
    def encode_images(self, pil_images: list, batch_size=64) -> np.ndarray:
        torch = self.torch
        feats = []
        for i in range(0, len(pil_images), batch_size):
            batch = [self.preprocess(im) for im in pil_images[i:i + batch_size]]
            with torch.no_grad():
                x = torch.stack(batch).to(self.device)
                f = self.model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.float().cpu().numpy().astype(np.float16))
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float16)

    def encode_texts(self, texts: list[str], batch_size=256) -> np.ndarray:
        torch = self.torch
        feats = []
        for i in range(0, len(texts), batch_size):
            toks = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            with torch.no_grad():
                f = self.model.encode_text(toks)
                f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.float().cpu().numpy().astype(np.float32))
    
        return np.concatenate(feats, 0) if feats else np.zeros((0, self.dim), np.float32)
    def forward(self, batch):
        model_dtype = next(self.model.visual.parameters()).dtype
        images = batch["image"].to(
            device=self.device,
            dtype=model_dtype,
            non_blocking=True)
        text_tokens = self.tokenizer(
            batch["sentence"]).to(
        self.device,
        non_blocking=True)
        image_features = self.model.encode_image(images,normalize=True)
        text_features = self.model.encode_text(text_tokens,normalize=True)
        scale = self.model.logit_scale.exp().clamp(max=100.0)
        logits = (image_features @ text_features.T) * scale

        targets = torch.arange(
            logits.shape[0],
            device=logits.device)

        loss_i2t = F.cross_entropy(
            logits,
            targets)

        loss_t2i = F.cross_entropy(
            logits.T,
            targets)
        
        return (loss_i2t + loss_t2i) / 2
