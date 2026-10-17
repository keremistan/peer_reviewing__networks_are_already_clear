import numpy as np
import torch
from typing import Tuple


def one_hot(x: torch.Tensor, num_classes: int, on_value: float = 1.0, off_value: float = 0.0) -> torch.Tensor:
    x = x.long().view(-1, 1)
    return torch.full((x.size(0), num_classes), off_value, device=x.device).scatter_(1, x, on_value)


class MixupCutmix:
    """
    Applies Mixup and/or CutMix data augmentation with label smoothing.
    """
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        mode: str = "batch",
        label_smoothing: float = 0.1,
        num_classes: int = 1000,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.switch_prob = switch_prob
        self.mode = mode
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes

    def _rand_bbox(self, img_shape: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        H = img_shape[-2]
        W = img_shape[-1]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniformly sample center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if np.random.rand() > self.prob:
            # Only label smoothing
            target = one_hot(
                target,
                num_classes=self.num_classes,
                on_value=1.0 - self.label_smoothing + (self.label_smoothing / self.num_classes),
                off_value=self.label_smoothing / self.num_classes,
            )
            return x, target

        batch_size = x.size(0)
        use_cutmix = np.random.rand() < self.switch_prob

        if use_cutmix and self.cutmix_alpha > 0.0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        elif not use_cutmix and self.mixup_alpha > 0.0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1.0

        # Permute batch
        perm_idx = torch.randperm(batch_size, device=x.device)

        if use_cutmix:
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.shape, lam)
            x_aug = x.clone()
            x_aug[:, :, bby1:bby2, bbx1:bbx2] = x[perm_idx, :, bby1:bby2, bbx1:bbx2]
            # Adjust lambda according to exact pixel ratio
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))
        else:
            x_aug = lam * x + (1.0 - lam) * x[perm_idx]

        target_a = one_hot(
            target,
            num_classes=self.num_classes,
            on_value=1.0 - self.label_smoothing + (self.label_smoothing / self.num_classes),
            off_value=self.label_smoothing / self.num_classes,
        )
        target_b = one_hot(
            target[perm_idx],
            num_classes=self.num_classes,
            on_value=1.0 - self.label_smoothing + (self.label_smoothing / self.num_classes),
            off_value=self.label_smoothing / self.num_classes,
        )

        target_mixed = lam * target_a + (1.0 - lam) * target_b
        return x_aug, target_mixed


class SoftTargetCrossEntropy(torch.nn.Module):
    """
    Cross entropy loss for soft one-hot targets (e.g. from Mixup/CutMix/Label Smoothing).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * torch.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
