import os
import time
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from tqdm import tqdm

from engine.evaluator import evaluate
from data.mixup import MixupCutmix, SoftTargetCrossEntropy


def get_parameter_groups(model: nn.Module, weight_decay: float = 0.05):
    """
    Separate 2D parameters (weights of Conv/Linear) which receive weight decay
    from 1D parameters (biases, normalization weights/biases) which do not.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


class CosineWarmupScheduler:
    """
    Linear warmup followed by cosine annealing learning rate scheduler.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        epochs: int = 100,
        warmup_epochs: int = 5,
        lr: float = 1e-3,
        min_lr: float = 1e-6,
        weight_decay: float = 0.05,
        clip_grad: float = 1.0,
        mixup_cutmix: Optional[MixupCutmix] = None,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
        wandb_run = None,
        resume_path: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.clip_grad = clip_grad
        self.mixup_cutmix = mixup_cutmix
        self.checkpoint_dir = checkpoint_dir
        self.wandb_run = wandb_run
        self.start_epoch = 0
        self.best_top1 = 0.0
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Loss function
        if mixup_cutmix is not None:
            self.criterion = SoftTargetCrossEntropy()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer: Modern AdamW with decoupled weight decay
        param_groups = get_parameter_groups(self.model, weight_decay=weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            base_lr=lr,
            min_lr=min_lr,
        )

        # Automatic Mixed Precision
        self.use_amp = use_amp and (device.type in ["cuda", "mps"])
        self.amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda") if (self.use_amp and device.type == "cuda") else None

        # Resume if path provided
        if resume_path:
            self._resume_from_checkpoint(resume_path)

    def _resume_from_checkpoint(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found at: {path}")
        print(f"[*] Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)

        if "epoch" in checkpoint:
            self.start_epoch = int(checkpoint["epoch"])
        if "best_top1" in checkpoint:
            self.best_top1 = float(checkpoint["best_top1"])
        elif "val_top1" in checkpoint:
            self.best_top1 = float(checkpoint["val_top1"])

        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                print(f"[!] Warning: Could not restore optimizer state: {e}")

        if self.scaler is not None and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
            except Exception as e:
                print(f"[!] Warning: Could not restore scaler state: {e}")

        print(
            f"[✓] Checkpoint loaded successfully! Resuming from epoch {self.start_epoch + 1}/{self.epochs} "
            f"(Previous Val Top-1: {self.best_top1:.2f}%)"
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        current_lr = self.scheduler.step(epoch)

        total_loss = 0.0
        total_samples = 0
        start_time = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [LR: {current_lr:.6f}]")
        for images, targets in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if self.mixup_cutmix is not None:
                images, targets = self.mixup_cutmix(images, targets)

            self.optimizer.zero_grad()

            if self.use_amp and self.device.type == "cuda":
                with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                if self.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(total_samples, 1)
        throughput = total_samples / max(elapsed, 1e-5)

        return {
            "train_loss": avg_loss,
            "lr": current_lr,
            "train_throughput_img_per_sec": throughput,
            "epoch_time_sec": elapsed,
        }

    def train(self) -> Dict[str, Any]:
        best_top1 = self.best_top1
        history = []

        print(f"[*] Starting training from epoch {self.start_epoch + 1} to {self.epochs} on device: {self.device}")
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_loss, val_top1, val_top5 = evaluate(self.model, self.val_loader, self.device)

            epoch_log = {
                "epoch": epoch + 1,
                **train_metrics,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "val_top5": val_top5,
            }
            history.append(epoch_log)

            print(
                f"--> Epoch {epoch+1:3d} Summary | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Top-1: {val_top1:5.2f}% | "
                f"Val Top-5: {val_top5:5.2f}% | "
                f"Speed: {train_metrics['train_throughput_img_per_sec']:.1f} img/s"
            )

            # Checkpoint saving
            is_best = val_top1 > best_top1
            if is_best:
                best_top1 = val_top1

            ckpt_payload = {
                "epoch": epoch + 1,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
                "val_top1": val_top1,
                "best_top1": best_top1,
            }

            if is_best:
                torch.save(ckpt_payload, os.path.join(self.checkpoint_dir, "best_model.pt"))

            torch.save(ckpt_payload, os.path.join(self.checkpoint_dir, "latest_model.pt"))

        print(f"[✓] Training completed! Best Top-1 Accuracy: {best_top1:.2f}%")
        return {"best_top1": best_top1, "history": history}
