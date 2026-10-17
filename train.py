import argparse
import os
import torch
import yaml

from models import create_modern_inception_v1, count_parameters, estimate_flops, SCALE_PRESETS
from data import build_dataloaders, MixupCutmix
from engine import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Modern InceptionV1 across varying scales up to 300x")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--scale", type=str, default="base", choices=list(SCALE_PRESETS.keys()), help="Model scale preset")
    parser.add_argument("--width-mult", type=float, default=None, help="Custom width multiplier (overrides preset)")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of classes (ImageNet=1000)")
    parser.add_argument("--no-residual", action="store_true", help="Disable residual connections")
    parser.add_argument("--residual-scale", type=float, default=1.0, help="Residual branch scaling factor")
    parser.add_argument("--drop-path", type=float, default=None, help="Stochastic depth rate")
    parser.add_argument("--dropout", type=float, default=None, help="Head dropout rate")

    # Data
    parser.add_argument("--data-dir", type=str, default=None, help="Path to ImageNet dataset (with train/ and val/)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset for benchmarking/dry-runs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--img-size", type=int, default=224, help="Input image resolution")
    parser.add_argument("--no-mixup", action="store_true", help="Disable Mixup/CutMix augmentation")

    # Training
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="AdamW decoupled weight decay")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name for experiment tracking")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load YAML config if provided
    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"============================================================")
    print(f" Training Modern InceptionV1 (Varying Scales up to 300x)")
    print(f" Device: {device} | OS: mac | PyTorch: {torch.__version__}")
    print(f"============================================================")

    # Instantiate model
    scale_arg = args.width_mult if args.width_mult is not None else args.scale
    use_res = not args.no_residual
    model_kwargs = {"use_residual": use_res, "residual_scale": args.residual_scale}
    if args.drop_path is not None:
        model_kwargs["drop_path_rate"] = args.drop_path
    if args.dropout is not None:
        model_kwargs["dropout_rate"] = args.dropout

    model = create_modern_inception_v1(
        scale_name_or_multiplier=scale_arg,
        num_classes=args.num_classes,
        **model_kwargs,
    )

    total_params = count_parameters(model)
    head_params = sum(p.numel() for p in model.head.parameters())
    backbone_params = total_params - head_params
    flops = estimate_flops(model, input_size=(3, args.img_size, args.img_size))

    print(f"[Model Architecture]")
    print(f" • Scale Setting:    {scale_arg}")
    print(f" • Width Multiplier: {model.width_multiplier:.3f}")
    print(f" • Total Parameters: {total_params:,d}")
    print(f" • Backbone Params:  {backbone_params:,d}")
    print(f" • Head Parameters:  {head_params:,d}")
    print(f" • FLOPs / MACs:     {flops/1e6:.1f} M")
    print(f" • Modern Upgrades:  BatchNorm + GELU + Factorized 5x5 + Residual Shortcuts ({use_res}) + DropPath")

    # Build DataLoaders
    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        synthetic=args.synthetic,
        img_size=args.img_size,
    )

    # Mixup / CutMix
    mixup_cutmix = None
    if not args.no_mixup:
        mixup_cutmix = MixupCutmix(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            prob=1.0,
            label_smoothing=0.1,
            num_classes=args.num_classes,
        )

    # Checkpoint output directory
    tag = f"scale_{scale_arg}_params_{total_params}"
    ckpt_dir = os.path.join(args.checkpoint_dir, tag)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        mixup_cutmix=mixup_cutmix,
        checkpoint_dir=ckpt_dir,
    )

    trainer.train()


if __name__ == "__main__":
    main()
