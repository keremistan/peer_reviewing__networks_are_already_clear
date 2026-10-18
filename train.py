import argparse
import os
import torch
import yaml

try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(65536, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except Exception:
    pass

from models import create_modern_inception_v1, count_parameters, estimate_flops, SCALE_PRESETS
from data import build_dataloaders, MixupCutmix
from engine import Trainer


def parse_args():
    # 1. Parse --config first so defaults can be extracted from the YAML file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="configs/pico_scale1x.yaml", help="Path to YAML configuration file")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    cfg = {}
    config_path = pre_args.config
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})

    # 2. Main parser with defaults pulled from YAML
    parser = argparse.ArgumentParser(description="Train Modern InceptionV1 across varying scales up to 300x")
    parser.add_argument("--config", type=str, default=config_path, help="Path to YAML configuration file")

    # Model architecture
    parser.add_argument("--scale", type=str, default=model_cfg.get("scale", "pico"), choices=list(SCALE_PRESETS.keys()), help="Model scale preset")
    parser.add_argument("--width-mult", type=float, default=model_cfg.get("width_multiplier", None), help="Custom width multiplier (overrides preset)")
    parser.add_argument("--num-classes", type=int, default=model_cfg.get("num_classes", 100), help="Number of classes")
    parser.add_argument("--residual", action=argparse.BooleanOptionalAction, default=model_cfg.get("use_residual", True), help="Use residual connections")
    parser.add_argument("--residual-scale", type=float, default=model_cfg.get("residual_scale", 1.0), help="Residual branch scaling factor")
    parser.add_argument("--drop-path", type=float, default=model_cfg.get("drop_path_rate", 0.0), help="Stochastic depth rate")
    parser.add_argument("--dropout", type=float, default=model_cfg.get("dropout_rate", 0.0), help="Head dropout rate")

    # Data
    parser.add_argument("--data-dir", type=str, default=data_cfg.get("data_dir", None), help="Path to ImageNet dataset (with train/ and val/)")
    parser.add_argument("--synthetic", action=argparse.BooleanOptionalAction, default=data_cfg.get("synthetic", False), help="Use synthetic dataset for benchmarking/dry-runs")
    parser.add_argument("--batch-size", type=int, default=train_cfg.get("batch_size", 128), help="Training batch size")
    parser.add_argument("--num-workers", type=int, default=data_cfg.get("num_workers", 4), help="Data loader workers")
    parser.add_argument("--img-size", type=int, default=data_cfg.get("img_size", 224), help="Input image resolution")
    parser.add_argument("--mixup", action=argparse.BooleanOptionalAction, default=not data_cfg.get("no_mixup", False), help="Use Mixup/CutMix augmentation")
    parser.add_argument("--mixup-alpha", type=float, default=data_cfg.get("mixup_alpha", 0.8), help="Mixup alpha")
    parser.add_argument("--cutmix-alpha", type=float, default=data_cfg.get("cutmix_alpha", 1.0), help="Cutmix alpha")
    parser.add_argument("--mixup-prob", type=float, default=data_cfg.get("mixup_prob", 1.0), help="Mixup/Cutmix probability")
    parser.add_argument("--label-smoothing", type=float, default=data_cfg.get("label_smoothing", 0.1), help="Label smoothing rate")

    # Training
    parser.add_argument("--epochs", type=int, default=train_cfg.get("epochs", 100), help="Total training epochs")
    parser.add_argument("--warmup-epochs", type=int, default=train_cfg.get("warmup_epochs", 5), help="Linear warmup epochs")
    parser.add_argument("--lr", type=float, default=train_cfg.get("lr", 1e-3), help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=train_cfg.get("min_lr", 1e-6), help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=train_cfg.get("weight_decay", 0.05), help="AdamW decoupled weight decay")
    parser.add_argument("--clip-grad", type=float, default=train_cfg.get("clip_grad", 1.0), help="Max gradient norm")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=train_cfg.get("use_amp", True), help="Use automatic mixed precision")
    parser.add_argument("--device", type=str, default=train_cfg.get("device", "auto"), choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--checkpoint-dir", type=str, default=train_cfg.get("checkpoint_dir", "checkpoints"), help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=train_cfg.get("resume", None), help="Path to checkpoint file to resume training from")
    parser.add_argument("--wandb-project", type=str, default=train_cfg.get("wandb_project", None), help="WandB project name for experiment tracking")

    return parser.parse_args(remaining_argv)


def main():
    args = parse_args()

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
    print(f" Config file: {args.config or 'None'}")
    print(f" Device: {device} | OS: mac | PyTorch: {torch.__version__}")
    if args.resume:
        print(f" Resume checkpoint: {args.resume}")
    print(f"============================================================")

    # Instantiate model
    scale_arg = args.width_mult if args.width_mult is not None else args.scale
    use_res = args.residual
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
    print(f" • Modern Upgrades:  BatchNorm + GELU + Factorized 5x5 + Residual Shortcuts ({use_res}) + DropPath ({args.drop_path})")

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
    if args.mixup:
        mixup_cutmix = MixupCutmix(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            label_smoothing=args.label_smoothing,
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
        use_amp=args.use_amp,
        resume_path=args.resume,
    )

    trainer.train()


if __name__ == "__main__":
    main()
