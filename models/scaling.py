import math
import torch
from typing import Dict, Any, Tuple
from models.modern_inception import ModernInceptionV1


# Standard presets covering a ~300x parameter span (matching Nick Cammarata's ConvNeXt 92k to 28M scale range)
# Base (w=1.0) has ~9.2M params.
# Pico has ~92K params (1x baseline of the 300x range)
# Large has ~28M params (300x relative to pico)
SCALE_PRESETS = {
    "pico": {"width_mult": 0.088, "drop_path": 0.0, "dropout": 0.0, "desc": "1x Scale (~92k params)"},
    "nano": {"width_mult": 0.155, "drop_path": 0.02, "dropout": 0.05, "desc": "3x Scale (~280k params)"},
    "micro": {"width_mult": 0.29, "drop_path": 0.05, "dropout": 0.1, "desc": "10x Scale (~950k params)"},
    "mini": {"width_mult": 0.52, "drop_path": 0.08, "dropout": 0.15, "desc": "30x Scale (~2.9M params)"},
    "base": {"width_mult": 0.95, "drop_path": 0.1, "dropout": 0.2, "desc": "100x Scale (~9.5M params)"},
    "large": {"width_mult": 1.66, "drop_path": 0.15, "dropout": 0.3, "desc": "300x Scale (~28.5M params)"},
}


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: torch.nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> int:
    """
    Estimate MACs/FLOPs using a single forward pass with hooks.
    """
    total_macs = 0

    def conv_hook(module, inp, out):
        nonlocal total_macs
        _, cin, h_in, w_in = inp[0].shape
        _, cout, h_out, w_out = out.shape
        kh, kw = module.kernel_size
        groups = module.groups
        macs_per_point = (cin * kh * kw * cout) // groups
        total_macs += macs_per_point * h_out * w_out

    def linear_hook(module, inp, out):
        nonlocal total_macs
        cin = module.in_features
        cout = module.out_features
        total_macs += cin * cout

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size, device=device)
    training_state = model.training
    model.eval()
    with torch.no_grad():
        _ = model(dummy_input)
    model.train(training_state)

    for h in hooks:
        h.remove()

    return total_macs


def create_modern_inception_v1(
    scale_name_or_multiplier: Any = "base",
    num_classes: int = 1000,
    use_residual: bool = True,
    residual_scale: float = 1.0,
    **kwargs
) -> ModernInceptionV1:
    """
    Factory function to instantiate ModernInceptionV1 either by preset name
    ('pico', 'nano', 'micro', 'mini', 'base', 'large') or by a float width multiplier.
    """
    if isinstance(scale_name_or_multiplier, str):
        preset_key = scale_name_or_multiplier.lower()
        if preset_key in SCALE_PRESETS:
            cfg = SCALE_PRESETS[preset_key]
            width_mult = cfg["width_mult"]
            drop_path = kwargs.get("drop_path_rate", cfg["drop_path"])
            dropout = kwargs.get("dropout_rate", cfg["dropout"])
        else:
            raise ValueError(f"Unknown preset '{scale_name_or_multiplier}'. Available: {list(SCALE_PRESETS.keys())}")
    elif isinstance(scale_name_or_multiplier, (int, float)):
        width_mult = float(scale_name_or_multiplier)
        drop_path = kwargs.get("drop_path_rate", 0.1)
        dropout = kwargs.get("dropout_rate", 0.2)
    else:
        raise TypeError(f"scale_name_or_multiplier must be str or float, got {type(scale_name_or_multiplier)}")

    model = ModernInceptionV1(
        num_classes=num_classes,
        width_multiplier=width_mult,
        use_residual=use_residual,
        residual_scale=residual_scale,
        drop_path_rate=drop_path,
        dropout_rate=dropout,
    )
    return model
