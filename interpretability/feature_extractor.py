import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class FeatureExtractor:
    """
    Hooks into ModernInceptionV1 layers to capture layer activations and
    individual neuron firing distributions for interpretability research.
    """
    def __init__(self, model: torch.nn.Module, target_layers: Optional[List[str]] = None):
        self.model = model
        self.target_layers = target_layers or ["mixed3a", "mixed3b", "mixed4a", "mixed4c", "mixed4e", "mixed5b"]
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(layer_name):
            def hook(module, inp, out):
                # Detach and move to CPU to avoid GPU/MPS memory accumulation
                self.activations[layer_name] = out.detach().cpu()
            return hook

        for name, module in self.model.named_modules():
            clean_name = name.split(".")[-1]
            if clean_name in self.target_layers:
                h = module.register_forward_hook(get_hook(clean_name))
                self.hooks.append(h)

    def clear(self):
        self.activations.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def compute_activation_kurtosis(activations: torch.Tensor) -> torch.Tensor:
    """
    Computes excess kurtosis for each neuron/channel.
    High kurtosis indicates high sparsity/selectivity (monosemantic-like firing),
    whereas low kurtosis indicates diffuse/dense activation (polysemantic superposition).
    """
    # activations: [B, C, H, W] -> flatten spatial dims to [B * H * W, C]
    if activations.ndim == 4:
        B, C, H, W = activations.shape
        acts = activations.permute(0, 2, 3, 1).reshape(-1, C)
    else:
        acts = activations

    mean = acts.mean(dim=0, keepdim=True)
    std = acts.std(dim=0, keepdim=True) + 1e-8
    norm_acts = (acts - mean) / std
    kurt = torch.mean(norm_acts ** 4, dim=0) - 3.0
    return kurt


def compute_lifetime_sparsity(activations: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
    """
    Measures the fraction of inputs for which each neuron fires above threshold.
    """
    if activations.ndim == 4:
        # Maximum activation across spatial map per sample
        acts, _ = activations.max(dim=-1)[0].max(dim=-1)  # [B, C]
    else:
        acts = activations

    max_act = acts.max(dim=0, keepdim=True)[0] + 1e-8
    rel_act = acts / max_act
    sparsity = (rel_act > threshold).float().mean(dim=0)
    return sparsity
