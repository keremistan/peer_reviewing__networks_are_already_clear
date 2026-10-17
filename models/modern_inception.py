import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


def _make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    Ensure channel counts are divisible by divisor (default 8) for optimal hardware acceleration.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class DropPath(nn.Module):
    """
    Stochastic Depth per sample (drop paths) for residual blocks.
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvBNAct(nn.Module):
    """
    Standard building block: Convolution -> BatchNorm -> Activation (GELU).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        act: bool = True,
        use_bn: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ModernInceptionModule(nn.Module):
    """
    Modernized InceptionV1 module:
    - BatchNorm after every conv layer
    - GELU activations instead of ReLU
    - Factorized 5x5 branch into two 3x3 convolutions (as in Inception-v2/v3)
    - Residual skip-connection with projection (Inception-ResNet style)
    - Stochastic depth (DropPath)
    """
    def __init__(
        self,
        in_channels: int,
        n1x1: int,
        n3x3_reduce: int,
        n3x3: int,
        n5x5_reduce: int,
        n5x5: int,
        pool_proj: int,
        use_residual: bool = True,
        residual_scale: float = 1.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.residual_scale = residual_scale

        # Branch 1: 1x1 conv
        self.branch1 = ConvBNAct(in_channels, n1x1, kernel_size=1)

        # Branch 2: 1x1 conv -> 3x3 conv
        self.branch2 = nn.Sequential(
            ConvBNAct(in_channels, n3x3_reduce, kernel_size=1),
            ConvBNAct(n3x3_reduce, n3x3, kernel_size=3, padding=1),
        )

        # Branch 3: Factorized 5x5 -> 1x1 conv -> 3x3 conv -> 3x3 conv
        self.branch3 = nn.Sequential(
            ConvBNAct(in_channels, n5x5_reduce, kernel_size=1),
            ConvBNAct(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            ConvBNAct(n5x5, n5x5, kernel_size=3, padding=1),
        )

        # Branch 4: 3x3 maxpool -> 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNAct(in_channels, pool_proj, kernel_size=1),
        )

        out_channels = n1x1 + n3x3 + n5x5 + pool_proj
        self.out_channels = out_channels

        # Residual shortcut & projection
        if self.use_residual:
            if in_channels != out_channels:
                self.shortcut = ConvBNAct(in_channels, out_channels, kernel_size=1, act=False)
            else:
                self.shortcut = nn.Identity()
            self.post_act = nn.GELU()
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)

        if self.use_residual:
            res = self.shortcut(x)
            out = self.post_act(res + self.drop_path(out * self.residual_scale))
        return out


class ModernInceptionV1(nn.Module):
    """
    Modern InceptionV1 with:
    - Width multiplier supporting ~300x scaling range (from ~90k to ~28M+ parameters)
    - BatchNorm, GELU, Factorized 5x5, Residual connections, DropPath
    - Modern Stem (with BatchNorm + GELU)
    - Global Average Pooling and modern Classifier head
    """
    def __init__(
        self,
        num_classes: int = 1000,
        width_multiplier: float = 1.0,
        in_channels: int = 3,
        use_residual: bool = True,
        residual_scale: float = 1.0,
        drop_path_rate: float = 0.1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.width_multiplier = width_multiplier
        self.num_classes = num_classes

        def c(v: int) -> int:
            return _make_divisible(v * width_multiplier)

        # Modern Stem
        # Original: conv 7x7 (s2) -> maxpool 3x3 (s2) -> conv 1x1 (s1) -> conv 3x3 (s1) -> maxpool 3x3 (s2)
        c_stem1 = c(64)
        c_stem2 = c(64)
        c_stem3 = c(192)

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c_stem1, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(c_stem1, c_stem2, kernel_size=1),
            ConvBNAct(c_stem2, c_stem3, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Baseline InceptionV1 channel specifications:
        # (n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj)
        specs = [
            # Stage 3: mixed3a, mixed3b
            ("mixed3a", (64, 96, 128, 16, 32, 32)),
            ("mixed3b", (128, 128, 192, 32, 96, 64)),
            # Stage 4: mixed4a, mixed4b, mixed4c, mixed4d, mixed4e
            ("mixed4a", (192, 96, 208, 16, 48, 64)),
            ("mixed4b", (160, 112, 224, 24, 64, 64)),
            ("mixed4c", (128, 128, 256, 24, 64, 64)),
            ("mixed4d", (112, 144, 288, 32, 64, 64)),
            ("mixed4e", (256, 160, 320, 32, 128, 128)),
            # Stage 5: mixed5a, mixed5b
            ("mixed5a", (256, 160, 320, 32, 128, 128)),
            ("mixed5b", (384, 192, 384, 48, 128, 128)),
        ]

        # Calculate linearly increasing drop path rates across blocks
        num_blocks = len(specs)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]

        self.blocks = nn.ModuleDict()
        current_in = c_stem3

        # We construct the 3 stages:
        # Stage 3: mixed3a, mixed3b -> maxpool
        # Stage 4: mixed4a, mixed4b, mixed4c, mixed4d, mixed4e -> maxpool
        # Stage 5: mixed5a, mixed5b
        block_idx = 0
        for name, (n1, n3r, n3, n5r, n5, pp) in specs:
            block = ModernInceptionModule(
                in_channels=current_in,
                n1x1=c(n1),
                n3x3_reduce=c(n3r),
                n3x3=c(n3),
                n5x5_reduce=c(n5r),
                n5x5=c(n5),
                pool_proj=c(pp),
                use_residual=use_residual,
                residual_scale=residual_scale,
                drop_path=dpr[block_idx],
            )
            self.blocks[name] = block
            current_in = block.out_channels
            block_idx += 1

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Head: Global Average Pool -> Dropout -> Linear
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else nn.Identity()
        self.head = nn.Linear(current_in, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass that returns both final feature representation and intermediate
        activation maps from all Inception blocks for mechanistic interpretability.
        """
        activations = {}
        x = self.stem(x)
        activations["stem"] = x

        # Stage 3
        x = self.blocks["mixed3a"](x)
        activations["mixed3a"] = x
        x = self.blocks["mixed3b"](x)
        activations["mixed3b"] = x
        x = self.maxpool3(x)

        # Stage 4
        x = self.blocks["mixed4a"](x)
        activations["mixed4a"] = x
        x = self.blocks["mixed4b"](x)
        activations["mixed4b"] = x
        x = self.blocks["mixed4c"](x)
        activations["mixed4c"] = x
        x = self.blocks["mixed4d"](x)
        activations["mixed4d"] = x
        x = self.blocks["mixed4e"](x)
        activations["mixed4e"] = x
        x = self.maxpool4(x)

        # Stage 5
        x = self.blocks["mixed5a"](x)
        activations["mixed5a"] = x
        x = self.blocks["mixed5b"](x)
        activations["mixed5b"] = x

        feat = self.gap(x).flatten(1)
        activations["gap"] = feat
        return feat, activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _ = self.extract_features(x)
        out = self.head(self.dropout(feat))
        return out
