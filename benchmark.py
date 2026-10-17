import time
import torch
import pandas as pd
from typing import Dict, List

from models import create_modern_inception_v1, count_parameters, estimate_flops, SCALE_PRESETS
from interpretability import FeatureExtractor, compute_activation_kurtosis, compute_lifetime_sparsity


def benchmark_scale(scale_name: str, device: torch.device, batch_size: int = 32, num_steps: int = 20) -> Dict:
    model = create_modern_inception_v1(scale_name, num_classes=1000)
    model.to(device)
    model.train()

    total_params = count_parameters(model)
    head_params = sum(p.numel() for p in model.head.parameters())
    backbone_params = total_params - head_params
    flops = estimate_flops(model, input_size=(3, 224, 224))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Warmup
    dummy_x = torch.randn(batch_size, 3, 224, 224, device=device)
    dummy_y = torch.randint(0, 1000, (batch_size,), device=device)

    for _ in range(3):
        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Time training steps
    start_time = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(dummy_x)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    total_time = time.time() - start_time
    throughput = (batch_size * num_steps) / total_time

    # Feature selectivity analysis on mixed4c
    extractor = FeatureExtractor(model, target_layers=["mixed4c"])
    with torch.no_grad():
        _ = model(dummy_x)
    mixed4c_act = extractor.activations.get("mixed4c", None)
    extractor.remove()

    avg_kurtosis = 0.0
    avg_sparsity = 0.0
    if mixed4c_act is not None:
        kurt = compute_activation_kurtosis(mixed4c_act)
        spars = compute_lifetime_sparsity(mixed4c_act)
        avg_kurtosis = kurt.mean().item()
        avg_sparsity = spars.mean().item()

    return {
        "Scale": scale_name.upper(),
        "Multiplier": SCALE_PRESETS[scale_name]["width_mult"],
        "Total Params": total_params,
        "Backbone Params": backbone_params,
        "FLOPs (M)": round(flops / 1e6, 1),
        "Throughput (img/s)": round(throughput, 1),
        "mixed4c Kurtosis": round(avg_kurtosis, 2),
        "mixed4c Sparsity": round(avg_sparsity, 3),
    }


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"==========================================================================")
    print(f" Modern InceptionV1: 300x Scale Range Benchmark")
    print(f" Device: {device} | OS: mac")
    print(f"==========================================================================")

    results = []
    for scale in SCALE_PRESETS:
        print(f"[*] Benchmarking {scale.upper()} scale...")
        res = benchmark_scale(scale, device=device, batch_size=32, num_steps=10)
        results.append(res)

    df = pd.DataFrame(results)
    pico_params = results[0]["Backbone Params"]
    df["Scale Ratio (Backbone)"] = df["Backbone Params"].apply(lambda p: f"{p / pico_params:.1f}x")

    print("\n" + "=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)

    df.to_csv("benchmark_results.csv", index=False)
    print("\n[✓] Results saved to benchmark_results.csv")


if __name__ == "__main__":
    main()
