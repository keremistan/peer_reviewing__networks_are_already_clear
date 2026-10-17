import torch
from typing import Tuple


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Tuple[float, ...]:
    """
    Computes top-k accuracy over the predictions.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return tuple(res)


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader, device: torch.device) -> Tuple[float, float, float]:
    """
    Evaluate model on validation loader.
    Returns: (val_loss, top1_acc, top5_acc)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, targets in val_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        top1, top5 = accuracy(outputs, targets, topk=(1, 5))
        bs = images.size(0)

        total_loss += loss.item() * bs
        total_top1 += top1 * bs
        total_top5 += top5 * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_top1 = total_top1 / max(total_samples, 1)
    avg_top5 = total_top5 / max(total_samples, 1)

    return avg_loss, avg_top1, avg_top5
