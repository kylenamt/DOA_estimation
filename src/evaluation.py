import numpy as np
import torch

from src.dataloader import make_loaders


@torch.no_grad()
def evaluate_per_k(model, loader, doa_scale=25.0, device=None):
    """Run the model on a test loader and report results separately for each number of sources K.

    Returns {K: {"n": #samples, "acc": source-count accuracy,
                 "rmse": DOA RMSE in degrees (using the head of the true K),
                 "pred_counts": how often each count 1..4 was predicted}}
    """
    device = device or next(model.parameters()).device
    model.to(device).eval()

    probs, heads, ks, doas = [], [[] for _ in range(4)], [], []
    for xb, kb, db in loader:
        out = model(xb.to(device))
        probs.append(out[0].cpu())
        for i in range(4):
            heads[i].append(out[i + 1].cpu())
        ks.append(kb)
        doas.append(db)
    probs, ks, doas = torch.cat(probs), torch.cat(ks), torch.cat(doas)
    heads = [torch.cat(h) for h in heads]
    pred_k = probs.argmax(1)

    results = {}
    for K in range(1, 5):
        mask = ks == K - 1
        if not mask.any():
            continue
        err = (heads[K - 1][mask] - doas[mask, :K]) * doa_scale  # degrees
        results[K] = {
            "n": int(mask.sum()),
            "acc": (pred_k[mask] == K - 1).float().mean().item(),
            "rmse": err.pow(2).mean().sqrt().item(),
            "pred_counts": np.bincount(pred_k[mask].numpy(), minlength=4),
        }
    return results


def evaluate_per_snr(model, test_sets, doa_scale=25.0, batch_size=256):
    """test_sets: {snr: (snapshots, doas)}, e.g. one generate_snapshot_dataset call per SNR.

    Returns {snr: evaluate_per_k results}.
    """
    return {snr: evaluate_per_k(model, make_loaders(s, d, batch_size, val_fraction=0, doa_scale=doa_scale),
                                doa_scale)
            for snr, (s, d) in test_sets.items()}


def print_results(results):
    print(f"{'K':>2} | {'n':>6} | {'acc':>6} | {'RMSE (deg)':>10} | predicted K = 1/2/3/4")
    for K, r in results.items():
        print(f"{K:>2} | {r['n']:>6} | {r['acc']:>6.3f} | {r['rmse']:>10.2f} | {r['pred_counts']}")
