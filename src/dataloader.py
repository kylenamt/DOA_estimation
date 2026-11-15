import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def to_tensors(snapshots, doas):
    """Snapshot dicts -> (x: real/imag features, k: #sources - 1, doa: sorted DOAs padded to 4)."""
    xs, ks, ds = [], [], []
    for key in snapshots:
        for snap, doa in zip(snapshots[key], doas[key]):
            snap = snap.reshape(-1)
            xs.append(np.concatenate([snap.real, snap.imag]))
            ks.append(len(doa) - 1)
            ds.append(np.pad(np.sort(doa), (0, 4 - len(doa))))
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(ks),
            torch.tensor(np.array(ds), dtype=torch.float32))


def make_loaders(snapshots, doas, batch_size=256, val_fraction=0.1, doa_scale=25.0):
    """Returns (train_loader, val_loader). DOAs are divided by doa_scale.

    With val_fraction=0, returns a single non-shuffled loader (e.g. for a test set).
    """
    x, k, doa = to_tensors(snapshots, doas)
    data = TensorDataset(x, k, doa / doa_scale)
    if val_fraction == 0:
        return DataLoader(data, batch_size)
    n_val = int(len(data) * val_fraction)
    train_set, val_set = random_split(data, [len(data) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)  # BatchNorm needs batch > 1
    return train_loader, DataLoader(val_set, batch_size)
