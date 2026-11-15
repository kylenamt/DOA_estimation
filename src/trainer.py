import copy

import torch
import torch.nn.functional as F


def compute_loss(outputs, k, doa, doa_weight=1.0):
    """Returns (total loss, sum of squared DOA errors, number of DOAs)."""
    probs, doa_heads = outputs[0], outputs[1:]
    cls_loss = F.nll_loss(torch.log(probs + 1e-8), k)  # model already applies softmax
    sq_err, n = torch.zeros((), device=k.device), 0
    for K in range(1, 5):                              # DOA loss with the head matching true K
        mask = k == K - 1
        if mask.any():
            sq_err = sq_err + ((doa_heads[K - 1][mask] - doa[mask, :K]) ** 2).sum()
            n += int(mask.sum()) * K
    return cls_loss + doa_weight * sq_err / n, sq_err, n


@torch.no_grad()
def evaluate(model, loader, doa_scale, doa_weight, device):
    """Returns (loss, source-count accuracy, DOA RMSE in degrees)."""
    model.eval()
    loss, correct, sq_err, n_doa = 0.0, 0, 0.0, 0
    for xb, kb, db in loader:
        xb, kb, db = xb.to(device), kb.to(device), db.to(device)
        out = model(xb)
        l, se, n = compute_loss(out, kb, db, doa_weight)
        loss += l.item() * len(xb)
        correct += (out[0].argmax(1) == kb).sum().item()
        sq_err += se.item()
        n_doa += n
    N = len(loader.dataset)
    return loss / N, correct / N, doa_scale * (sq_err / n_doa) ** 0.5


def train(model, train_loader, val_loader, epochs=20, lr=1e-3,
          doa_scale=25.0, doa_weight=1.0, save_path=None):
    """Train the model, keep the weights with the lowest validation loss.

    doa_scale must match the one used in make_loaders (predictions * doa_scale = degrees).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_rmse": []}
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, kb, db in train_loader:
            xb, kb, db = xb.to(device), kb.to(device), db.to(device)
            loss = compute_loss(model(xb), kb, db, doa_weight)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        val_loss, val_acc, val_rmse = evaluate(model, val_loader, doa_scale, doa_weight, device)
        for key, v in zip(history, [total / len(train_loader), val_loss, val_acc, val_rmse]):
            history[key].append(v)
        print(f"Epoch {epoch + 1}: train {history['train_loss'][-1]:.4f} | val {val_loss:.4f} "
              f"| acc {val_acc:.3f} | RMSE {val_rmse:.2f} deg")

        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            if save_path:
                torch.save(best_state, save_path)

    model.load_state_dict(best_state)
    return history
