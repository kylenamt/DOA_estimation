import json
import math
import os

import torch
import torch.nn.functional as F

from src.data_generation import SCENARIOS, generate_snapshot_dataset


def closest_sq_err(est, true):
    return ((est[:, None, :] - true[:, :, None]) ** 2).min(dim=2).values


def compute_loss(outputs, k, doa):
    # Classification loss
    probs, doa_heads = outputs[0], outputs[1:]
    loss = F.nll_loss(torch.log(probs + 1e-8), k)

    # DOA loss, head of the true K
    sq_err, n = 0.0, 0
    for K in range(1, 5):
        mask = k == K - 1
        se = closest_sq_err(doa_heads[K - 1][mask], doa[mask, :K])
        loss = loss + torch.sqrt(se.mean() + 1e-12)
        sq_err, n = sq_err + se.sum(), n + se.numel()
    return loss, sq_err, n


@torch.no_grad()
def evaluate(model, val, batch_size):
    model.eval()
    x, k, doa = val

    # Accumulate over batches
    loss, correct, sq_err, n_doa = 0.0, 0, 0.0, 0
    for i in range(0, len(k), batch_size):
        out = model(x[i:i + batch_size])
        l, se, n = compute_loss(out, k[i:i + batch_size], doa[i:i + batch_size])
        loss += l.item() * len(out[0]) / len(k)
        correct += (out[0].argmax(1) == k[i:i + batch_size]).sum().item()
        sq_err, n_doa = sq_err + se.item(), n_doa + n
    return loss, correct / len(k), math.sqrt(sq_err / n_doa)


def make_sampler(scenario, snr_range):
    half_fov, amp_range = SCENARIOS[scenario]["half_fov"], SCENARIOS[scenario]["amp_range"]

    # Fresh shuffled batch: K = 1..4, SNR evenly over snr_range
    def sample(n, seed):
        x, k, doa = generate_snapshot_dataset(1, 4, 16, torch.linspace(*snr_range, n // 4), -half_fov, half_fov,
                                              1, amp_range, seed=seed)
        perm = torch.randperm(n, device="cuda")
        return x[perm], k[perm], doa[perm] / half_fov

    return sample


def train(model, sample, val, steps, lr, doa_scale, out_dir, batch_size=4000, eval_every=2000):
    os.makedirs(out_dir, exist_ok=True)
    model.cuda()

    # Adam, 2 % warm-up, cosine decay
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    warmup = steps // 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / steps)))

    # Resume from checkpoint
    history = {"step": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_rmse": []}
    start, best_loss = 1, float("inf")
    if os.path.exists(f"{out_dir}/checkpoint.pth"):
        ckpt = torch.load(f"{out_dir}/checkpoint.pth")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start, best_loss, history = ckpt["step"] + 1, ckpt["best_loss"], ckpt["history"]

    with open(f"{out_dir}/train.log", "a" if start > 1 else "w") as log_file:
        def log(line):
            print(line)
            log_file.write(line + "\n")
            log_file.flush()

        log(f"steps {steps} | lr {lr} | batch {batch_size} | parameters {sum(p.numel() for p in model.parameters()):,} "
            f"| start at step {start}")
        running = 0.0
        for step in range(start, steps + 1):
            # Update
            model.train()
            xb, kb, db = sample(batch_size, step)
            loss = compute_loss(model(xb), kb, db)[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running += loss.detach()

            # Validate, log, save best and checkpoint
            if step % eval_every == 0:
                val_loss, val_acc, val_rmse = evaluate(model, val, batch_size)
                history["step"].append(step)
                history["train_loss"].append(running.item() / eval_every)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["val_rmse"].append(doa_scale * val_rmse)
                running = 0.0
                log(f"step {step:>6} | train {history['train_loss'][-1]:.4f} | val {val_loss:.4f} "
                    f"| acc {val_acc:.3f} | RMSE {history['val_rmse'][-1]:.3f} deg")
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), f"{out_dir}/best.pth")
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "step": step, "best_loss": best_loss, "history": history}, f"{out_dir}/checkpoint.pth")

    # Save last and history, restore best
    torch.save(model.state_dict(), f"{out_dir}/last.pth")
    with open(f"{out_dir}/history.json", "w") as f:
        json.dump(history, f)
    model.load_state_dict(torch.load(f"{out_dir}/best.pth"))
    return history
