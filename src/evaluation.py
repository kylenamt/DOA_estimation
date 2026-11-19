import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.baselines import alternating_projection, bartlett_spectrum, ml_two_sources, omp, ss_model_order, ss_music
from src.data_generation import SCENARIOS, generate_snapshot_dataset, impairment_matrix, steering_vector
from src.trainer import closest_sq_err

TEST_SNRS = list(range(0, 45, 5))
PAPER_SNRS = np.arange(0, 40.1, 2.5)
TEST_SEED = 1_000_000  # test seeds above all training seeds (= step)


def rmse(est, true):
    return closest_sq_err(est, true).mean().sqrt().item()


# Objective evaluation: same test set for every model
@torch.no_grad()
def evaluate_per_k(model, x, k, doa, doa_scale):
    model.cuda().eval()
    out = model(x)
    pred_k = out[0].argmax(1)

    # Accuracy, RMSE (Eq. 3) and predicted counts per true K
    results = {}
    for K in range(1, 5):
        mask = k == K - 1
        results[K] = {
            "n": int(mask.sum()),
            "acc": (pred_k[mask] == K - 1).float().mean().item(),
            "rmse": rmse(out[K][mask] * doa_scale, doa[mask, :K]),
            "pred_counts": np.bincount(pred_k[mask].cpu().numpy(), minlength=4),
        }
    return results


def evaluate_per_snr(model, scenario):
    half_fov, amp_range = SCENARIOS[scenario]["half_fov"], SCENARIOS[scenario]["amp_range"]
    results = {}
    for snr in TEST_SNRS:
        x, k, doa = generate_snapshot_dataset(1, 4, 16, [snr], -half_fov, half_fov, 1000, amp_range, seed=TEST_SEED + 1)
        results[snr] = evaluate_per_k(model, x, k, doa, half_fov)
    return results


def print_results(results):
    print(f"{'K':>2} | {'n':>6} | {'acc':>6} | {'RMSE (deg)':>10} | predicted K = 1/2/3/4")
    for K, r in results.items():
        print(f"{K:>2} | {r['n']:>6} | {r['acc']:>6.3f} | {r['rmse']:>10.2f} | {r['pred_counts']}")


def plot_learning_curve(history, title=""):
    step = np.array(history["step"]) / 1000
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(step, history["train_loss"], label="Train")
    axes[0].plot(step, history["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Step (thousands)", ylabel="Loss")
    axes[0].legend(frameon=False)
    axes[1].plot(step, history["val_acc"])
    axes[1].set(title="Validation source-count accuracy", xlabel="Step (thousands)", ylabel="Accuracy")
    axes[2].plot(step, history["val_rmse"])
    axes[2].set(title="Validation DOA RMSE", xlabel="Step (thousands)", ylabel="RMSE (degrees)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_results(results, title=""):
    snrs = list(results)
    Ks = range(1, 5)
    acc = {K: [results[snr][K]["acc"] for snr in snrs] for K in Ks}
    rmse_ = {K: [results[snr][K]["rmse"] for snr in snrs] for K in Ks}
    counts = np.array([sum(results[snr][K]["pred_counts"] for snr in snrs) for K in Ks])
    confusion = counts / counts.sum(axis=1, keepdims=True)

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Accuracy and RMSE vs SNR
    for ax, metric, name, ylabel in [(axes[0], acc, "Source-count accuracy", "Accuracy"),
                                     (axes[1], rmse_, "DOA RMSE", "RMSE (degrees)")]:
        for K, c in zip(Ks, colors):
            ax.plot(snrs, metric[K], "-o", color=c, lw=2, ms=6, label=f"K = {K}")
        ax.set(title=name, xlabel="SNR (dB)", ylabel=ylabel, xticks=snrs)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(bottom=0)

    # Confusion matrix
    ax = axes[2]
    im = ax.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{confusion[i, j]:.2f}", ha="center", va="center",
                    color="white" if confusion[i, j] > 0.5 else "black")
    ax.set(title="Confusion matrix, all SNRs (row-normalized)", xlabel="Predicted K", ylabel="True K",
           xticks=range(4), xticklabels=Ks, yticks=range(4), yticklabels=Ks)
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


# Paper figures
@torch.no_grad()
def dnn(model, y):
    model.cuda().eval()
    out = model(torch.cat([y.real, y.imag], 1))
    return out[0], out[1:]


def test_set(scenario, snr, n, K, seed):
    half_fov, amp_range = SCENARIOS[scenario]["half_fov"], SCENARIOS[scenario]["amp_range"]
    x, k, doa = generate_snapshot_dataset(1, K, 16, [snr], -half_fov, half_fov, n, amp_range, seed)
    x, doa = x[k == K - 1], doa[k == K - 1, :K]
    return torch.complex(x[:, :16], x[:, 16:]), doa


def fig2_data(estimators):
    res = {}
    for label in estimators:
        res[label] = []
    for i, snr in enumerate(PAPER_SNRS):
        y, doa = test_set("B", snr, 4000, K=2, seed=TEST_SEED + 200 + i)
        for label, estimate in estimators.items():
            res[label].append(rmse(estimate(y), doa))
    return res


def fig3_data(estimators):
    res = {}
    for label in estimators:
        res[label] = []
    for i, snr in enumerate(PAPER_SNRS):
        y, doa = test_set("A", snr, 2000, K=4, seed=TEST_SEED + 300 + i)
        for label, estimate in estimators.items():
            res[label].append(rmse(estimate(y), doa))
    return res


def fig5_data():
    torch.manual_seed(5)
    angles = torch.linspace(-80, 80, 3201, device="cuda")
    y0 = steering_vector(16, torch.tensor([[10.0]], device="cuda"))[:, :, 0]

    # Noiseless spectrum per impairment level, one random H each
    res = {"angle": angles.tolist()}
    for label, y in [("No impairments", y0),
                     ("Medium impairments", (impairment_matrix(1, 16, "medium") @ y0[:, :, None])[:, :, 0]),
                     ("High impairments", (impairment_matrix(1, 16, "high") @ y0[:, :, None])[:, :, 0])]:
        p = bartlett_spectrum(y, angles)[0]
        res[label] = (10 * torch.log10(p / p.max())).tolist()
    return res


def fig6_data(estimators):
    res = {}
    for label in estimators:
        res[f"{label} - Medium impairments"] = []
        res[f"{label} - High impairments"] = []

    # Same H for every call: reseed
    torch.manual_seed(6)
    for i, snr in enumerate(PAPER_SNRS):
        y0, doa = test_set("A", snr, 2000, K=4, seed=TEST_SEED + 600 + i)
        for level in ["Medium", "High"]:
            y = (impairment_matrix(2000, 16, level.lower()) @ y0[:, :, None])[:, :, 0]
            for label, estimate in estimators.items():
                res[f"{label} - {level} impairments"].append(rmse(estimate(y), doa))
    return res


def fig7_data(counters):
    res = {}
    for label in counters:
        res[label] = []
    for i, snr in enumerate(PAPER_SNRS):
        x, k, _ = generate_snapshot_dataset(1, 4, 16, [snr], -25.0, 25.0, 2500, (0.5, 1.5), seed=TEST_SEED + 700 + i)
        y = torch.complex(x[:, :16], x[:, 16:])
        for label, count in counters.items():
            res[label].append((count(y) == k + 1).float().mean().item())
    return res


def compute_baselines(path="results/baselines.json"):
    # Load if saved
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)

    data = {
        "fig2": fig2_data({"Maximum Likelihood": lambda y: ml_two_sources(y, 10.0)}),
        "fig3": fig3_data({"Alternating Projections": lambda y: alternating_projection(y, 4, 25.0),
                           "Spatial smooth. + MUSIC": lambda y: ss_music(y, 4, 25.0),
                           "OMP": lambda y: omp(y, 4, 25.0)}),
        "fig5": fig5_data(),
        "fig6": fig6_data({"Alt. Proj.": lambda y: alternating_projection(y, 4, 25.0)}),
        "fig7": fig7_data({"Spatial smooth. + MDL": lambda y: ss_model_order(y)[0],
                           "Spatial smooth. + AIC": lambda y: ss_model_order(y)[1]}),
    }

    # Save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
    return data


def dnn_doa(model, K, half_fov):
    return lambda y: dnn(model, y)[1][K - 1] * half_fov


def dnn_count(model):
    return lambda y: dnn(model, y)[0].argmax(1) + 1


def compute_models(models):
    fig2, fig3, fig6, fig7 = {}, {}, {}, {}
    for name, m in models.items():
        if m["scenario"] == "B":
            fig2[f"DNN {name}"] = dnn_doa(m["model"], 2, 10.0)
        else:
            fig3[f"DNN {name}"] = dnn_doa(m["model"], 4, 25.0)
            fig6[f"DNN {name}"] = dnn_doa(m["model"], 4, 25.0)
            fig7[f"DNN {name}"] = dnn_count(m["model"])
    return {"fig2": fig2_data(fig2), "fig3": fig3_data(fig3), "fig6": fig6_data(fig6), "fig7": fig7_data(fig7)}


def plot_fig1(ax):
    ax.set_axis_off()
    ax.set(xlim=(0, 11.2), ylim=(0, 5))
    box = dict(boxstyle="round,pad=0.8", fc="#dbe7f5", ec="#4a6fa5")

    # Hidden layers
    layers = ["Input 32", "64 neurons", "64 neurons", "128 neurons", "256 neurons", "512 neurons",
              "1024 neurons", "2048 neurons", "256 neurons"]
    for i, c in enumerate(layers):
        ax.text(0.4 + i * 0.95, 2.5, c, rotation=90, ha="center", va="center", fontsize=8, bbox=box)
        if i:
            ax.annotate("FC", xy=(0.25 + i * 0.95, 2.5), xytext=(i * 0.95 - 0.25, 2.5), fontsize=7,
                        va="center", arrowprops=dict(arrowstyle="->"))

    # Heads
    heads = ["Classification\nSoftmax 4 neurons", "1 Target AOA\n1 neuron", "2 Target AOA\n2 neurons",
             "3 Target AOA\n3 neurons", "4 Target AOA\n4 neurons"]
    for j, t in enumerate(heads):
        ax.annotate(t, xy=(8.25, 2.5), xytext=(10.2, 4.5 - j), ha="center", va="center", fontsize=8,
                    bbox=box, arrowprops=dict(arrowstyle="<-"))
    ax.set_title("Fig. 1: 8 hidden FC layers (each + BatchNorm + ReLU), number-of-sources classification "
                 "and AOA regression per class", fontsize=9)


def plot_curves(ax, res, title, ylabel="RMSE [deg.]"):
    for label, values in res.items():
        ax.plot(PAPER_SNRS, values, ls="--" if label.startswith("DNN") else "-", lw=1.6, label=label)
    ax.set(title=title, xlabel="SNR [dB]", ylabel=ylabel, xlim=(0, 40))
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8)


def plot_paper_figures(baselines, models):
    data = {"fig5": baselines["fig5"]}
    for name in ["fig2", "fig3", "fig6", "fig7"]:
        data[name] = baselines[name] | models[name]

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.6, 1, 1])
    plot_fig1(fig.add_subplot(gs[0, :]))
    plot_curves(fig.add_subplot(gs[1, 0]), data["fig2"], "Fig. 2: DNN vs. ML, two sources, FOV 20 deg")
    ax = fig.add_subplot(gs[1, 1])
    plot_curves(ax, data["fig3"], "Fig. 3: four sources, FOV 50 deg")
    ax.set_ylim(0, 4)

    # Fig. 5
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(data["fig5"]["angle"], data["fig5"]["Medium impairments"], "C0--", lw=1.2, label="Medium impairments")
    ax.plot(data["fig5"]["angle"], data["fig5"]["No impairments"], "k-", lw=1.2, label="No impairments")
    ax.plot(data["fig5"]["angle"], data["fig5"]["High impairments"], "m:", lw=1.2, label="High impairments")
    ax.set(title="Fig. 5: Bartlett beampattern with impairments", xlabel="Angle [deg.]",
           ylabel="Amplitude [dB]", xlim=(-80, 80), ylim=(-60, 0))
    ax.grid(alpha=0.35)
    ax.legend(fontsize=8)

    plot_curves(fig.add_subplot(gs[2, 0]), data["fig6"], "Fig. 6: four sources, FOV 50 deg, with impairments")
    plot_curves(fig.add_subplot(gs[2, 1]), data["fig7"], "Fig. 7: number of sources estimation",
                ylabel="Probability of correct number of sources")
    fig.tight_layout()
    plt.show()
