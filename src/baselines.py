import math

import torch

from src.data_generation import steering_vector

STEP = 0.1  # grid spacing [deg]
N_SUB = 9   # spatial smoothing sub-arrays


def angle_grid(half_fov, step=STEP):
    return torch.linspace(-half_fov, half_fov, int(round(2 * half_fov / step)) + 1, device="cuda",
                          dtype=torch.float64)


def bartlett_spectrum(y, theta_deg):
    A = steering_vector(y.shape[1], theta_deg.double())
    return (y.to(torch.complex128) @ A.conj()).abs() ** 2 / y.shape[1]


# ML, 2 sources
def _pair_energy(b1, b2, g, M):
    return (M * (b1.abs() ** 2 + b2.abs() ** 2) - 2 * (b1.conj() * g * b2).real) / (M ** 2 - g.abs() ** 2).clamp_min(1e-6)


def ml_two_sources(y, half_fov):
    y = y.to(torch.complex128)
    M = y.shape[1]
    grid = angle_grid(half_fov)
    A = steering_vector(M, grid)

    # Search all pairs on the grid
    i, j = torch.triu_indices(len(grid), len(grid), offset=1, device="cuda")
    g = (A[:, i].conj() * A[:, j]).sum(0)
    est = []
    for yc in y.split(500):
        B = yc @ A.conj()
        best = _pair_energy(B[:, i], B[:, j], g, M).argmax(1)
        est.append(torch.stack([grid[i[best]], grid[j[best]]], 1))
    est = torch.cat(est)

    # Refine on 0.01° and 0.001° local grids
    for prev, fine in [(STEP, 0.01), (0.01, 0.001)]:
        offs = torch.arange(-prev, prev + fine / 2, fine, device="cuda", dtype=torch.float64)
        t1, t2 = est[:, :1] + offs, est[:, 1:] + offs
        a1, a2 = steering_vector(M, t1), steering_vector(M, t2)
        b1, b2 = (a1.conj() * y[:, :, None]).sum(1), (a2.conj() * y[:, :, None]).sum(1)
        e = _pair_energy(b1[:, :, None], b2[:, None, :], a1.conj().transpose(1, 2) @ a2, M)
        idx, L = e.flatten(1).argmax(1), len(offs)
        est = torch.stack([t1.gather(1, (idx // L)[:, None])[:, 0], t2.gather(1, (idx % L)[:, None])[:, 0]], 1)
    return est.sort(1).values.float()


# Alternating projection
def _best_angle(A_fixed, y, Agrid):
    Q = torch.linalg.qr(A_fixed)[0]
    P = torch.eye(y.shape[1], dtype=y.dtype, device="cuda") - Q @ Q.conj().transpose(1, 2)
    num = ((P @ y[:, :, None])[:, :, 0].conj() @ Agrid).abs() ** 2
    den = (Agrid.conj() * (P @ Agrid)).sum(1).real
    return torch.where(den > 1e-6, num / den.clamp_min(1e-12), 0.0).argmax(1)


def alternating_projection(y, K, half_fov):
    y = y.to(torch.complex128)
    grid = angle_grid(half_fov)
    Agrid = steering_vector(y.shape[1], grid)

    # Initialization: add sources one at a time
    idx = (y.conj() @ Agrid).abs().argmax(1)[:, None]
    for k in range(1, K):
        idx = torch.cat([idx, _best_angle(Agrid[:, idx].permute(1, 0, 2), y, Agrid)[:, None]], 1)

    # 10 rounds: re-estimate each source with the others fixed
    for _ in range(10 if K > 1 else 0):
        for k in range(K):
            others = torch.cat([idx[:, :k], idx[:, k + 1:]], 1)
            idx[:, k] = _best_angle(Agrid[:, others].permute(1, 0, 2), y, Agrid)
    return grid[idx].sort(1).values.float()


# Spatial smoothing + MUSIC
def smoothed_covariance(y):
    Y = y.to(torch.complex128).unfold(1, y.shape[1] - N_SUB + 1, 1)
    return Y.transpose(1, 2) @ Y.conj() / N_SUB


def ss_music(y, K, half_fov):
    # Noise subspace (eigh on CPU)
    R = smoothed_covariance(y)
    En = torch.linalg.eigh(R.cpu())[1][:, :, :R.shape[1] - K].cuda()

    # MUSIC spectrum
    grid = angle_grid(half_fov)
    spec = 1 / (En.conj().transpose(1, 2) @ steering_vector(R.shape[1], grid)).abs().pow(2).sum(1)

    # K largest peaks (missing peaks -> largest)
    pad = torch.full_like(spec[:, :1], -torch.inf)
    is_peak = (spec >= torch.cat([pad, spec[:, :-1]], 1)) & (spec > torch.cat([spec[:, 1:], pad], 1))
    vals, idx = torch.where(is_peak, spec, -torch.inf).topk(K, dim=1)
    idx = torch.where(torch.isfinite(vals), idx, idx[:, :1])
    return grid[idx].sort(1).values.float()


# OMP
def omp(y, K, half_fov):
    y = y.to(torch.complex128)
    grid = angle_grid(half_fov)
    D = steering_vector(y.shape[1], grid)

    # Pick best atom, re-fit by least squares, K times
    r, idx = y, []
    for _ in range(K):
        idx.append((r @ D.conj()).abs().argmax(1))
        As = D[:, torch.stack(idx, 1)].permute(1, 0, 2)
        r = y - (As @ torch.linalg.lstsq(As, y[:, :, None]).solution)[:, :, 0]
    return grid[torch.stack(idx, 1)].sort(1).values.float()


# MDL and AIC
def ss_model_order(y):
    R = smoothed_covariance(y)
    p, N = R.shape[1], N_SUB
    lam = torch.linalg.eigvalsh(R.cpu()).cuda().flip(1)

    # Criteria for k = 1..4
    mdl, aic = [], []
    for k in range(1, 5):
        tail = lam[:, k:]
        L = N * (p - k) * (tail.mean(1).log() - tail.log().mean(1))
        aic.append(2 * L + 2 * k * (2 * p - k))
        mdl.append(L + 0.5 * k * (2 * p - k) * math.log(N))
    return torch.stack(mdl, 1).argmin(1) + 1, torch.stack(aic, 1).argmin(1) + 1
