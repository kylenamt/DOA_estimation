import math

import torch

SCENARIOS = {
    "A": dict(half_fov=25.0, amp_range=(0.5, 1.5)),  # FOV 50°, |s| ~ U[0.5, 1.5]
    "B": dict(half_fov=10.0, amp_range=(1.0, 1.0)),  # FOV 20°, |s| = 1
}

IMPAIRMENTS = {"medium": (5.0, 30.0), "high": (10.0, 20.0)}  # (sigma_theta [deg], Gamma [dB])


def sind(degrees):
    return torch.sin(torch.deg2rad(degrees))


def steering_vector(m_elems, angle_deg):
    n = torch.arange(m_elems, device=angle_deg.device, dtype=angle_deg.dtype)
    return torch.exp(-1j * math.pi * n[:, None] * sind(angle_deg)[..., None, :])


def compute_snapshot(signals, doas, T, m_elems, rng):
    n = signals.shape[0]
    noise = torch.complex(torch.randn(n, m_elems, T, generator=rng, device="cuda"),
                          torch.randn(n, m_elems, T, generator=rng, device="cuda")) / math.sqrt(2)
    a = steering_vector(m_elems, doas)  # array response
    noiseless_signal = a @ signals
    snapshot = noiseless_signal + noise
    return snapshot


def generate_K_source_signals(K, T, gamma, rng, amp_range):
    n = len(gamma)
    mag = amp_range[0] + (amp_range[1] - amp_range[0]) * torch.rand(n, K, T, generator=rng, device="cuda")
    phase = 2 * math.pi * torch.rand(n, K, T, generator=rng, device="cuda")
    return torch.sqrt(gamma)[:, None, None] * mag * torch.exp(1j * phase)


def generate_snapshot_dataset(nof_snapshots, nof_signals, m_elems, SNRs, fov_start, fov_end, samples_per_snr,
                              amp_range, seed=73):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    gamma = (10 ** (torch.as_tensor(SNRs, device="cuda", dtype=torch.float32) / 10)).repeat_interleave(samples_per_snr)

    # Per number of sources: x = [Re(y), Im(y)], k = K - 1, DOAs sorted and zero-padded to 4
    xs, ks, ds = [], [], []
    for K in range(1, nof_signals + 1):
        s = generate_K_source_signals(K, nof_snapshots, gamma, rng, amp_range)
        doa = fov_start + (fov_end - fov_start) * torch.rand(len(gamma), K, generator=rng, device="cuda")
        y = compute_snapshot(s, doa, nof_snapshots, m_elems, rng).flatten(1)
        xs.append(torch.cat([y.real, y.imag], 1))
        ks.append(torch.full((len(gamma),), K - 1, device="cuda"))
        ds.append(torch.nn.functional.pad(doa.sort(1).values, (0, 4 - K)))

    return torch.cat(xs), torch.cat(ks), torch.cat(ds)


def impairment_matrix(n, m_elems, level):
    sigma, gamma_db = IMPAIRMENTS[level]
    phase = torch.deg2rad(sigma * torch.randn(n, m_elems, device="cuda"))
    cross_talk = torch.complex(torch.randn(n, m_elems, m_elems, device="cuda"),
                               torch.randn(n, m_elems, m_elems, device="cuda")) / math.sqrt(2)
    # Diagonal: phase offsets, off-diagonal: cross talk
    eye = torch.eye(m_elems, device="cuda", dtype=torch.bool)
    return torch.where(eye, torch.diag_embed(torch.exp(1j * phase)), cross_talk * 10 ** (-gamma_db / 20))
