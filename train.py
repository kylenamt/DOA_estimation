"""Train BaseDOAModel from the terminal. Example:
    python train.py --epochs 50 --batch-size 256 --half-fov 25 --save base_model.pth
"""
import argparse
import json

from src.data_generation import generate_snapshot_dataset
from src.dataloader import make_loaders
from src.models import BaseDOAModel
from src.trainer import train

p = argparse.ArgumentParser()
p.add_argument("--m-elems", type=int, default=16)
p.add_argument("--snrs", type=float, nargs="+", default=[0, 5, 10, 15, 20])
p.add_argument("--half-fov", type=float, default=90)
p.add_argument("--samples-per-snr", type=int, default=10000)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--val-frac", type=float, default=0.2)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--doa-weight", type=float, default=1.0)
p.add_argument("--seed", type=int, default=73)
p.add_argument("--save", default="base_model.pth", help="model weights; history goes to <save>.json")
args = p.parse_args()
print(vars(args))

snapshots, doas = generate_snapshot_dataset(
    nof_snapshots=1, nof_signals=4, m_elems=args.m_elems, SNRs=args.snrs,
    fov_start=-args.half_fov, fov_end=args.half_fov,
    samples_per_snr=args.samples_per_snr, seed=args.seed)
train_loader, val_loader = make_loaders(snapshots, doas, batch_size=args.batch_size,
                                        val_fraction=args.val_frac, doa_scale=args.half_fov)

model = BaseDOAModel(d_model=2 * args.m_elems)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

history = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr,
                doa_scale=args.half_fov, doa_weight=args.doa_weight, save_path=args.save)

# Save the learning curve so the notebook can plot it without retraining
with open(args.save + ".json", "w") as f:
    json.dump({"args": vars(args), "history": history}, f, indent=1)
print(f"Saved {args.save} and {args.save}.json")
