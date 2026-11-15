# Deep DOA estimation from a single snapshot

MLP-based direction-of-arrival (DOA) estimation for a uniform linear array, replicating
*Performance Advantages of Deep Neural Networks for Angle of Arrival Estimation*
(Bialer, Garnett, Tirer, 2019, [arXiv:1902.03569](https://arxiv.org/abs/1902.03569)).

## Structure

```
src/
├── data_generation.py   # simulated ULA snapshots (signal model y = A s + n)
├── dataloader.py        # snapshots -> tensors, train/val/test DataLoaders
├── models.py            # MLP: source-count classifier + DOA regression heads
└── trainer.py           # loss, training loop, evaluation
HW5.ipynb                # report: experiments and results
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU; skip for CPU
pip install -r requirements.txt
```

Then open `HW5.ipynb` and select the `.venv` kernel.
