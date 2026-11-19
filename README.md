# Deep DOA estimation from a single snapshot

MLP-based direction-of-arrival (DOA) estimation for a uniform linear array, replicating
*Performance Advantages of Deep Neural Networks for Angle of Arrival Estimation*
(Bialer, Garnett, Tirer, 2019, [arXiv:1902.03569](https://arxiv.org/abs/1902.03569)).


## Setup

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu130   # GPU; skip for CPU
pip install -r requirements.txt
```

