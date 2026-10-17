# notebooks/

Exploration only. Every number in the README, the dashboard and the paper must come from
`python -m ml.run_experiment --config ...` (see the Makefile), never from a notebook.

Handy starting point:

```python
from ml.config import load_config
from ml.data import load_dataset
from ml.features import build_features

cfg = load_config("configs/default.yaml")
df = load_dataset(cfg)
feat = build_features(df, cfg["features"]["windows_s"], cfg["features"]["history_cap"])
feat.groupby("label")[["z_amt", "hour_dev", "cluster_cnt_86400", "new_device"]].mean()
```
