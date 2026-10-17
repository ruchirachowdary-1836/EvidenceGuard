"""Write the synthetic dataset to data/raw/synthetic.csv and print a data report (`make data`)."""
from __future__ import annotations

import argparse

from .config import ROOT, load_config
from .data import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = load_dataset(cfg)
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "synthetic.csv", index=False)
    print(f"{cfg['data']['source']}: {len(df):,} rows, {int(df.label.sum()):,} fraud ({df.label.mean():.2%}), "
          f"{df.entity_key.nunique():,} entities, {df.ts.max() / 86400:.0f} days -> {out / 'synthetic.csv'}")
    print("Reminder: this is simulated data. For IEEE-CIS or PaySim, place the Kaggle files in data/raw/ and set data.source in the config.")


if __name__ == "__main__":
    main()
