"""Draw SHAP beeswarm + bar plots from the shap_values.npz files written by the lock-report scripts (run under /usr/bin/python3, which has the shap library)."""
import argparse, sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--title", default=""); a = ap.parse_args()
    for f in sorted(Path(a.root).rglob("shap_values.npz")):
        d = np.load(f, allow_pickle=True); sv, Z, names = d["sv"], d["Z"], list(d["names"]); label = f.parent.name.replace("__", " | ").replace("_", "+") if False else f.parent.name.replace("__", " | ")
        for kind, fn in (("dot", "shap_beeswarm.png"), ("bar", "shap_bar.png")):
            plt.figure(figsize=(8.5, max(2.6, 0.32 * len(names) + 1.6)))
            shap.summary_plot(sv, Z, feature_names=names, plot_type=kind, show=False, max_display=20)
            plt.title(f"{a.title} {label}".strip(), fontsize=10); plt.tight_layout(); plt.savefig(f.parent / fn, dpi=150); plt.close()
        print("plotted", f.parent.name, flush=True)


if __name__ == "__main__":
    main()
