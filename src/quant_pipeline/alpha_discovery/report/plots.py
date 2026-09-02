from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt


def effect_curve(frame, path: str | Path, x: str = "target_horizon", y: str = "top_bottom_spread") -> Path:
    target = Path(path); figure, axis = plt.subplots(figsize=(8, 4.5)); axis.plot(frame[x].astype(str), frame[y], marker="o"); axis.axhline(0, color="black", linewidth=.8); axis.set(xlabel=x, ylabel=y); figure.tight_layout(); figure.savefig(target, dpi=160); plt.close(figure); return target
