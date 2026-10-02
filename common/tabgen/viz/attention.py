"""Fig. 4 of the paper: how much attention the label token pays to each feature.

For a serialized row we take the attention from the target-VALUE token (averaged over heads, last
`n_layers` layers) to every earlier token and sum it per field. With the target at the end
(permute_x) every feature is visible; with the target shuffled to the front (permute_xy) the label
token cannot attend to features that come after it, which is the paper's argument for permute_x.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def field_attention(model, tok, text: str, fields: list[str], target_field: str, n_layers: int = 2) -> dict:
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        att = model(**enc, output_attentions=True).attentions            # tuple(layers) of (1, H, L, L)
    A = torch.stack(att[-n_layers:]).float().mean(dim=(0, 2))[0].cpu().numpy()   # (L, L)
    # character span of every field "name is value"
    spans, pos = {}, 0
    for piece in text.split(", "):
        name = piece.split(" is ")[0]
        spans[name] = (pos, pos + len(piece))
        pos += len(piece) + 2
    ts, te = spans[target_field]
    val_start = ts + len(target_field) + 4
    q = [i for i, (a, b) in enumerate(offsets) if a >= val_start and b <= te and b > a]
    q = q[0] if q else len(offsets) - 1
    scores = {}
    for f in fields:
        s, e = spans[f]
        idx = [i for i, (a, b) in enumerate(offsets) if a >= s and b <= e and b > a and i <= q]
        scores[f] = float(A[q, idx].sum()) if idx else 0.0
    return scores


def plot_attention_comparison(results: dict[str, dict], path: Path, title: str = "") -> Path:
    """results: {'permute_x (target last)': {field: score}, 'permute_xy (target first)': {...}}"""
    fields = list(next(iter(results.values())).keys())
    x = np.arange(len(fields))
    w = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(fields) + 2), 3.5))
    for i, (label, sc) in enumerate(results.items()):
        ax.bar(x + i * w, [sc[f] for f in fields], width=w, label=label)
    ax.set_xticks(x + w * (len(results) - 1) / 2, fields, rotation=60, fontsize=8, ha="right")
    ax.set_ylabel("attention from label token")
    ax.set_title(title or "Label-token attention per feature (Fig. 4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
