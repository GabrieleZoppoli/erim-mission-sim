"""Small, dependency-free persistence: pytrees to .npz, CSV rows appended with a header written once, JSON meta."""
import csv
import json
import os
import subprocess
import jax
import jax.numpy as jnp
import numpy as np


def save_pytree(path, tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    np.savez_compressed(path, treedef=np.array(str(treedef)), n=np.array(len(leaves)),
                        **{f"leaf_{i}": np.asarray(l) for i, l in enumerate(leaves)})


def load_pytree(path, like):
    """Load leaves into the structure of `like` (a pytree with the same shape)."""
    d = np.load(path, allow_pickle=False)
    n = int(d["n"])
    leaves = [jnp.asarray(d[f"leaf_{i}"]) for i in range(n)]
    _, treedef = jax.tree_util.tree_flatten(like)
    return jax.tree_util.tree_unflatten(treedef, leaves)


def append_csv(path, rows, fieldnames=None):
    if not rows:
        return
    fieldnames = fieldnames or list(rows[0].keys())
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in fieldnames})


def _fmt(v):
    if v is None:
        return "NA"
    if isinstance(v, (float, np.floating, jnp.ndarray)) and np.ndim(v) == 0:
        v = float(v)
        return "NA" if np.isnan(v) else f"{v:.6g}"
    return v


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return "unknown"


def device_info():
    d = jax.devices()[0]
    return {"jax": jax.__version__, "backend": d.platform, "device": str(d), "n_devices": len(jax.devices())}


def done_marker(out, unit_id):
    return os.path.join(out, "units", unit_id.replace("/", "_") + ".done")


def mark_done(out, unit_id):
    p = done_marker(out, unit_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def is_done(out, unit_id):
    return os.path.exists(done_marker(out, unit_id))
