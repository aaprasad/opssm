"""Load the Kato et al. 2015 C. elegans whole-brain calcium data (OSF osf.io/2395t; MATLAB v7.3 .mat).

Each file (WT_Stim.mat = 7 worms, WT_NoStim.mat = 5 worms, AVA_HisCl.mat) is a MATLAB struct saved under a
single root variable (e.g. `WT_Stim`), whose fields are per-worm cell arrays of length n_worms:
  traces           (N, T)  bleaching-corrected dF/F  <- what we use (transpose to (T, N))
  traces_raw       (N, T)  uncorrected
  tracesDif        (N, T)  derivative
  States           (T, 1)  1-indexed behavior-state label per timepoint
  IDs              (N, 1)  identified neuron names (many empty/unlabeled)
  fps              scalar  frames/sec (VARIES per worm -> dt = 1/fps)
  timeVectorSeconds(T,)    time axis
See readme_Kato2015.txt (downloaded alongside). v7.3 = HDF5, so read with h5py + reference deref.
"""
import h5py
import numpy as np

# integer-code -> behavior-state name, per file (readme_Kato2015.txt). 1-indexed in the .mat.
STATE_NAMES = {
    "WT_Stim":   ["FWD", "REV", "REVSUS", "TURN"],                                    # 4 states
    "AVA_HisCl": ["FWD", "REV", "REVSUS", "TURN"],
    "WT_NoStim": ["FWD", "SLOW", "DT", "VT", "REV1", "REV2", "REVSUS", "NOSTATE"],    # 8 states
}

# field-name aliases -- the two files use DIFFERENT schemas (WT_Stim vs WT_NoStim).
_ALIAS = {
    "traces": ["traces", "deltaFOverF_bc", "deltaFOverF"],   # bleaching-corrected dF/F
    "ids":    ["IDs", "NeuronNames"],
    "states": ["States"],
    "fps":    ["fps"],
}


def _root(f):
    return next(k for k in f.keys() if not k.startswith("#"))


def _field(g, key):
    for name in _ALIAS[key]:
        if name in g:
            return name
    raise KeyError(f"none of {_ALIAS[key]} in {list(g.keys())}")


def n_worms(path):
    with h5py.File(path, "r") as f:
        g = f[_root(f)]
        return g[_field(g, "traces")].shape[0]


def _char(f, obj):
    """Decode a MATLAB char array possibly wrapped in nested HDF5 refs -> str ('' if empty)."""
    a = np.array(obj)
    while a.size and isinstance(a.ravel()[0], h5py.Reference):
        a = np.array(f[a.ravel()[0]])
    return "".join(chr(int(c)) for c in a.ravel() if 0 < int(c) < 0x110000)


def _decode_ids(f, ref_ds):
    """ref_ds = the worm's (N,1) neuron-name cell. -> list[str] ('' = unlabeled)."""
    return [_char(f, f[ref_ds[i, 0]]) for i in range(ref_ds.shape[0])]


# WT_NoStim stores States as a struct of one-hot indicator vectors; keep this readme order.
_NOSTIM_ORDER = ["fwd", "slow", "dt", "vt", "rev1", "rev2", "revsus", "nostate"]


def _states(f, obj):
    """obj = a worm's deref'd States. WT_Stim: plain (T,1) 1-indexed int vector. WT_NoStim: a struct of
    8 one-hot vectors -> argmax. Returns (T,) int64, 0-indexed."""
    if isinstance(obj, h5py.Group):                                  # WT_NoStim one-hot struct
        M = np.stack([np.array(obj[k]).ravel() for k in _NOSTIM_ORDER], axis=1)   # (T, 8)
        lab = M.argmax(1)
        lab[M.sum(1) == 0] = _NOSTIM_ORDER.index("nostate")          # no active state -> NOSTATE
        return lab.astype(np.int64)
    return np.array(obj).ravel().astype(np.int64) - 1                # WT_Stim: 1-indexed -> 0-indexed


def load_worm(path, worm):
    """Load one worm -> dict:
      traces (T, N) float32 dF/F, states (T,) int64 0-indexed behavior label, dt float, fps float,
      neuron_ids list[str] (len N; '' = unlabeled), state_names list[str], name str, n_nan int.
    """
    with h5py.File(path, "r") as f:
        root = _root(f); g = f[root]
        cell = lambda key: np.array(f[g[_field(g, key)][worm, 0]])   # deref worm's cell -> ndarray
        traces = cell("traces").astype(np.float32).T                 # (N,T) -> (T,N)
        states = _states(f, f[g[_field(g, "states")][worm, 0]])      # (T,) 0-indexed
        fps = float(cell("fps").ravel()[0])
        ids = _decode_ids(f, f[g[_field(g, "ids")][worm, 0]])
    return dict(traces=traces, states=states, dt=1.0 / fps, fps=fps, neuron_ids=ids,
                state_names=STATE_NAMES.get(root, []), name=f"{root}_worm{worm}",
                n_nan=int(np.isnan(traces).sum()))
