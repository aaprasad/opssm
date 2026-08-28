"""JAX-backend training entrypoint. Run in the .[jax] venv, AFTER scripts/bridge.py has produced the data.

The .[jax] venv has no torch, so JAX training cannot build the (torch/Lightning) datamodule. scripts/bridge.py
(run in the .[torch] venv) resolves the SAME Hydra config once and bakes the data + all resolved/data-derived
hyperparameters + training-loop knobs + eval refs into <train_dir>/bridge.npz. This entrypoint just consumes
that npz -- the single source of truth -- so it needs no config framework of its own.

    python scripts/bridge.py    experiment=kato ... train_dir=dump/kato_nostim0 backend=jax   # .[torch] venv
    python scripts/train_jax.py                      train_dir=dump/kato_nostim0             # .[jax] venv

Positional <train_dir> (or train_dir=...); optional key=value overrides (seed=, n_steps=, val_every=, or any
hyperparameter, e.g. warmup=2000). Promoted from the validated dump/jax_port/run.py.
"""
import os
import sys

import jax


def _coerce(s):
    for f in (int, float):
        try:
            return f(s)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None, "null": None}.get(s.lower(), s)


def main(argv):
    train_dir, overrides = None, {}
    for a in argv:
        k, _, v = a.partition("=")
        if v == "":
            train_dir = k                                            # bare positional -> train_dir
        elif k == "train_dir":
            train_dir = v
        else:
            overrides[k] = _coerce(v)
    if train_dir is None:
        raise SystemExit("usage: train_jax.py <train_dir> [seed=N n_steps=N val_every=N <hp>=<val> ...]")

    from opssm.models.jax.train import train, load_refs, whole_trace_recon

    npz_path = os.path.join(train_dir, "bridge.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} not found -- run scripts/bridge.py (in the .[torch] venv) first:\n"
                         f"  python scripts/bridge.py experiment=<exp> train_dir={train_dir} backend=jax")
    refs, hp = load_refs(npz_path)
    if str(refs.get("backend", "jax")) != "jax":
        raise SystemExit(f"{npz_path} was bridged with backend={refs.get('backend')!r}, not 'jax'. "
                         f"Re-bridge with backend=jax.")

    seed = int(overrides.pop("seed", int(refs.get("seed", 0))))
    n_steps = int(overrides.pop("n_steps", int(refs.get("n_steps", 14000))))
    val_every = int(overrides.pop("val_every", int(refs.get("val_every", 2000))))
    for k, v in overrides.items():                                   # remaining overrides layer onto the bridged hp
        hp[k] = v
        print(f"[override] {k}={v!r}")
    fig_dir = os.path.join(train_dir, "figs")
    print(f"train_jax: {npz_path} system={hp['system']} d={hp['latent_dim']} data_size={hp['data_size']} "
          f"init={hp.get('init_method')} bootstrap={hp.get('bootstrap_mstep')} warmup={hp.get('warmup')} "
          f"n_steps={n_steps} seed={seed} noise_std_eff={float(hp['noise_std']):.4f}", flush=True)

    state, hist = train(refs, hp, n_steps=n_steps, key=jax.random.PRNGKey(seed), val_every=val_every,
                        fig_dir=fig_dir)

    if "y_full_raw" in refs:                                         # Kato: whole-trace recon + eval figures
        import numpy as np
        import jax.numpy as jnp
        from opssm.models.jax.kato_eval import kato_report_jax
        r2w, z_hat = whole_trace_recon(state["op"], state["C_cur"], state["d_cur"], refs, hp,
                                       jax.random.PRNGKey(7))
        C = np.asarray(state["C_cur"]); dd = np.asarray(state["d_cur"])
        y_hat_std = z_hat @ C.T + dd
        dn = state["drift_net"]
        drift_fn = lambda z: np.asarray(dn.net(jnp.asarray(z, dtype=jnp.float32)))   # noqa: E731
        name = refs.get("name", hp["system"])
        outdir = os.path.join(train_dir, "kato_eval")
        print(f"train_jax: WHOLE-TRACE recon_r2 = {r2w:.4f}; rendering eval figures -> {outdir}", flush=True)
        kato_report_jax(outdir, name, z_hat, y_hat_std, refs["y_full_raw"], refs["y_full_std"],
                        refs["obs_mean"], refs["obs_scale"], refs["states_full"], refs["state_names"],
                        refs["neuron_ids"], hp["dt"], drift_fn)
    print("train_jax: done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
