#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import numpy as np

from evo import MLPPolicy
from solar_ants_env import SolarAntsEnv
from viewer import Viewer


def _latest_params_path(run_dir: str, pattern: str) -> str | None:
    candidates = glob.glob(os.path.join(run_dir, pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def _infer_act_dim(param_dim: int, obs_dim: int, hidden_dim: int) -> int | None:
    base = hidden_dim * obs_dim + hidden_dim
    rem = param_dim - base
    denom = hidden_dim + 1
    if rem <= 0 or rem % denom != 0:
        return None
    return rem // denom


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize an EDA-trained controller using the Viewer."
    )
    parser.add_argument(
        "--params",
        default=None,
        help="Path to saved params (.npy). Defaults to latest in eda_runs/.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Hidden layer width used during EDA training.",
    )
    parser.add_argument(
        "--act-dim",
        type=int,
        default=None,
        help="Action dimension override (inferred by default).",
    )
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--initial-zoom", type=float, default=0.25)
    parser.add_argument("--substeps-per-frame", type=int, default=1)
    args = parser.parse_args()

    params_path = args.params
    if params_path is None:
        params_path = _latest_params_path("eda_runs", "best_params_fullcov_*.npy")
        if params_path is None:
            print("No params found in eda_runs/. Provide --params.", file=sys.stderr)
            return 1

    params = np.load(params_path)
    params = np.asarray(params, dtype=np.float64).ravel()

    system = SolarAntsEnv.demo_solar_system()
    agent = system.agents[0]
    obs = np.asarray(agent.getSensorReadings(), dtype=np.float64)
    obs_dim = int(obs.shape[0])

    if args.act_dim is None:
        act_dim = _infer_act_dim(params.shape[0], obs_dim, args.hidden_dim)
        if act_dim is None:
            print(
                "Could not infer action dim. Provide --act-dim.",
                file=sys.stderr,
            )
            return 1
    else:
        act_dim = args.act_dim

    policy = MLPPolicy(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=args.hidden_dim,
        params=params,
    )

    def control_handle() -> None:
        obs = np.asarray(agent.getSensorReadings(), dtype=np.float64)
        action = policy.act(obs)
        agent.applyControlForce(action)

    viewer = Viewer(
        system,
        control_handle,
        width=args.width,
        height=args.height,
        initial_zoom=args.initial_zoom,
        substeps_per_frame=args.substeps_per_frame,
        window_name=f"EDA Viewer ({os.path.basename(params_path)})",
    )
    viewer.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
