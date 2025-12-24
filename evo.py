#!/usr/bin/env python3
"""
Global-Archive Full-Covariance Gaussian EDA for SolarAnts (Gymnasium env)
- Multiprocessing parallel evaluation
- Keeps a global archive of all (params, fitness)
- Global elites = top-N fitness across *all generations*
- Fits a full-covariance Gaussian N(mu, Sigma) to global elites
- Eigenvalue clamping (covariance floor) prevents collapse
- Exploration scale inflates covariance for broader sampling
- Diverse initialization: mixture of Gaussians (multi-scale) + uniform fraction
- Logs JSONL + saves best params (npy)

Run:
  python3 evolve_archive_eda_fullcov.py

Adjust:
  - Import of SolarAntsEnv below
  - EDAConfig fields (population size, elite size, etc.)
"""

import os
import time
import json
import sys
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass

# ---- adjust this import to your project layout ----
from solar_ants_env import SolarAntsEnv  # the module where your SolarAntsEnv lives


# ============================================================
# Policy (MLP controller): obs -> hidden -> act
# ============================================================

class MLPPolicy:
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 32, params=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim

        if params is None:
            self.W1 = np.random.randn(hidden_dim, obs_dim) * 0.05
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(act_dim, hidden_dim) * 0.05
            self.b2 = np.zeros(act_dim)
        else:
            self.unpack(params)

    def act(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ obs + self.b1)
        a = np.tanh(self.W2 @ h + self.b2)
        return a

    def pack(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def unpack(self, params: np.ndarray) -> None:
        idx = 0
        w1_size = self.hidden_dim * self.obs_dim
        self.W1 = params[idx:idx + w1_size].reshape(self.hidden_dim, self.obs_dim)
        idx += w1_size

        self.b1 = params[idx:idx + self.hidden_dim]
        idx += self.hidden_dim

        w2_size = self.act_dim * self.hidden_dim
        self.W2 = params[idx:idx + w2_size].reshape(self.act_dim, self.hidden_dim)
        idx += w2_size

        self.b2 = params[idx:idx + self.act_dim]
        idx += self.act_dim


def policy_param_dim(obs_dim: int, act_dim: int, hidden_dim: int) -> int:
    return hidden_dim * obs_dim + hidden_dim + act_dim * hidden_dim + act_dim


# ============================================================
# Parallel evaluation
# ============================================================

def evaluate_policy_worker(args):
    """
    Worker-safe evaluation.
    Each worker builds its own env and runs n_episodes rollouts.

    args:
      params: np.ndarray (d,)
      env_builder: callable that returns initialized System (your builder)
      hidden_dim: int
      n_episodes: int
      base_seed: int
      max_steps: int
      n_substeps: int
    """
    idx, params, env_builder, hidden_dim, n_episodes, base_seed, max_steps, n_substeps = args

    total = 0.0
    for ep in range(n_episodes):
        env = SolarAntsEnv(
            system_builder=env_builder,
            n_substeps=n_substeps,
            max_steps=max_steps,
        )
        obs, _ = env.reset(seed=base_seed + ep)

        policy = MLPPolicy(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            hidden_dim=hidden_dim,
            params=params,
        )

        done = False
        ep_reward = 0.0
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            done = bool(terminated) or bool(truncated)

        env.close()
        total += ep_reward

    return idx, total / float(n_episodes)


# ============================================================
# Global-archive Full-Covariance Gaussian EDA
# ============================================================

@dataclass
class EDAConfig:
    # Population / archive
    population_size: int = 1000
    elite_size: int = 1000                  # elites selected from *global* archive
    archive_max_size: int = 10000          # keep only best K overall
    min_archive_to_fit: int = 3000          # before this, keep exploring (broad sampling)

    # Policy
    hidden_dim: int = 32

    # Evaluation
    eval_episodes: int = 3                # increase to 3 later for robustness
    max_steps: int = 10000
    n_substeps: int = 1

    # Init diversity
    init_mix_sigmas: tuple = (0.01, 0.1, 0.5)
    init_uniform_frac: float = 0.2
    init_uniform_range: float = 0.5       # uniform in [-range, +range]

    # Full covariance fit + stabilization
    eigen_floor: float = 1e-6             # clamp eigenvalues of covariance to at least this
    ridge: float = 1e-12                  # tiny diagonal ridge for numerical stability

    # Exploration inflation (covariance multiplier)
    exploration_scale_start: float = 2.0  # Sigma <- (scale^2) * Sigma, early
    exploration_scale_end: float = 1.0    # anneal toward this
    exploration_anneal_gens: int = 80

    # Misc
    generations: int = 100
    seed: int = 42
    n_workers: int | None = 20          # None = use all cores


class GlobalArchiveFullCovEDA:
    def __init__(self, param_dim: int, rng: np.random.Generator, cfg: EDAConfig):
        self.param_dim = param_dim
        self.rng = rng
        self.cfg = cfg

        # Global archive: best K retained (params, fitness)
        self.archive_params: list[np.ndarray] = []
        self.archive_fitness: list[float] = []

    # ---------- archive handling ----------

    def add_to_archive(self, params_list, fitness_list):
        for p, f in zip(params_list, fitness_list):
            self.archive_params.append(np.asarray(p, dtype=np.float64))
            self.archive_fitness.append(float(f))

        # Keep only top archive_max_size
        if len(self.archive_fitness) > self.cfg.archive_max_size:
            idx = np.argsort(np.asarray(self.archive_fitness))[::-1]
            idx = idx[: self.cfg.archive_max_size]
            self.archive_params = [self.archive_params[i] for i in idx]
            self.archive_fitness = [self.archive_fitness[i] for i in idx]

    def best_so_far(self):
        if not self.archive_fitness:
            return None, None
        i = int(np.argmax(np.asarray(self.archive_fitness)))
        return self.archive_params[i], self.archive_fitness[i]

    def elites(self):
        if not self.archive_fitness:
            return [], []
        idx = np.argsort(np.asarray(self.archive_fitness))[::-1]
        k = min(self.cfg.elite_size, len(idx))
        elite_idx = idx[:k]
        return [self.archive_params[i] for i in elite_idx], [self.archive_fitness[i] for i in elite_idx]

    # ---------- initialization ----------

    def sample_initial_population(self):
        n = self.cfg.population_size
        pop = []

        n_uniform = int(round(n * self.cfg.init_uniform_frac))
        n_rest = n - n_uniform

        # Uniform part
        if n_uniform > 0:
            r = self.cfg.init_uniform_range
            for _ in range(n_uniform):
                pop.append(self.rng.uniform(-r, r, size=self.param_dim))

        # Mixture of Gaussians at different scales
        sigmas = list(self.cfg.init_mix_sigmas)
        for _ in range(n_rest):
            s = sigmas[int(self.rng.integers(0, len(sigmas)))]
            pop.append(self.rng.normal(0.0, s, size=self.param_dim))

        self.rng.shuffle(pop)
        return pop

    # ---------- covariance fitting + clamping ----------

    def fit_full_cov_from_elites(self):
        elite_params, elite_fit = self.elites()
        if len(elite_params) < 2:
            # Degenerate: cannot estimate covariance; return isotropic
            mu = elite_params[0].copy()
            Sigma = np.eye(self.param_dim, dtype=np.float64) * self.cfg.eigen_floor
            return mu, Sigma

        X = np.stack(elite_params, axis=0)  # (k, d)
        mu = X.mean(axis=0)
        Xc = X - mu

        # ML covariance (ddof=0): (1/k) Xc^T Xc
        # Using dot is fast and stable
        k = X.shape[0]
        Sigma = (Xc.T @ Xc) / float(k)

        # Numerical ridge
        if self.cfg.ridge > 0.0:
            Sigma = Sigma + np.eye(self.param_dim, dtype=np.float64) * self.cfg.ridge

        # Eigenvalue clamp: Sigma = Q diag(max(lam, floor)) Q^T
        # Use eigh (symmetric PSD)
        lam, Q = np.linalg.eigh(Sigma)
        lam_clamped = np.maximum(lam, self.cfg.eigen_floor)
        Sigma_clamped = (Q * lam_clamped) @ Q.T  # Q diag(lam) Q^T, using broadcasting

        # Ensure symmetry (tiny numerical drift)
        Sigma_clamped = 0.5 * (Sigma_clamped + Sigma_clamped.T)

        return mu, Sigma_clamped

    # ---------- exploration scaling ----------

    def exploration_scale(self, gen: int) -> float:
        g = min(gen, self.cfg.exploration_anneal_gens)
        t = g / float(self.cfg.exploration_anneal_gens)
        return (1.0 - t) * self.cfg.exploration_scale_start + t * self.cfg.exploration_scale_end

    # ---------- sampling ----------

    def sample_population(self, gen: int):
        # If not enough archive, keep broad exploration
        if len(self.archive_fitness) < self.cfg.min_archive_to_fit:
            return self.sample_initial_population()

        mu, Sigma = self.fit_full_cov_from_elites()
        scale = self.exploration_scale(gen)
        Sigma_scaled = (scale * scale) * Sigma

        # Cholesky for sampling: Sigma = L L^T
        # If Sigma isn't numerically PD, add more diagonal until it is.
        jitter = 0.0
        L = None
        for _ in range(8):
            try:
                L = np.linalg.cholesky(Sigma_scaled + np.eye(self.param_dim) * jitter)
                break
            except np.linalg.LinAlgError:
                jitter = max(1e-12, 10.0 * (jitter if jitter > 0 else 1e-12))
        if L is None:
            # Last resort: fall back to diagonal from eigenvalues
            lam, Q = np.linalg.eigh(Sigma_scaled)
            lam = np.maximum(lam, self.cfg.eigen_floor)
            # Sample via eigen basis: mu + Q diag(sqrt(lam)) z
            sqrt_lam = np.sqrt(lam)
            pop = []
            for _ in range(self.cfg.population_size):
                z = self.rng.normal(0.0, 1.0, size=self.param_dim)
                pop.append(mu + (Q * sqrt_lam) @ z)
            return pop

        pop = []
        for _ in range(self.cfg.population_size):
            z = self.rng.normal(0.0, 1.0, size=self.param_dim)
            pop.append(mu + L @ z)
        return pop


# ============================================================
# Main training loop
# ============================================================

def main():
    cfg = EDAConfig()
    rng = np.random.default_rng(cfg.seed)

    # Determine obs/act dims from a dummy env instance
    env_builder = SolarAntsEnv.demo_solar_system

    dummy_env = SolarAntsEnv(
        system_builder=env_builder,
        n_substeps=cfg.n_substeps,
        max_steps=cfg.max_steps,
    )
    obs_dim = int(dummy_env.observation_space.shape[0])
    act_dim = int(dummy_env.action_space.shape[0])
    dummy_env.close()

    d = policy_param_dim(obs_dim, act_dim, cfg.hidden_dim)
    eda = GlobalArchiveFullCovEDA(param_dim=d, rng=rng, cfg=cfg)

    # Init population
    population = eda.sample_initial_population()

    # Multiprocessing context (Linux)
    ctx = mp.get_context("fork")
    n_workers = cfg.n_workers

    # Output dir / files
    out_dir = "eda_runs"
    os.makedirs(out_dir, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    best_path = os.path.join(out_dir, f"best_params_fullcov_{run_tag}.npy")
    log_path = os.path.join(out_dir, f"log_fullcov_{run_tag}.jsonl")

    print(f"obs_dim={obs_dim} act_dim={act_dim} hidden_dim={cfg.hidden_dim} param_dim={d}")
    print(f"population={cfg.population_size} elites(all-time)={cfg.elite_size} archive_max={cfg.archive_max_size}")
    print(f"eigen_floor={cfg.eigen_floor} expl_scale={cfg.exploration_scale_start}->{cfg.exploration_scale_end} over {cfg.exploration_anneal_gens} gens")
    print(f"Logging to: {log_path}")
    print(f"Best params will be saved to: {best_path}")

    def render_progress(gen_idx: int, done: int, total: int, bar_width: int = 30):
        if total <= 0:
            return
        filled = int(round(bar_width * done / float(total)))
        bar = "#" * filled + "-" * (bar_width - filled)
        msg = f"\rGen {gen_idx:04d} eval [{bar}] {done}/{total}"
        sys.stdout.write(msg)
        sys.stdout.flush()

    with ctx.Pool(processes=n_workers) as pool, open(log_path, "w", encoding="utf-8") as logf:
        best_seen = -np.inf

        for gen in range(cfg.generations):
            # Parallel eval
            base_seed = cfg.seed + 10_000 * gen

            tasks = [
                (
                    i,
                    params,
                    env_builder,
                    cfg.hidden_dim,
                    cfg.eval_episodes,
                    base_seed,
                    cfg.max_steps,
                    cfg.n_substeps,
                )
                for i, params in enumerate(population)
            ]

            fitness = np.empty(len(population), dtype=np.float64)
            done = 0
            render_progress(gen, done, len(population))
            for idx, fit in pool.imap_unordered(evaluate_policy_worker, tasks, chunksize=1):
                fitness[idx] = fit
                done += 1
                render_progress(gen, done, len(population))
            sys.stdout.write("\n")
            sys.stdout.flush()

            # Update archive with every evaluated individual
            eda.add_to_archive(population, fitness.tolist())

            # Stats
            best_params, best_fit = eda.best_so_far()
            elite_params, elite_fit = eda.elites()
            elite_fit_arr = np.asarray(elite_fit, dtype=np.float64) if elite_fit else np.array([np.nan])

            gen_best = float(np.max(fitness))
            gen_mean = float(np.mean(fitness))
            elite_mean = float(np.mean(elite_fit_arr)) if elite_fit else float("nan")
            scale = float(eda.exploration_scale(gen))

            # Optional: diversity proxy among elites (mean std over dims)
            elite_std_mean = float(np.mean(np.std(np.stack(elite_params, axis=0), axis=0))) if elite_params else float("nan")

            print(
                f"Gen {gen:04d} | "
                f"gen_best: {gen_best:8.3f} | gen_mean: {gen_mean:8.3f} | "
                f"archive_best: {float(best_fit):8.3f} | elite_mean(all-time): {elite_mean:8.3f} | "
                f"elite_std_mean: {elite_std_mean:9.6f} | "
                f"expl_scale: {scale:5.2f} | archive: {len(eda.archive_fitness):4d}"
            )

            # Save best params if improved (or periodically)
            if float(best_fit) > best_seen:
                best_seen = float(best_fit)
                np.save(best_path, best_params)

            if gen % 10 == 0:
                np.save(best_path, best_params)

            # Log JSONL
            rec = {
                "gen": gen,
                "gen_best": gen_best,
                "gen_mean": gen_mean,
                "archive_best": float(best_fit),
                "elite_mean_all_time": elite_mean,
                "elite_std_mean": elite_std_mean,
                "exploration_scale": scale,
                "archive_size": int(len(eda.archive_fitness)),
                "timestamp": time.time(),
            }
            logf.write(json.dumps(rec) + "\n")
            logf.flush()

            # Sample next generation from full-cov Gaussian fit on global elites
            population = eda.sample_population(gen + 1)

    print("Done.")
    print(f"Best params saved at: {best_path}")
    print(f"Logs saved at: {log_path}")


if __name__ == "__main__":
    main()
