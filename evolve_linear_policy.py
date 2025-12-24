import numpy as np
import multiprocessing as mp

from solar_ants_env import SolarAntsEnv  # adjust import if needed


# ------------------------------------------------------------
# Linear policy
# ------------------------------------------------------------

class LinearPolicy:
    def __init__(self, obs_dim, act_dim, params=None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        if params is None:
            self.W = np.random.randn(act_dim, obs_dim) * 0.01
            self.b = np.zeros(act_dim)
        else:
            self.unpack(params)

    def act(self, obs):
        return np.tanh(self.W @ obs + self.b)

    def pack(self):
        return np.concatenate([self.W.flatten(), self.b])

    def unpack(self, params):
        w_size = self.act_dim * self.obs_dim
        self.W = params[:w_size].reshape(self.act_dim, self.obs_dim)
        self.b = params[w_size:w_size + self.act_dim]


# ------------------------------------------------------------
# Fitness evaluation (worker-safe)
# ------------------------------------------------------------

def evaluate_policy_worker(args):
    params, env_builder, n_episodes, seed = args

    total_reward = 0.0

    for ep in range(n_episodes):
        env = SolarAntsEnv(system_builder=env_builder)
        obs, _ = env.reset(seed=seed + ep)

        policy = LinearPolicy(
            obs_dim=env.observation_space.shape[0],
            act_dim=env.action_space.shape[0],
            params=params,
        )

        done = False
        ep_reward = 0.0

        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        env.close()

    return total_reward / n_episodes


# ------------------------------------------------------------
# Evolutionary Strategy (parallel)
# ------------------------------------------------------------

def run_evolution(
    env_builder,
    population_size=64,
    elite_fraction=0.2,
    mutation_sigma=0.05,
    generations=100,
    eval_episodes=1,
    n_workers=None,
    seed=0,
):
    rng = np.random.default_rng(seed)

    dummy_env = SolarAntsEnv(system_builder=env_builder)
    obs_dim = dummy_env.observation_space.shape[0]
    act_dim = dummy_env.action_space.shape[0]
    dummy_env.close()

    param_dim = act_dim * obs_dim + act_dim
    n_elite = max(1, int(population_size * elite_fraction))

    population = [
        rng.normal(0.0, 0.1, size=param_dim)
        for _ in range(population_size)
    ]

    ctx = mp.get_context("fork")  # important on Linux
    with ctx.Pool(processes=n_workers) as pool:
        for gen in range(generations):
            tasks = [
                (params, env_builder, eval_episodes, seed + gen)
                for params in population
            ]

            fitness = pool.map(evaluate_policy_worker, tasks)
            fitness = np.asarray(fitness)

            order = np.argsort(fitness)[::-1]
            population = [population[i] for i in order]
            fitness = fitness[order]

            elites = population[:n_elite]

            print(
                f"Gen {gen:04d} | "
                f"best: {fitness[0]:8.3f} | "
                f"mean: {fitness.mean():8.3f} | "
                f"elite mean: {fitness[:n_elite].mean():8.3f}"
            )

            # Refill population
            new_population = elites.copy()
            while len(new_population) < population_size:
                parent = elites[rng.integers(0, n_elite)]
                child = parent + rng.normal(
                    0.0, mutation_sigma, size=param_dim
                )
                new_population.append(child)

            population = new_population

    return population[0]


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    best_params = run_evolution(
        env_builder=SolarAntsEnv.demo_solar_system,
        population_size=64,
        elite_fraction=0.25,
        mutation_sigma=0.05,
        generations=50,
        eval_episodes=1,
        n_workers=None,  # None = use all cores
        seed=42,
    )

    print("Evolution finished.")
