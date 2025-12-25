from evo.evo import Evo
from evo.task import SolarAntsTask
import numpy as np
import torch
import os

evo = Evo(
    task=SolarAntsTask(n_trials=5),
    population_size=1000,
    elite_k=100,
    model_lr=1e-3,
    dataset_max=100_000,
    seed=42,
)

os.makedirs("results", exist_ok=True)

evo.initialize_population()
for i in range(500):
    print(f"Generation {i}")
    evo.step_generation()
    evo.dataset.save_best(f"results/best_genome_gen_{i:03d}.npy")