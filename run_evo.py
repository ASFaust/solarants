from evo.evo import Evo
from evo.task import SolarAntsTask


evo = Evo(
    task=SolarAntsTask(n_trials=3),
    population_size=1000,
    elite_k=100,
    model_lr=1e-3,
    dataset_max=100_000,
    seed=42,
)

evo.initialize_population()
for i in range(500):
    print(f"Generation {i}")
    evo.step_generation()