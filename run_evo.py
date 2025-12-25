from evo.evo import Evo
from evo.task import SolarAntsTask


evo = Evo(
    task=SolarAntsTask(n_trials=3),
    population_size=10,
    elite_k=2,
    model_lr=1e-3,
    dataset_max=100_000,
    seed=42,
)

evo.initialize_population()

evo.step_generation()