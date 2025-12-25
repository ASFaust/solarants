import numpy as np
import random
import torch
import torch.nn.functional as F
import multiprocessing as mp
import numpy as np

from .genomeDataset import GenomeDataset

def _eval_one(args):
    task, genome, gen_seed, ind_seed = args
    fitness = task.evaluate_genome(genome, generation_seed=int(gen_seed), individual_seed=int(ind_seed))
    return genome, fitness

class Evo:
    def __init__(
        self,
        task,
        population_size=1000,
        elite_k=100,
        device=None,
        dataset_max=200_000,
        model_lr=1e-3,
        seed=0,
    ):
        self.task = task
        self.population_size = int(population_size)
        self.elite_k = int(elite_k)

        self.genome_length = int(task.genome_length())
        self.dataset = GenomeDataset(self.genome_length, max_size=dataset_max)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.population = self.initialize_population()

        self.model = self.build_model().to(self.device)
        self.model_opt = torch.optim.Adam(self.model.parameters(), lr=model_lr)

    def initialize_population(self):
        return [np.asarray(self.task.random_genome(), dtype=np.float32)
                for _ in range(self.population_size)]

    def evaluate_population(self, n_workers=None, chunksize=1):
        if n_workers is None:
            n_workers = mp.cpu_count()

        if n_workers > 30:
            print("Warning: clamping n_workers to 30 to avoid overload.")
            n_workers = 30

        fitnesses = np.empty((self.population_size,), dtype=np.float32)
        
        gen_seed = self.rng.integers(0, 2**31 - 1)

        # Important: zip task with genome so workers are stateless
        work = [(self.task, genome, gen_seed, self.rng.integers(0, 2**31 - 1))
                for genome in self.population]

        with mp.get_context("fork").Pool(processes=n_workers) as pool:
            for i, (genome, fit) in enumerate(
                pool.imap(_eval_one, work, chunksize=chunksize)
            ):
                fitnesses[i] = fit
                self.dataset.add_sample(genome, fit)
                print(f"\reval {i + 1}/{self.population_size}", end="", flush=True)

        print()

        return fitnesses


    def build_model(self):
        # simple MLP ranking model
        return torch.nn.Sequential(
            torch.nn.Linear(self.genome_length, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )

    def train_model(self, steps=1000, batch_size=128, margin=0.1):
        self.model.train()
        for step in range(1, steps + 1):
            sample = self.dataset.sample_pairs(batch_size)
            if sample is None:
                return
            g1, g2, y = sample

            g1t = torch.from_numpy(g1).to(self.device)
            g2t = torch.from_numpy(g2).to(self.device)
            yt  = torch.from_numpy(y).to(self.device)

            p1 = self.model(g1t).squeeze(-1)  # [B]
            p2 = self.model(g2t).squeeze(-1)  # [B]

            # want p1 > p2 when y=+1, else p1 < p2
            loss = F.margin_ranking_loss(p1, p2, yt, margin=margin)

            self.model_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.model_opt.step()
            print(f"\rtrain step {step}/{steps} loss {loss.item():.4f}", end="", flush=True)

    @torch.no_grad()
    def model_score(self, genome: np.ndarray) -> float:
        self.model.eval()
        g = torch.from_numpy(np.asarray(genome, dtype=np.float32)).to(self.device)
        return float(self.model(g).item())

    def mutate_genome(self, genome, mutation_rate=0.1, sigma=0.1):
        g = np.asarray(genome, dtype=np.float32).copy()
        mask = self.rng.random(g.shape[0]) < mutation_rate
        g[mask] += self.rng.normal(0.0, sigma, size=int(mask.sum())).astype(np.float32)
        return g

    def optimize_genome_with_model(
        self,
        genome: np.ndarray,
        parent: np.ndarray,
        existing_pop: list,
        steps: int = 60,
        lr: float = 3e-2,
        trust_radius: float = 0.3,
        trust_lambda: float = 5.0,
        repel_radius: float = 0.1,
        repel_lambda: float = 2.0,
        hard_clip: bool = True,
    ):
        """
        Maximize model(genome) with:
          - trust region to parent (quadratic penalty, optional hard clip)
          - repulsion from already-chosen genomes (smooth barrier)
        """
        self.model.eval()

        parent_t = torch.from_numpy(np.asarray(parent, dtype=np.float32)).to(self.device)

        x = torch.tensor(np.asarray(genome, dtype=np.float32), device=self.device, requires_grad=True)
        opt = torch.optim.Adam([x], lr=lr)

        # Build a tensor of existing genomes once (and update outside if needed).
        # We'll pass in existing_pop incrementally; inside, we just snapshot.
        if len(existing_pop) > 0:
            others = torch.from_numpy(np.stack(existing_pop).astype(np.float32)).to(self.device)  # [N, D]
        else:
            others = None

        for _ in range(steps):
            pred = self.model(x).squeeze()

            # Trust region: penalize distance from parent
            dx = x - parent_t
            dist_to_parent = torch.norm(dx)
            trust_pen = trust_lambda * torch.clamp(dist_to_parent - trust_radius, min=0.0) ** 2

            # Repulsion: smooth barrier when closer than repel_radius to any existing genome
            repel_pen = torch.tensor(0.0, device=self.device)
            if others is not None:
                # distances to all others
                d = torch.norm(others - x.unsqueeze(0), dim=1)  # [N]
                # barrier activates when d < repel_radius
                repel_pen = repel_lambda * torch.mean(torch.clamp(repel_radius - d, min=0.0) ** 2) 

            # We maximize pred, so minimize (-pred + penalties)
            loss = -pred + trust_pen + repel_pen

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if hard_clip:
                # hard projection back into trust ball around parent
                with torch.no_grad():
                    dx = x - parent_t
                    norm = torch.norm(dx)
                    if norm > trust_radius:
                        x.copy_(parent_t + dx / (norm + 1e-8) * trust_radius)

        return x.detach().cpu().numpy().astype(np.float32)

    def step_generation(
        self,
        model_train_steps=1200,
        mutation_rate=0.1,
        mutation_sigma=0.1,
        repel_dist=0.1,
    ):
        # 1) Evaluate and store data
        print("evaluating population...")
        fitnesses = self.evaluate_population()
        best = float(np.max(fitnesses))
        mean = float(np.mean(fitnesses))
        print(f"eval | best {best:.3f} | mean {mean:.3f} | dataset {len(self.dataset)}")

        # 2) Train ranking model on accumulated dataset
        print("training ranking model...")
        self.train_model(steps=model_train_steps, batch_size=128, margin=0.1)

        # 3) Elite selection (optionally re-evaluate to reduce noise)
        elites = self.dataset.pop_top_genomes(self.elite_k)

        print("sampling new population...")
        # 4) Rebuild population: mutate elites + model-guided ascent + repulsion
        new_pop = elites  # keep elites (that's the reevaluation process)
        while len(new_pop) < self.population_size:
            print(f"\r  Sampling individual {len(new_pop)+1}/{self.population_size}", end="", flush=True)
            parent = random.choice(elites)
            child = self.mutate_genome(parent, mutation_rate=mutation_rate, sigma=mutation_sigma)
            child = self.optimize_genome_with_model(
                child, parent, new_pop,
                steps=60,
                lr=3e-2,
                trust_radius=0.35,
                trust_lambda=6.0,
                repel_radius=repel_dist,
                repel_lambda=0.25,
                hard_clip=True,
            )
            new_pop.append(child)

        self.population = new_pop

        return best, mean


# -----------------------
# Notes on the task API:
# -----------------------
# Your task object should implement:
#   - genome_length() -> int
#   - random_genome(length) -> np.ndarray shape [length]
#   - evaluate_genome(genome: np.ndarray, generation_seed=None, individual_seed=None) -> float
# -----------------------
