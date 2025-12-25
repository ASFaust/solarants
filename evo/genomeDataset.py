import numpy as np


class GenomeDataset:
    def __init__(self, genome_length: int, max_size: int = 200_000):
        self.genome_length = genome_length
        self.max_size = max_size
        self.genomes = []   # list[np.ndarray]
        self.fitnesses = [] # list[float]

    def __len__(self):
        return len(self.genomes)

    def add_sample(self, genome: np.ndarray, fitness: float):
        g = np.asarray(genome, dtype=np.float32).copy()
        if g.shape != (self.genome_length,):
            raise ValueError(f"genome has shape {g.shape}, expected {(self.genome_length,)}")
        self.genomes.append(g)
        self.fitnesses.append(float(fitness))

        # simple FIFO cap
        if len(self.genomes) > self.max_size:
            overflow = len(self.genomes) - self.max_size
            del self.genomes[:overflow]
            del self.fitnesses[:overflow]

    def sample_pairs(self, batch_size: int):
        """Return (g1, g2, y) where y=+1 means g1 should rank above g2."""
        n = len(self.genomes)
        if n < 2:
            return None

        i1 = np.random.randint(0, n, size=batch_size)
        i2 = np.random.randint(0, n, size=batch_size)

        g1 = np.stack([self.genomes[i] for i in i1], axis=0)  # [B, D]
        g2 = np.stack([self.genomes[i] for i in i2], axis=0)  # [B, D]

        f1 = np.array([self.fitnesses[i] for i in i1], dtype=np.float32)
        f2 = np.array([self.fitnesses[i] for i in i2], dtype=np.float32)

        # ties: randomly break, or treat as 0-margin; here we random-break to keep gradients alive
        y = np.where(f1 > f2, 1.0, -1.0).astype(np.float32)
        ties = (f1 == f2)
        if ties.any():
            y[ties] = np.random.choice([-1.0, 1.0], size=int(ties.sum())).astype(np.float32)

        return g1, g2, y

    def pop_top_genomes(self, k: int):
        """Return top-k genomes and delete them from the dataset."""
        if k <= 0:
            return []

        # get indices of top-k fitnesses
        sorted_indices = np.argsort(self.fitnesses)[::-1]  # descending order
        topk_indices = sorted_indices[:k]

        topk_genomes = [self.genomes[i] for i in topk_indices]

        # remove them from dataset
        for index in sorted(topk_indices, reverse=True):
            del self.genomes[index]
            del self.fitnesses[index]

        return topk_genomes

    def save_best(self,path):
        """Save the best genome to a file."""
        if len(self.genomes) == 0:
            raise ValueError("No genomes in dataset to save.")

        best_index = np.argmax(self.fitnesses)
        best_genome = self.genomes[best_index]

        np.save(path, best_genome)