import torch
import numpy as np
from .solar_ants_env import SolarAntsEnv

# -----------------------
# Notes on the task API:
# -----------------------
# Your task object should implement:
#   - genome_length() -> int
#   - random_genome(length) -> np.ndarray shape [length]
#   - evaluate_genome(genome: np.ndarray, generation_seed=None, individual_seed=None) -> float
# -----------------------

#a gym env for the solar ants that we have to adapt to the task api
#env = SolarAntsEnv(
#    system_builder=SolarAntsEnv.system_01,
#    n_substeps=2,
#    max_steps=10000)

class SolarAntsTask:
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.policy = Policy(self.observation_length(), self.action_length())
        print(f"Initialized SolarAntsTask with policy having {self.policy.num_params()} parameters.")

    def genome_length(self) -> int:
        return self.policy.num_params()
    
    def random_genome(self) -> np.ndarray:
        return self.policy.generate_random_params()

    def evaluate_genome(self, genome: np.ndarray, generation_seed=None, individual_seed=None) -> float:
        np.random.seed(generation_seed)
        total_reward = 0.0
        self.policy.set_params(genome)
        for trial in range(self.n_trials):
            self.policy.reset()
            env = SolarAntsEnv(
                system_builder=SolarAntsEnv.system_01,
                n_substeps=2,
                max_steps=10000)
            obs, info = env.reset(seed=individual_seed)
            done = False
            trial_reward = 0.0
            while not done:
                action = self.policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                trial_reward += reward
            total_reward += trial_reward
        
        avg_reward = total_reward / self.n_trials
        return float(avg_reward)

    def observation_length(self) -> int:
        env = SolarAntsEnv(
            system_builder=SolarAntsEnv.system_01,
            n_substeps=2,
            max_steps=10000)    
        return env.observation_space.shape[0]

    def action_length(self) -> int:
        env = SolarAntsEnv(
            system_builder=SolarAntsEnv.system_01,
            n_substeps=2,
            max_steps=10000)
        return env.action_space.shape[0]

#a rnn which maps 
class Policy(torch.nn.Module):
    def __init__(self, obs_length, action_length, hidden_size=8):
        super(Policy, self).__init__()
        self.obs_length = obs_length
        self.action_length = action_length
        self.hidden_size = hidden_size
        self.state = torch.zeros(hidden_size)

        self.l1 = torch.nn.Linear(obs_length, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size * 2, hidden_size) #the recurrent connection
        self.l3 = torch.nn.Linear(hidden_size, action_length)
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            h1 = self.silu(self.l1(x))
            h2 = self.silu(self.l2(torch.cat([h1, self.state], dim=-1)))
            self.state = h2
            out = self.l3(h2)
            return out.numpy()

    def act(self, observation):
        return self.forward(observation)

    def num_params(self):
        total = 0
        for param in self.parameters():
            total += param.numel()
        return total

    def generate_random_params(self):
        genome = np.zeros(self.num_params(), dtype=np.float32)
        current_index = 0
        for param in self.parameters():
            param_length = param.numel()
            random_values = np.random.uniform(-1.0, 1.0, size=param_length).astype(np.float32)
            genome[current_index:current_index + param_length] = random_values
            current_index += param_length
        return genome

    def set_params(self, genome: np.ndarray):
        # Set the parameters of the policy from the genome
        current_index = 0
        for param in self.parameters():
            param_length = param.numel()
            new_param = genome[current_index:current_index + param_length]
            new_param = new_param.reshape(param.size())
            param.data = torch.from_numpy(new_param).float()
            current_index += param_length

    def reset(self):
        self.state = torch.zeros(self.hidden_size)
        