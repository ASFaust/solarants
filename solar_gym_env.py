import gymnasium as gym
from gymnasium import spaces
import numpy as np

from solarants import System


class SolarAntsEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        system_builder,
        n_substeps=10,
        max_steps=1000,
        render_mode=None,
    ):
        """
        system_builder: callable that returns a fully initialized System
        """
        super().__init__()

        self.system_builder = system_builder
        self.n_substeps = n_substeps
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.system = None
        self.agent = None
        self.step_count = 0

        # --- action: 2D control force in [-1, 1] ---
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # --- observation: fixed 18-dim vector ---
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(18,),
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Static world definitions
    # ------------------------------------------------------------------

    @staticmethod
    def demo_solar_system():
        """
        Reuses your existing demo setup verbatim.
        """
        import numpy as np

        typical_orbital_radius = 1000.0
        typical_steps_per_orbit = 100000.0
        delta_time = 0.01

        sun_mass = 1e4
        planet_mass = 10.0

        typical_orbital_period = typical_steps_per_orbit * delta_time
        typical_orbital_velocity = (
            2.0 * np.pi * typical_orbital_radius / typical_orbital_period
        )
        g_const = (
            typical_orbital_velocity**2
            * typical_orbital_radius
            / (sun_mass + planet_mass)
        )

        system = System(G=g_const, deltaTime=delta_time)

        system.addCelestial(
            "sun",
            (0, 0),
            (0, 0),
            sun_mass,
            1.0,
            True,
        )

        system.splitCelestial(
            "sun",
            "planet1",
            planet_mass,
            1.0,
            True,
            typical_orbital_radius,
            0.0,
            1.0,
        )

        system.splitCelestial(
            "planet1",
            "moon1",
            0.01,
            1.0,
            True,
            10.0,
            0.0,
            1.0,
        )

        system.splitCelestial(
            "sun",
            "planet2",
            planet_mass,
            0.25,
            True,
            typical_orbital_radius * 3.0,
            np.pi / 4.0,
            1.2,
            True,
        )

        for _ in range(30):
            system.addResourceInOrbit(
                "planet2",
                0.0001,
                1.0,
                25,
                np.random.uniform(0.0, 2.0 * np.pi),
                np.random.uniform(0.8, 1.2),
                True,
            )

        system.splitCelestial(
            "planet2",
            "planet2.5",
            planet_mass * 0.5,
            0.25,
            True,
            15.0,
            np.pi / 2.0,
            1.0,
            False,
        )

        system.addAgent(
            "planet1",
            0.001,
            0.05,
            np.pi * 0.5,
            1.0,
            23.0 * 0.001,
            0.001,
        )

        for _ in range(20):
            system.addResourceInOrbit(
                "planet1",
                0.0001,
                1.0,
                5.0 + np.random.uniform(-2.0, 2.0),
                np.random.uniform(0.0, 2.0 * np.pi),
                np.random.uniform(0.5, 1.5),
                True,
            )

        for i in range(30):
            system.addResourceOnSurface(
                "moon1",
                0.0001,
                1.0,
                i * (2.0 * np.pi / 30.0),
            )

        for _ in range(50):
            system.addResourceInOrbit(
                "sun",
                0.0001,
                1.0,
                1500.0 + np.random.uniform(-300.0, 300.0),
                np.random.uniform(0.0, 2.0 * np.pi),
                np.random.uniform(0.7, 1.3),
                True,
            )

        system.initialize()
        return system

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.system = self.system_builder()
        self.agent = self.system.agents[0]
        self.step_count = 0

        obs = np.asarray(self.agent.getSensorReadings(), dtype=np.float64)
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float64)

        for _ in range(self.n_substeps):
            self.agent.applyControlForce(action)
            self.system.step()

        obs = np.asarray(self.agent.getSensorReadings(), dtype=np.float64)
        reward = self.agent.computeReward()

        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        # Rendering intentionally delegated to Viewer
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = SolarAntsEnv(system_builder=SolarAntsEnv.demo_solar_system)
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Episode finished with total reward: {total_reward}")