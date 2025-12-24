import numpy as np
import time 
import random

from solarants import System
from viewer import Viewer


def build_system():
    typical_orbital_radius = 1000.0
    typical_steps_per_orbit = 100000.0
    delta_time = 0.01

    sun_mass = 1e4
    planet_mass = 10.0

    typical_orbital_period = typical_steps_per_orbit * delta_time
    typical_orbital_velocity = 2.0 * np.pi * typical_orbital_radius / typical_orbital_period
    g_const = (typical_orbital_velocity ** 2) * typical_orbital_radius / (sun_mass + planet_mass)

    system = System(
        G=g_const,
        deltaTime=delta_time,
    )

    system.addCelestial(
        "sun",
        (0,0),  # position
        (0,0),  # velocity
        sun_mass,
        1.0,    # density
        True   # emits gravity
    )

    system.splitCelestial(
        "sun",
        "planet1",
        planet_mass,
        1.0,    # density
        True,   # emits gravity
        typical_orbital_radius,
        0.0,    # initial angle
        1.0     # ellipsity
    )

    # add a moon to planet1
    system.splitCelestial(
        "planet1",
        "moon1",
        0.01,   # mass
        1.0,    # density
        True,   # emits gravity
        10.0,  # orbital radius
        0.0,    # initial angle
        1.0     # ellipsity
    )
    
    #  Add a second planet
    system.splitCelestial(
        "sun",
        "planet2",
        planet_mass,
        0.25,    # density
        True,   # emits gravity
        typical_orbital_radius * 3.0,
        np.pi / 4.0,    # initial angle
        1.2,     # ellipsity
        True
    )

    #add resources around the twin planets (farther out)
    for i in range(30):
        system.addResourceInOrbit(
            "planet2",
            0.0001,  # mass
            1.0,    # density
            25,
            np.random.uniform(0.0, 2.0 * np.pi),    # initial angle
            np.random.uniform(0.8, 1.2),     # ellipsity
            True     # prograde
        )

    #split planet2 into two twin planets
    system.splitCelestial(
        "planet2",
        "planet2.5", #twin planets
        planet_mass * 0.5, #
        0.25,    # density
        True,   # emits gravity
        15.0,  # orbital radius
        np.pi / 2.0,    # initial angle
        1.0,     # ellipsity
        False     # Retrograde
    )

    system.addAgent(
        "planet1",
        0.001,  # mass
        0.05,    # radius
        3.14 * 0.5,    # initial angle
        1.0, #collection radius
        23.0 * 0.001, # max control force: mass* surface gravity of planet1
        0.001 # cargo capacity
    )

    #add some resources in close orbit around planet1
    for i in range(20):
        system.addResourceInOrbit(
            "planet1",
            0.0001,  # mass
            1.0,    # density
            5.0 + np.random.uniform(-2.0, 2.0),  # orbital radius
            np.random.uniform(0.0, 2.0 * np.pi),    # initial angle
            np.random.uniform(0.5, 1.5),     # ellipsity
            True     # prograde
        )

    #add some resources on the surface of moon1
    for i in range(30):
        system.addResourceOnSurface(
            "moon1",
            0.0001,  # mass
            1.0,    # density
            i * (2.0 * np.pi / 30.0)    # initial angle
        )

    #add some resources in orbit around sun
    for i in range(50):
        system.addResourceInOrbit(
            "sun",
            0.0001,  # mass
            1.0,    # density
            1500.0 + np.random.uniform(-300.0, 300.0),  # orbital radius
            np.random.uniform(0.0, 2.0 * np.pi),    # initial angle
            np.random.uniform(0.7, 1.3),     # ellipsity
            True     # prograde
        )


    system.initialize()
    return system


def main():
    system = build_system()

    def control_handle():
        agent = system.agents[0]
        agent.applyControlForce((0.0, 0.8))  # initial zero control force
        state = agent.getSensorReadings()  # initial sensor readings
        rew = agent.computeReward()  # initial reward
    

    viewer = Viewer(system,control_handle, substeps_per_frame=1, initial_zoom=0.25)
    viewer.run()


if __name__ == "__main__":
    main()
