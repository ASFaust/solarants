import numpy as np
import time 

from solarants import System
from viewer import Viewer


def build_system():
    typical_orbital_radius = 1000.0
    typical_steps_per_orbit = 10000.0
    delta_time = 0.01

    sun_mass = 1e4
    planet_mass = 10.0

    typical_orbital_period = typical_steps_per_orbit * delta_time
    typical_orbital_velocity = 2.0 * np.pi * typical_orbital_radius / typical_orbital_period
    g_const = (typical_orbital_velocity ** 2) * typical_orbital_radius / (sun_mass + planet_mass)

    system = System(
        sunMass=sun_mass,
        sunDensity=0.5,
        G=g_const,
        deltaTime=delta_time,
    )

    system.addBody(
        "home",
        planet_mass,
        1.0,    # density
        True,   # emits gravity
        typical_orbital_radius,
        0.0,    # initial angle
        1.0,    # ellipticity. 1.0 = circular orbit
    )

    system.addMoon(
        "moon1",
        "home",
        2.0,    # mass
        True,   # emits gravity
        1.0,    # density
        20.0,  # orbital radius
        0.0,    # initial angle
        1.0,    # ellipticity
    )

    system.addMoon(
        "moonmoon",
        "moon1",
        0.1,    # mass
        True,   # emits gravity
        1.0,    # density
        4.0,    # orbital radius
        0.0,    # initial angle
        1.0,    # ellipticity
    )

    system.initialize()
    return system


def main():
    system = build_system()
    viewer = Viewer(system, substeps_per_frame=1, initial_zoom=0.25)
    viewer.run()


if __name__ == "__main__":
    main()
