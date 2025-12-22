import numpy as np
import time 

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

    system.addBody(
        "sun",
        (0,0),  # position
        (0,0),  # velocity
        sun_mass,
        1.0,    # density
        True   # emits gravity
    )

    system.splitBody(
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
    system.splitBody(
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
    system.splitBody(
        "sun",
        "planet2",
        planet_mass,
        1.0,    # density
        True,   # emits gravity
        typical_orbital_radius * 3.0,
        np.pi / 4.0,    # initial angle
        1.2,     # ellipsity
        False   # retrograde orbit
    )
    # and a small moon to planet2
    system.splitBody(
        "planet2",
        "moon2",
        0.01,   # mass
        1.0,    # density
        True,   # emits gravity
        15.0,  # orbital radius
        np.pi / 2.0,    # initial angle
        0.8     # ellipsity
    )

    """
            .def("addAgent", &System::addAgent,
            pybind11::arg("hostBodyName"),
            pybind11::arg("agentName"),
            pybind11::arg("mass"),
            pybind11::arg("radius"),
            pybind11::arg("initial_angle"),
            pybind11::arg("emitGravity"))
        """

    system.addAgent(
        "planet1",
        "agent1",
        0.001,  # mass
        0.05,    # radius
        3.14 * 0.5,    # initial angle
        False    # emits gravity
    )

    system.initialize()
    return system


def main():
    system = build_system()
    viewer = Viewer(system, substeps_per_frame=1, initial_zoom=0.25)
    viewer.run()


if __name__ == "__main__":
    main()
