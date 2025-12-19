from solarants import System
import cv2
import numpy as np

# ----------------------------
# Simulation setup
# ----------------------------

typical_orbital_radius = 1000.0
typical_steps_per_orbit = 10000.0
delta_time = 0.01
substeps_per_frame = 100

sun_mass = 1e4
planet_mass = 5.0

typical_orbital_period = typical_steps_per_orbit * delta_time
typical_orbital_velocity = 2.0 * np.pi * typical_orbital_radius / typical_orbital_period
G = (typical_orbital_velocity ** 2) * typical_orbital_radius / (sun_mass + planet_mass)

print(f"Using G = {G}")

system = System(
    sunMass=sun_mass,
    sunDensity=1.0,
    G=G,
    deltaTime=delta_time
)

system.addBody(
    planet_mass,
    True,   # emits gravity
    1.0,    # density
    typical_orbital_radius,
    0.0,    # initial angle
    1.1     # ellipticity
)

system.initialize()

# ----------------------------
# Visualization parameters
# ----------------------------

canvas_size = (800, 800)
canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
canvas_scale = 1.0 / 4.0
steps = 1000

# ----------------------------
# Drawing helpers
# ----------------------------

def sim_to_canvas(x, y):
    cx = int(x * canvas_scale + canvas_size[0] // 2)
    cy = int(y * canvas_scale + canvas_size[1] // 2)
    return cx, cy


def draw_body(canvas, prop):
    x, y = sim_to_canvas(prop["position_x"], prop["position_y"])
    radius = max(2, int(prop["radius"] * canvas_scale))
    cv2.circle(canvas, (y, x), radius, (0, 255, 0), -1)

# ----------------------------
# Pre-step to settle orbit
# ----------------------------

for _ in range(steps):
    system.step()

# ----------------------------
# Main loop
# ----------------------------

for step in range(steps):
    canvas[:] = 0

    for _ in range(substeps_per_frame):
        system.step()

    props = system.getAllBodyProperties()

    for prop in props:
        draw_body(canvas, prop)

    cv2.imshow("Solar Ants Simulation (Bodies Only)", canvas)

    if cv2.waitKey(100) & 0xFF == 27:
        break

cv2.destroyAllWindows()
