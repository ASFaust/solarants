#include "System.h"
#include "Body.h"
#include "Vec2.h"
#include "Collisions.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

System::System(double G_, double deltaTime_){
    G = G_;
    deltaTime = deltaTime_;
}

Celestial* System::getCelestialByName(const string& name) const {
    for (Celestial* cel : celestials) {
        if (cel->name == name) {
            return cel;
        }
    }
    return nullptr; // Not found
}

void System::addCelestial(
        const string& name,
        Vec2 position,
        Vec2 velocity,
        double mass,
        double density,
        bool emitGravity
    ) {
    if(getCelestialByName(name)){
        throw std::runtime_error("Body name already exists: " + name);
    }
    
    Celestial* cel = new Celestial(this, name, position, velocity, mass, density);

    celestials.push_back(cel);

    bodies.push_back(cel);
    if(emitGravity){
        gravityGeneratingBodies.push_back(cel);
    }
}

void System::splitCelestial(
    const string& hostName,
    const string& splitCelestialName,
    double mass,
    double density,
    bool emitGravity,
    double orbital_radius,   // separation distance host <-> child
    double initial_angle,    // direction of separation
    double ellipsity,        // 1.0 = circular, <1 elliptical (apo/peri assumed)
    bool prograde
) {
    // ------------------------------------------------------------------
    // Find host
    // ------------------------------------------------------------------
    Celestial* host = getCelestialByName(hostName);
    if (!host) {
        throw std::runtime_error("Host not found: " + hostName);
    }
    if (getCelestialByName(splitCelestialName)) {
        throw std::runtime_error("Name already exists: " + splitCelestialName);
    }

    double M_child = mass;
    double M_host_old = host->getMass();

    if (M_host_old <= M_child) {
        throw std::runtime_error("Host does not have enough mass to split.");
    }

    // ------------------------------------------------------------------
    // Mass update
    // ------------------------------------------------------------------
    double M_host = M_host_old - M_child;
    double M_total = M_host + M_child;
    host->setMass(M_host);

    // ------------------------------------------------------------------
    // Geometry
    // ------------------------------------------------------------------
    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 tangent_dir(-radial_dir.y, radial_dir.x); // 90° CCW

    Vec2 r_rel = radial_dir * orbital_radius;      // host -> child

    // ------------------------------------------------------------------
    // Orbital mechanics (two-body relative orbit)
    // ------------------------------------------------------------------
    double a = orbital_radius / ellipsity;          // assume apo/periapsis
    double mu = G * M_total;                         // <-- CRUCIAL FIX

    double v_mag = std::sqrt(mu * (2.0 / orbital_radius - 1.0 / a));
    Vec2 v_rel = tangent_dir * v_mag;

    if (!prograde) {
        v_rel *= -1.0;
    }

    // ------------------------------------------------------------------
    // Save original host state
    // ------------------------------------------------------------------
    Vec2 old_pos = host->position;
    Vec2 old_vel = host->velocity;

    // ------------------------------------------------------------------
    // Barycentric split (position + velocity)
    // ------------------------------------------------------------------
    host->position  = old_pos - r_rel * (M_child / M_total);
    Vec2 child_pos  = old_pos + r_rel * (M_host  / M_total);

    host->velocity  = old_vel - v_rel * (M_child / M_total);
    Vec2 child_vel  = old_vel + v_rel * (M_host  / M_total);

    // ------------------------------------------------------------------
    // Create child
    // ------------------------------------------------------------------
    Celestial* child = new Celestial(
        this,
        splitCelestialName,
        child_pos,
        child_vel,
        M_child,
        density
    );

    child->parent = host;

    bodies.push_back(child);
    celestials.push_back(child);
    if (emitGravity) {
        gravityGeneratingBodies.push_back(child);
    }
}


void System::addAgent(
    const string& hostName,
    double mass,
    double radius,
    double initial_angle, 
    double collectionRadius,
    double maxControlForce,
    double cargoCapacity
) {
    Celestial* home = getCelestialByName(hostName);
    if (!home) {
        throw std::runtime_error("Host not found: " + hostName);
    }

    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 position = home->position + radial_dir * (home->getRadius() + radius);
    Vec2 velocity = home->velocity; // start with host body's velocity

    double density = (3.0 * mass) / (4.0 * M_PI * radius * radius * radius);

    Agent* agent = new Agent(
        this, position, velocity, mass, density, home, 
        collectionRadius, maxControlForce, cargoCapacity);

    //cast to Body* and add to bodies
    bodies.push_back(static_cast<Body*>(agent));
    agents.push_back(agent);
}

void System::addResourceInOrbit(
    const string& hostName,
    double mass,
    double density,
    double orbital_radius,
    double initial_angle,
    double ellipsity,
    bool prograde
) {
    Celestial* host = getCelestialByName(hostName);
    if (!host) {
        throw std::runtime_error("Host not found: " + hostName);
    }

    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 tangent_dir(-radial_dir.y, radial_dir.x); // 90° CCW
    Vec2 position = host->position + radial_dir * orbital_radius;

    double a = orbital_radius / ellipsity;
    double mu = G * host->getMass();
    double orbital_speed = std::sqrt(mu * (2.0 / orbital_radius - 1.0 / a));

    Vec2 v_rel = tangent_dir * orbital_speed;
    if (!prograde) {
        v_rel = v_rel * -1.0;
    }
    Vec2 velocity = host->velocity + v_rel;

    Resource* res = new Resource(this, position, velocity, mass, density);

    resources.push_back(res);
}

void System::addResourceOnSurface(
    const string& hostName,
    double mass,
    double density,
    double initial_angle 
) {
    Celestial* host = getCelestialByName(hostName);
    if (!host) {
        throw std::runtime_error("Host not found: " + hostName);
    }

    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 position = host->position + radial_dir * host->getRadius();
    Vec2 velocity = host->velocity; // start with host body's velocity

    Resource* res = new Resource(this, position, velocity, mass, density);

    resources.push_back(res);
}   

void System::initialize(){
    //initialize all bodies for leapfrog integration
    for (Body* body : bodies) {
        Vec2 totalGravity = calculateGravity(body);
        body->velocity += totalGravity * 0.5 * deltaTime;
    }
}

void System::step(int n) {
    for (int i = 0; i < n; ++i) {   

        // 1. drift
        for (Body* body : bodies) {
            body->position += body->velocity * deltaTime;
        }
        for (Resource* res : resources) {
            res->position += res->velocity * deltaTime;
        }

        // 2. full kick
        for (Body* body : bodies) {
            body->velocity += calculateGravity(body) * deltaTime;
        }
        for (Resource* res : resources) {
            res->velocity += calculateGravity(res) * deltaTime;
        }

        // 3 control forces from agents
        for (Agent* agent : agents) {
            agent->velocity += agent->controlForce * agent->getInvMass() * deltaTime;
            agent->controlForce = Vec2(0.0, 0.0); // reset control force after applying
        }

        // 4. collision resolution
        resolveCollisions();


        // 5. compute agent logic
        computeAgentLogic();
    }
}

void System::computeAgentLogic() {
    // Placeholder for agent logic computation
    /*
    For each agent:
    * Check for nearby resources within collection radius
    * If resources are found, collect them (update cargo), and remove resource from system
    * If we are near the home celestial and have cargo, drop off cargo, set reward flag for amount of cargo dropped off
    */
    for (Agent* agent : agents) {

        for (auto it = resources.begin(); it != resources.end(); ) {
            Resource* res = *it;

            Vec2 toResource = res->position - agent->position;
            if (toResource.norm() <= agent->collectionRadius) {
                double mass = res->getMass();

                if (agent->currentCargo + mass <= agent->cargoCapacity) {
                    agent->currentCargo += mass;
                    agent->setMass(agent->getMass() + mass);

                    delete res;
                    it = resources.erase(it);  // safe, O(1)
                    continue;
                }
            }
            ++it;
        }

        Vec2 toHome = agent->home->position - agent->position;
        double distanceToHome = toHome.norm();
        double acceptableRadius = agent->home->getRadius() + agent->getRadius() + agent->collectionRadius;
        if (distanceToHome <= acceptableRadius && agent->currentCargo > 0.0) {
            // Drop off cargo
            agent->droppedOffCargo = agent->currentCargo;
            agent->setMass(agent->getMass() - agent->currentCargo);
            agent->currentCargo = 0.0;
        } else {
            agent->droppedOffCargo = 0.0;
        }
    }
}

void System::resolveCollisions() {
    //collisions happen between agents, and between resources
    //resources and agents also collide with celestials, which do not collide with each other
    //we can call our collide function for each pair
    for (auto it1 = resources.begin(); it1 != resources.end(); ++it1) {
        Resource* res1 = *it1;

        // collide with celestials
        for (Celestial* cel : celestials) {
            collideCelestial(res1, cel);
        }

        // collide with other resources (only those after res1)
        auto it2 = it1;
        ++it2;
        for (; it2 != resources.end(); ++it2) {
            Resource* res2 = *it2;
            collide(res1, res2);
        }

        // collide with agents
        for (Agent* agent : agents) {
            collide(res1, agent);
        }
    }

    //we need to reset the collision speed for each agent
    for (Agent* agent : agents) {
        agent->setCollided(0.0);
    }
    for (int i = 0; i < agents.size(); ++i) {
        Agent* agent1 = agents[i];
        //collide with celestials
        for (Celestial* cel : celestials) {
            double collisionSpeed = collideCelestial(agent1, cel);
            agent1->setCollided(collisionSpeed);
        }
        //collide with other agents
        for (int j = i + 1; j < agents.size(); ++j) {
            Agent* agent2 = agents[j];
            collide(agent1, agent2);
        }
    }
}


Vec2 System::calculateGravity(Body* targetBody) {
    Vec2  totalGravity(0.0, 0.0);

    constexpr double eps = 1e-4;  // softening length (tune in sim units)

    for (Body* body : gravityGeneratingBodies) {
        if (body == targetBody) continue; // skip self-gravity
        Vec2 d = body->position - targetBody->position;

        double r2 = d.norm2() + eps * eps;
        double inv_r = 1.0 / std::sqrt(r2);
        double inv_r3 = inv_r / r2;

        totalGravity += d * (G * body->getMass() * inv_r3);
    }

    return totalGravity;
}

Vec2 System::calculateGravity(const Vec2& location) const {
    Vec2 totalGravity(0.0, 0.0);

    constexpr double eps = 1e-4;  // softening length (tune in sim units)

    for (Body* body : gravityGeneratingBodies) {
        Vec2 d = body->position - location;

        double r2 = d.norm2() + eps * eps;
        double inv_r = 1.0 / std::sqrt(r2);
        double inv_r3 = inv_r / r2;

        totalGravity += d * (G * body->getMass() * inv_r3);
    }

    return totalGravity;
}

double System::getTotalKineticEnergy() const {
    double totalKE = 0.0;
    for (const Body* body : bodies) {
        totalKE += body->getKineticEnergy();
    }
    return totalKE;
}

double System::getTotalPotentialEnergy() const {
    double totalPE = 0.0;
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (size_t j = i + 1; j < bodies.size(); ++j) {
            Vec2 direction = bodies[j]->position - bodies[i]->position;
            double distance = direction.norm();
            if (distance > 1e-12) { // avoid division by zero
                totalPE -= G * bodies[i]->getMass() * bodies[j]->getMass() / distance;
            }
        }
    }
    return totalPE;
}

double System::getTotalEnergy() const {
    return getTotalKineticEnergy() + getTotalPotentialEnergy();
}