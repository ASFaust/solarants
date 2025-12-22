#include "System.h"
#include "Body.h"
#include "Vec2.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

System::System(double G_, double deltaTime_){
    G = G_;
    deltaTime = deltaTime_;
}

Body* System::getBodyByName(const string& name) const {
    for (Body* body : allBodies) {
        if (body->name == name) {
            return body;
        }
    }
    return nullptr; // Not found
}

void System::addBody(
        const string& name,
        Vec2 position,
        Vec2 velocity,
        double mass,
        double density,
        bool emitGravity
    ) {
    if(getBodyByName(name)){
        throw std::runtime_error("Body name already exists: " + name);
    }
    
    Body* body = new Body(this, name, position, velocity, mass, density);

    allBodies.push_back(body);

    if(emitGravity){
        gravityGeneratingBodies.push_back(body);
    }
}

// e.g. to split a mass into a moon orbiting a host body
//or a planet orbiting a sun
//the mass is deducted from the host body
//the orbit (radius etc.) is calculated relative to the host body
void System::splitBody(
    const string& hostName,
    const string& splitBodyName,
    double mass,
    double density,
    bool emitGravity,
    double orbital_radius,
    double initial_angle,
    double ellipsity,
    bool prograde
) {
    // Find host body
    Body* hostBody = getBodyByName(hostName);
    if (!hostBody) {
        throw std::runtime_error("Host body not found: " + hostName);
    }
    if(getBodyByName(splitBodyName)){
        throw std::runtime_error("Split body name already exists: " + splitBodyName);
    }
    // Deduct mass from host body
    if (hostBody->getMass() < mass) {
        throw std::runtime_error("Host body does not have enough mass to split.");
    }
    hostBody->setMass(hostBody->getMass() - mass);
    // Direction from host body
    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 tangent_dir(-radial_dir.y, radial_dir.x); // 90° CCW
    // Position relative to host body
    Vec2 position = hostBody->position + radial_dir * orbital_radius;
    // Semi-major axis interpretation
    double a = orbital_radius / ellipsity;
    // Gravitational parameter
    double mu = G * hostBody->getMass();
    // Vis-viva equation
    double orbital_speed = std::sqrt(mu * (2.0 / orbital_radius - 1.0 / a));
    // Velocity (purely tangential at apo/periapsis)
    double massRatio = mass / (hostBody->getMass() + mass);
    Vec2 v_rel = tangent_dir * orbital_speed;

    if (!prograde) {
        v_rel = v_rel * -1.0;
    }

    hostBody->velocity -= v_rel * massRatio;
    Vec2 child_velocity = hostBody->velocity + v_rel * (1.0 - massRatio);

    Body* body = new Body(this, splitBodyName, position, child_velocity, mass, density);

    body->parent = hostBody;

    allBodies.push_back(body);

    if(emitGravity){
        gravityGeneratingBodies.push_back(body);
    }
}

void System::addAgent(
    const string& hostBodyName,
    const string& agentName,
    double mass,
    double radius,
    double initial_angle, //where on the surface to spawn. in radians
    bool emitGravity
) {
    Body* hostBody = getBodyByName(hostBodyName);
    if (!hostBody) {
        throw std::runtime_error("Host body not found: " + hostBodyName);
    }
    if(getBodyByName(agentName)){
        throw std::runtime_error("Agent name already exists: " + agentName);
    }

    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 position = hostBody->position + radial_dir * (hostBody->getRadius() + radius);
    Vec2 velocity = hostBody->velocity; // start with host body's velocity

    double density = (3.0 * mass) / (4.0 * M_PI * radius * radius * radius);

    Agent* agent = new Agent(this, agentName, position, velocity, mass, density, hostBody);

    //cast to Body* and add to allBodies
    allBodies.push_back(static_cast<Body*>(agent));
    allAgents.push_back(agent);

    if(emitGravity){
        gravityGeneratingBodies.push_back(static_cast<Body*>(agent));
    }
}

void System::initialize(){
    //initialize all bodies for leapfrog integration
    for (Body* body : allBodies) {
        Vec2 totalGravity = calculateGravity(body);
        body->velocity += totalGravity * 0.5 * deltaTime;
    }
}

void System::step(int n) {
    for (int i = 0; i < n; ++i) {   
        Vec2 totalVelocity(0.0, 0.0);

        // 1. drift
        for (Body* body : allBodies) {
            body->position += body->velocity * deltaTime;
        }

        // 2. full kick
        for (Body* body : allBodies) {
            body->velocity += calculateGravity(body) * deltaTime;
        }

        // 3 control forces from agents
        for (Agent* agent : allAgents) {
            agent->velocity += agent->controlForce * agent->getInvMass() * deltaTime;
            agent->controlForce = Vec2(0.0, 0.0); // reset control force after applying
        }

        // 4 compute collisions
        resolveCollisions();

        for (Body* body : allBodies) {
            body->velocity += body->collisionForce * body->getInvMass() * deltaTime;
            body->collisionForce = Vec2(0.0, 0.0); // reset collision force after applying
        }
    }
}

void System::resolveCollisions() {
    for (size_t i = 0; i < allBodies.size(); ++i) {
        Body* A = allBodies[i];
        const double invMassA = A->getInvMass();
        if (invMassA == 0.0) continue;

        for (size_t j = 0; j < allBodies.size(); ++j) {
            if (i == j) continue;
            Body* B = allBodies[j];
            const double invMassB = B->getInvMass();
            Vec2 deltaPos = B->position - A->position;
            double dist2 = deltaPos.norm2();

            double radiusSum = A->getRadius() + B->getRadius();
            double radiusSum2 = radiusSum * radiusSum;
            if (dist2 < radiusSum2) {
                // Collision detected
                double dist = std::sqrt(dist2);
                Vec2 collisionNormal;
                if (dist > 1e-12) {
                    collisionNormal = deltaPos / dist;
                } else {
                    collisionNormal = Vec2(1.0, 0.0); // arbitrary direction    
                }
                double penetrationDepth = radiusSum - dist;
                Vec2 force = collisionNormal * penetrationDepth;
                A->collisionForce -= force;
                B->collisionForce += force;
                Vec2 relativeVelocity = B->velocity - A->velocity;
                //we also dampen relative velocity by applying force opposite to relative velocity
                //weighted by the inverse masses so that lighter bodies are affected more
                double factor = invMassA / (invMassA + invMassB);
                double scaling = 0.01;
                A->collisionForce -= relativeVelocity * factor * scaling;
                B->collisionForce += relativeVelocity * (1.0 - factor) * scaling;
            }
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
    for (const Body* body : allBodies) {
        totalKE += body->getKineticEnergy();
    }
    return totalKE;
}

double System::getTotalPotentialEnergy() const {
    double totalPE = 0.0;
    for (size_t i = 0; i < allBodies.size(); ++i) {
        for (size_t j = i + 1; j < allBodies.size(); ++j) {
            Vec2 direction = allBodies[j]->position - allBodies[i]->position;
            double distance = direction.norm();
            if (distance > 1e-12) { // avoid division by zero
                totalPE -= G * allBodies[i]->getMass() * allBodies[j]->getMass() / distance;
            }
        }
    }
    return totalPE;
}

double System::getTotalEnergy() const {
    return getTotalKineticEnergy() + getTotalPotentialEnergy();
}