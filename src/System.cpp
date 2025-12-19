#include "System.h"
#include "Body.h"
#include "Vec2.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

System::System(double sunMass, double sunDensity, double G_, double deltaTime_){
    G = G_;
    deltaTime = deltaTime_;
    sun = new Body("Sun",Vec2(0,0),Vec2(0,0), sunMass, sunDensity);
    allBodies.push_back(sun);
    gravityGeneratingBodies.push_back(sun);
}

unordered_map<string,unordered_map<string, double>> System::getAllBodyProperties() const {
    unordered_map<string,unordered_map<string, double>> propertiesMap;
    for (const Body* body : allBodies) {
        propertiesMap[body->name] = body->getProperties();
    }
    return propertiesMap;
}

void System::addBody(
        const string& name,
        double mass,
        double density,
        bool emitGravity,
        double orbital_radius,
        double initial_angle,
        double ellipsity) {
    // Direction from sun
    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 tangent_dir(-radial_dir.y, radial_dir.x); // 90° CCW

    // Position relative to sun
    Vec2 position = sun->position + radial_dir * orbital_radius;

    // Semi-major axis interpretation
    double a = orbital_radius / ellipsity;

    // Gravitational parameter
    double mu = G * sun->mass;

    // Vis-viva equation
    double orbital_speed = std::sqrt(mu * (2.0 / orbital_radius - 1.0 / a));

    // Velocity (purely tangential at apo/periapsis)
    Vec2 velocity = sun->velocity + tangent_dir * orbital_speed;

    Body* body = new Body(name, position, velocity, mass, density);

    allBodies.push_back(body);

    if(emitGravity){
        gravityGeneratingBodies.push_back(body);
    }
}

void System::addMoon(
    const string& name,
    const string& hostName,
    double mass,
    double density,
    bool emitGravity,
    double orbital_radius,
    double initial_angle,
    double ellipsity) {
    // Find host body
    Body* hostBody = nullptr;
    for (Body* body : allBodies) {
        if (body->name == hostName) {
            hostBody = body;
            break;
        }
    }   
    if (!hostBody) {
        throw std::runtime_error("Host body not found: " + hostName);
    }
    // subtract moon mass from host body mass
    hostBody->mass -= mass; 
    // Direction from host
    Vec2 radial_dir(std::cos(initial_angle), std::sin(initial_angle));
    Vec2 tangent_dir(-radial_dir.y, radial_dir.x); // 90° CCW   
    // Position relative to host
    Vec2 position = hostBody->position + radial_dir * orbital_radius;
    // Semi-major axis interpretation
    double a = orbital_radius / ellipsity;
    // Gravitational parameter
    double mu = G * hostBody->mass;
    // Vis-viva equation
    double orbital_speed = std::sqrt(mu * (2.0 / orbital_radius - 1.0 / a));
    // Velocity (purely tangential at apo/periapsis)
    Vec2 velocity = hostBody->velocity + tangent_dir * orbital_speed;   
    Body* body = new Body(name, position, velocity, mass, density);   
    allBodies.push_back(body);   
    if(emitGravity){
        gravityGeneratingBodies.push_back(body);
    }
}


void System::initialize(){
    //initialize all bodies for leapfrog integration
    for (Body* body : allBodies) {
        Vec2 totalGravity = calculateGravity(body);
        body->velocity += totalGravity * 0.5 * deltaTime;
    }
}

void System::step() {
    Vec2 totalVelocity(0.0, 0.0);

    // 1. drift
    for (Body* body : allBodies) {
        body->position += body->velocity * deltaTime;
    }

    // 2. full kick
    for (Body* body : allBodies) {
        Vec2 totalGravity = calculateGravity(body);
        body->velocity += totalGravity * deltaTime;
        totalVelocity += body->velocity;
    }


    // 3. remove sun position drift: sun always sits at (0,0)
    Vec2 sunDrift = sun->position;
    sun->position = Vec2(0.0, 0.0);
    for (Body* body : allBodies) {
        body->position -= sunDrift;
    }
}



py::array_t<double> System::drawGravityField(int resolution, double radius) const {
    // shape = (resolution, resolution)
    py::array_t<double> field({resolution, resolution});
    auto buf = field.mutable_unchecked<2>();

    for (int i = 0; i < resolution; ++i) {
        for (int j = 0; j < resolution; ++j) {
            double x = (static_cast<double>(i) / (resolution - 1) - 0.5) * 2.0 * radius;
            double y = (static_cast<double>(j) / (resolution - 1) - 0.5) * 2.0 * radius;
            buf(i, j) = calculateGravity(Vec2(x, y)).norm();
        }
    }

    return field;
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

            totalGravity += d * (G * body->mass * inv_r3);
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

        totalGravity += d * (G * body->mass * inv_r3);
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
                totalPE -= G * allBodies[i]->mass * allBodies[j]->mass / distance;
            }
        }
    }
    return totalPE;
}

double System::getTotalEnergy() const {
    return getTotalKineticEnergy() + getTotalPotentialEnergy();
}