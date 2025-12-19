#ifndef SYSTEM_H
#define SYSTEM_H
#include <vector>
#include <unordered_map>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "Vec2.h"
#include "Body.h"

//A 2D solar system simulation

using namespace std;

class System {
public:
    System(double sunMass, double sunDensity, double G_, double deltaTime_);
    double deltaTime;
    void initialize(); // to be called after all bodies are created, for leapfrog setup
    Body* sun;
    double G; //gravitational constant
    void step();

    void addBody(
        const string& name, 
        double mass,
        double density,
        bool emitGravity,
        double orbital_radius,
        double initial_angle,
        double ellipsity);

    void addMoon(
        const string& name,
        const string& hostName,
        double mass,
        double density,
        bool emitGravity,
        double orbital_radius,
        double initial_angle,
        double ellipsity);

    vector<Body*> allBodies, gravityGeneratingBodies;
    Vec2 calculateGravity(Body* targetBody);
    Vec2 calculateGravity(const Vec2& location) const;
    double getTotalKineticEnergy() const;
    double getTotalPotentialEnergy() const;
    double getTotalEnergy() const;
    unordered_map<string,unordered_map<string, double>> getAllBodyProperties() const;
    py::array_t<double> drawGravityField(int resolution, double radius) const;

};

#endif