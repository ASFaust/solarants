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
#include "Agent.h"

//A 2D solar system simulation

using namespace std;

class System {
public:
    System(double G_, double deltaTime_);
    double deltaTime;
    void initialize(); // to be called after all bodies are created, for leapfrog setup
    double G; //gravitational constant
    void step(int n);

    void addBody(
        const string& name, 
        Vec2 position, 
        Vec2 velocity, 
        double mass, 
        double density, 
        bool emitGravity);

    //e.g. to split a mass into a moon orbiting a host body
    void splitBody(
        const string& hostName,
        const string& splitBodyName,
        double mass,
        double density,
        bool emitGravity,
        double orbital_radius,
        double initial_angle,
        double ellipsity,
        bool prograde
    );

    //agents spawn somewhere on the surface of their host body
    //other than all other bodies, agents are defined by mass and radius instead of mass and density
    void addAgent(
        const string& hostBodyName,
        const string& agentName,
        double mass,
        double radius, 
        double initial_angle, //where on the surface to spawn. in radians
        bool emitGravity
    ); 

    Body* getBodyByName(const string& name) const;

    vector<Body*> allBodies;
    vector<Body*> allCelestialBodies; // bodies that are not agents
    vector<Body*> gravityGeneratingBodies;
    vector<Agent*> allAgents;

    void resolveCollisions(); 

    Vec2 calculateGravity(Body* targetBody);
    Vec2 calculateGravity(const Vec2& location) const;
    double getTotalKineticEnergy() const;
    double getTotalPotentialEnergy() const;
    double getTotalEnergy() const;
};

#endif