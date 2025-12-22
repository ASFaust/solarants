#include "Agent.h"

Agent::Agent(
        System* system_, const string& name_, const Vec2& position_, const Vec2& velocity_, 
        double mass_, double density_, Body* hostBody_) : Body(system_, name_, position_, velocity_, mass_, density_) {
    hostBody = hostBody_;
    controlForce = Vec2(0.0, 0.0);
}

void Agent::applyControlForce(const Vec2& force) {
    controlForce += force;
}

vector<double> Agent::getSensorReadings() const {
    // Placeholder: return empty sensor readings
    /*
    some sensor readings could be:
    * Distance and delta v to host body surface (1/(distance+1) to keep it bounded)
    * Local gravitational field strength (2d vector)
    * distance and delta-v to nearest other body
    later we can add:
    * distance and delta-v to nearest resource deposit
    * cargo bay status (percentage full)
    */
    return vector<double>();
}