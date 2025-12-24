#include "Agent.h"
#include "System.h"

Agent::Agent(System* system_, const Vec2& position_, 
    const Vec2& velocity_, double mass_, 
    double density_, Celestial* home_,
    double collectionRadius_, double maxControlForce_,
    double cargoCapacity_) 
    : Body(system_, position_, velocity_, mass_, density_), 
      home(home_), 
      collectionRadius(collectionRadius_), 
      maxControlForce(maxControlForce_),
      cargoCapacity(cargoCapacity_) {
    name = "Agent";
}

void Agent::applyControlForce(const Vec2& force) {
    // force is assumed to be in len 0...1 range
    Vec2 clampedForce = force;
    if (clampedForce.norm() > 1.0) {
        clampedForce = clampedForce.normalized();
    }
    controlForce = clampedForce * maxControlForce;
}

vector<double> Agent::getSensorReadings() const {
    // Placeholder: return empty sensor readings
    /*
    some sensor readings could be:
    * Distance and delta v to home body surface (1/(distance+1) to keep it bounded)
    * distance and delta-v to nearest other body
    * distance and delta-v to nearest resource
    * Local gravitational field strength (2d vector)
    * cargo bay status (percentage full)
    */
    //return vector<double>();
    vector<double> readings;
    // Distance and delta v to home body surface
    Vec2 toHome = home->position - position;
    double distanceToHomeSurface = toHome.norm() - home->getRadius();
    Vec2 relativeVelocityToHome = home->velocity - velocity;
    readings.push_back(1.0 / (distanceToHomeSurface + 1.0));
    readings.push_back(relativeVelocityToHome.x);
    readings.push_back(relativeVelocityToHome.y);
    //normalize toHome and append it
    Vec2 toHomeDir = toHome.normalized();
    readings.push_back(toHomeDir.x);
    readings.push_back(toHomeDir.y);

    // Distance and delta v to nearest other body
    double nearestBodyDistance = std::numeric_limits<double>::max();;
    Vec2 nearestBodyRelVelocity(0.0, 0.0);
    Vec2 nearestBodyDir(0.0, 0.0);
    for (Body* body : system->bodies) {
        if (body == this) continue;
        Vec2 toBody = body->position - position;
        double distToBodySurface = toBody.norm() - body->getRadius();
        if (distToBodySurface < nearestBodyDistance) {
            nearestBodyDistance = distToBodySurface;
            nearestBodyDir = toBody.normalized();
            nearestBodyRelVelocity = body->velocity - velocity;
        }
    }
    readings.push_back(1.0 / (nearestBodyDistance + 1.0));
    readings.push_back(nearestBodyRelVelocity.x);
    readings.push_back(nearestBodyRelVelocity.y);
    readings.push_back(nearestBodyDir.x);
    readings.push_back(nearestBodyDir.y);

    // Distance and delta v to nearest resource
    double nearestResourceDistance = std::numeric_limits<double>::max();;
    Vec2 nearestResourceRelVelocity(0.0, 0.0);
    Vec2 nearestResourceDir(0.0, 0.0);
    for (Resource* res : system->resources) {
        Vec2 toRes = res->position - position;
        double distToResSurface = toRes.norm() - res->getRadius();
        if (distToResSurface < nearestResourceDistance) {
            nearestResourceDistance = distToResSurface;
            nearestResourceDir = toRes.normalized();
            nearestResourceRelVelocity = res->velocity - velocity;
        }
    }
    readings.push_back(1.0 / (nearestResourceDistance + 1.0));
    readings.push_back(nearestResourceRelVelocity.x);
    readings.push_back(nearestResourceRelVelocity.y);
    readings.push_back(nearestResourceDir.x);
    readings.push_back(nearestResourceDir.y);

    // Local gravitational field strength
    Vec2 localGravity = system->calculateGravity(position);
    readings.push_back(localGravity.x);
    readings.push_back(localGravity.y);

    // Cargo bay status
    double cargoStatus = currentCargo / cargoCapacity;
    readings.push_back(cargoStatus);

    return readings;
}

double Agent::computeReward() {
    // Placeholder for reward computation
    /*
    Possible reward components:
    * Positive reward for dropping off cargo at home
    * Negative reward for collisions (based on collisionSpeed)
    * Negative reward for spent fuel (based on controlForce magnitude)
    */
    double reward = 0.0;
    reward += droppedOffCargo * 100.0; // e.g., 100 points per unit cargo dropped off
    reward -= collisionSpeed * 0.1;   // e.g., -0.1 points per unit collision speed
    reward -= controlForce.norm() * 0.01; // e.g., -0.01 points per unit control force applied
    return reward;
}
