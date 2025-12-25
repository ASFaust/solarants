#include "Agent.h"
#include "System.h"

std::size_t Agent::global_id = 0;

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
    name = "Agent " + std::to_string(++global_id) + " of " + home->name;
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
    constexpr int numSlices = 8;
    constexpr double TWO_PI = 2.0 * M_PI;
    constexpr double EPS = 1e-8;

    vector<double> readings;
    readings.reserve(
        // angular body slices
        numSlices +
        // angular resource slices
        numSlices +
        // home: dx, dy, dvx, dvy, exp(-dist)
        5 +
        // nearest resource: dx, dy, dvx, dvy, exp(-dist)
        5 +
        // gravity
        2 +
        // cargo
        1
    );

    // ------------------------------------------------------------
    // Angular slices: bodies (nearest per slice)
    // ------------------------------------------------------------
    vector<double> bodySliceValue(numSlices, 0.0);
    vector<double> bodySliceNearestDist(numSlices,
        std::numeric_limits<double>::infinity());

    for (Body* body : system->bodies) {
        if (body == home) continue;

        Vec2 toBody = body->position - position;
        double centerDist = toBody.norm();
        double surfaceDist = std::max(0.0, centerDist - body->getRadius());

        if (centerDist < EPS) continue;

        double angle = std::atan2(toBody.y, toBody.x); // [-pi, pi)
        int slice = int((angle + M_PI) / TWO_PI * numSlices);
        slice = std::clamp(slice, 0, numSlices - 1);

        if (surfaceDist < bodySliceNearestDist[slice]) {
            bodySliceNearestDist[slice] = surfaceDist;
            bodySliceValue[slice] = std::exp(-surfaceDist);
        }
    }

    for (double v : bodySliceValue)
        readings.push_back(v);

    // ------------------------------------------------------------
    // Angular slices: resources (exp-weighted sum)
    // ------------------------------------------------------------
    vector<double> resourceSliceValue(numSlices, 0.0);

    for (Resource* res : system->resources) {
        Vec2 toRes = res->position - position;
        double centerDist = toRes.norm();
        double surfaceDist = std::max(0.0, centerDist - res->getRadius());

        if (centerDist < EPS) continue;

        double angle = std::atan2(toRes.y, toRes.x);
        int slice = int((angle + M_PI) / TWO_PI * numSlices);
        slice = std::clamp(slice, 0, numSlices - 1);

        double w = std::exp(-surfaceDist);
        resourceSliceValue[slice] += w;
    }

    for (double v : resourceSliceValue)
        readings.push_back(v);

    // ------------------------------------------------------------
    // Home planet features (retain)
    // ------------------------------------------------------------
    Vec2 toHome = home->position - position;
    Vec2 relVelHome = home->velocity - velocity;

    double homeCenterDist = toHome.norm();
    double homeSurfaceDist =
        std::max(0.0, homeCenterDist - home->getRadius());

    readings.push_back(toHome.x);
    readings.push_back(toHome.y);
    readings.push_back(relVelHome.x);
    readings.push_back(relVelHome.y);
    readings.push_back(std::exp(-homeSurfaceDist));

    // ------------------------------------------------------------
    // Nearest resource features (retain)
    // ------------------------------------------------------------
    double nearestResDist = std::numeric_limits<double>::infinity();
    Vec2 nearestResOffset(0.0, 0.0);
    Vec2 nearestResRelVel(0.0, 0.0);

    for (Resource* res : system->resources) {
        Vec2 toRes = res->position - position;
        double centerDist = toRes.norm();
        double surfaceDist = std::max(0.0, centerDist - res->getRadius());

        if (surfaceDist < nearestResDist) {
            nearestResDist = surfaceDist;
            nearestResOffset = toRes;
            nearestResRelVel = res->velocity - velocity;
        }
    }

    readings.push_back(nearestResOffset.x);
    readings.push_back(nearestResOffset.y);
    readings.push_back(nearestResRelVel.x);
    readings.push_back(nearestResRelVel.y);
    readings.push_back(
        std::isfinite(nearestResDist) ? std::exp(-nearestResDist) : 0.0
    );

    // ------------------------------------------------------------
    // Local gravity
    // ------------------------------------------------------------
    Vec2 g = system->calculateGravity(position);
    readings.push_back(g.x);
    readings.push_back(g.y);

    // ------------------------------------------------------------
    // Cargo status
    // ------------------------------------------------------------
    readings.push_back(
        cargoCapacity > 0.0 ? currentCargo / cargoCapacity : 0.0
    );

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
