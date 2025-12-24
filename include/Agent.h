#ifndef AGENT_H
#define AGENT_H

#include "Body.h"
#include "Celestial.h"
#include "Vec2.h"
#include <string>
using namespace std;

class Agent : public Body {
public:
    Agent(System* system_, const Vec2& position_, 
        const Vec2& velocity_, double mass_, 
        double density_, Celestial* home_,
        double collectionRadius_, double maxControlForce_,
        double cargoCapacity_);
    Celestial* home; 
    Vec2 controlForce;
    double collectionRadius;
    double maxControlForce;
    double cargoCapacity;
    double currentCargo = 0.0;
    double collisionSpeed = 0.0;
    double droppedOffCargo = 0.0; //amount of cargo dropped off at home in last step
    void applyControlForce(const Vec2& force);
    vector<double> getSensorReadings() const;
    double computeReward();
    inline void setCollided(double collisionSpeed_) { collisionSpeed = collisionSpeed_; }
};

#endif // AGENT_H