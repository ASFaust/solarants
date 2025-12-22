#ifndef AGENT_H
#define AGENT_H

#include "Body.h"
#include "Vec2.h"
#include <string>
using namespace std;

class Agent : public Body {
public:
    Agent(System* system_, const string& name_, const Vec2& position_, const Vec2& velocity_, double mass_, double density_, Body* hostBody_);
    Body* hostBody; 
    Vec2 controlForce;
    void applyControlForce(const Vec2& force);
    vector<double> getSensorReadings() const;
};

#endif // AGENT_H