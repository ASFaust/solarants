#ifndef BODY_H
#define BODY_H

#include "Vec2.h"
#include <vector>
#include <unordered_map>
#include <string>

using namespace std;

// a body in the solar system
// all things are bodies: planets, suns, asteroids, spaceships, etc.

class Body {
    public:
        Vec2 position;
        Vec2 velocity;
        string name;
        double mass;
        double density;
        double radius;
        //Body(const Vec2& position_, const Vec2& velocity_, double mass_, double density_);
        Body(const string& name_, const Vec2& position_, const Vec2& velocity_, double mass_, double density_);
        void applyForce(const Vec2& force);
        double getKineticEnergy() const;
        unordered_map<string, double> getProperties() const;
};

#endif // BODY_H