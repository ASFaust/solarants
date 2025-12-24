#ifndef BODY_H
#define BODY_H

#include "Vec2.h"
#include <vector>
#include <string>

using namespace std;

// a body in the solar system
// all things are bodies: planets, suns, asteroids, spaceships, etc.

class System;

class Body {
    public:
        Body(System* system_, const Vec2& position_, const Vec2& velocity_, double mass_, double density_);

        inline double getMass() const { return mass; }
        inline double getInvMass() const { return invMass; }
        void  setMass(double m);
        inline double getDensity() const { return density; }
        void  setDensity(double d);
        inline double getRadius() const { return radius; }
        Vec2 position;
        Vec2 velocity;

        System* system;
        string name = "";

        double getKineticEnergy() const;
        double getSurfaceGravity() const;

    private:
        void computeSurfaceGravity();
        void computeRadius();

        double mass;
        double invMass;
        double density;
        double radius;
        double surfaceGravity;
};

#endif // BODY_H