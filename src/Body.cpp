#include "Body.h"
#include "System.h"

Body::Body(System* system_, const Vec2& position_, const Vec2& velocity_, double mass_, double density_){
    system = system_;
    position = position_;
    velocity = velocity_;
    mass = mass_;
    invMass = 1.0 / mass;
    density = density_;
    computeRadius();
    computeSurfaceGravity();
}

double Body::getKineticEnergy() const {
    double speed2 = velocity.norm2();
    return 0.5 * mass * speed2;
}

double Body::getSurfaceGravity() const {
    return surfaceGravity;
}

void Body::setMass(double m) {
    mass = m;
    invMass = 1.0 / mass;
    computeRadius();
    computeSurfaceGravity();
}

void Body::setDensity(double d) {
    density = d;
    computeRadius();
    computeSurfaceGravity();
}

void Body::computeRadius() {
    radius = cbrt((3.0 * mass) / (4.0 * M_PI * density));
}

void Body::computeSurfaceGravity() {
    surfaceGravity = (system->G * mass) / (radius * radius);
}


