#include "Body.h"
#include "System.h"

Body::Body(System* system_, const string& name_, const Vec2& position_, const Vec2& velocity_, double mass_, double density_){
    system = system_;
    name = name_;
    position = position_;
    velocity = velocity_;
    mass = mass_;
    invMass = 1.0 / mass;
    density = density_;
    computeRadius();
    computeSurfaceGravity();
    domainRadius = computeDomainRadius();
    collisionForce = Vec2(0.0, 0.0);
}

double Body::getKineticEnergy() const {
    double speed2 = velocity.norm2();
    return 0.5 * mass * speed2;
}

double Body::getDomainRadius() {
    //the domain radius is a lower bound on radius where this body dominates the gravity field
    double newRadius = computeDomainRadius();
    if (newRadius < domainRadius) {
        domainRadius = newRadius; 
    }
    return domainRadius;
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

double Body::computeDomainRadius() {
    double minRadius = std::numeric_limits<double>::infinity();

    // Gravity at the body center (background acceleration)
    Vec2 g_center = system->calculateGravity(position);

    for (Body* other : system->gravityGeneratingBodies) {
        if (other == this) continue;

        // Direction from this body toward the other
        Vec2 dir = (other->position - position).normalized();

        double low = 0.0;
        double high = (other->position - position).norm(); // natural upper bound

        for (int iter = 0; iter < 12; iter++) {
            double mid = 0.5 * (low + high);
            Vec2 x = position + mid * dir;

            // Net gravity at test point
            Vec2 g_here = system->calculateGravity(x);

            // Relative (tidal) gravity in this body's free-fall frame
            Vec2 g_rel = g_here - g_center;

            // Vector pointing back toward this body
            Vec2 toUs = position - x;

            if (g_rel.dot(toUs) > 0.0) {
                low = mid;    // still dominated by this body
            } else {
                high = mid;   // dominance lost
            }
        }

        minRadius = std::min(minRadius, low);
    }

    return minRadius;
}

