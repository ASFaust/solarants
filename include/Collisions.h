#ifndef COLLISIONS_H
#define COLLISIONS_H

#include "Body.h"
#include "Celestial.h"
#include "Vec2.h"
#include <algorithm>
#include <cmath>

using namespace std;

inline void collide(
    Body* A,
    Body* B,
    double restitution = 0.5,
    double frictionCoeff = 0.5,
    double penetrationPercent = 0.2,
    double slop = 1e-6,
    double eps2 = 1e-20
) {
    Vec2 deltaPos = B->position - A->position;
    double dist2 = deltaPos.norm2();

    double radiusSum = A->getRadius() + B->getRadius();
    double radiusSum2 = radiusSum * radiusSum;

    if (!std::isfinite(dist2))
        return;

    // No collision or degenerate case
    if (dist2 >= radiusSum2 || dist2 < eps2)
        return;

    double dist = std::sqrt(dist2);
    Vec2 normal = deltaPos / dist;

    // Relative velocity
    Vec2 rv = B->velocity - A->velocity;
    double velAlongNormal = rv.dot(normal);

    // Separating
    if (velAlongNormal >= 0.0)
        return;
    const double invMassA = A->getInvMass();
    const double invMassB = B->getInvMass();
    const double invMassSum = invMassA + invMassB;
    if (invMassSum <= 0.0)
        return;

    // ----------------------------
    // Normal impulse
    // ----------------------------
    double jn = -(1.0 + restitution) * velAlongNormal;
    jn /= invMassSum;

    Vec2 impulseN = normal * jn;
    A->velocity -= impulseN * invMassA;
    if (invMassB > 0.0)
        B->velocity += impulseN * invMassB;

    // ----------------------------
    // Tangential (friction) impulse
    // ----------------------------
    Vec2 vt = rv - normal * velAlongNormal;
    double vt2 = vt.norm2();

    if (vt2 > eps2) {
        Vec2 tangent = vt / std::sqrt(vt2);

        double jt = -rv.dot(tangent);
        jt /= invMassSum;

        double maxFriction = frictionCoeff * jn;
        jt = std::clamp(jt, -maxFriction, maxFriction);

        Vec2 impulseT = tangent * jt;
        A->velocity -= impulseT * invMassA;
        if (invMassB > 0.0)
            B->velocity += impulseT * invMassB;
    }

    // ----------------------------
    // Positional correction
    // ----------------------------
    double penetration = radiusSum - dist;
    if (penetration > slop) {
        double corrMag =
            (penetration - slop) / invMassSum * penetrationPercent;
        Vec2 correction = normal * corrMag;
        A->position -= correction * invMassA;
        if (invMassB > 0.0)
            B->position += correction * invMassB;
    }
}

inline double collideCelestial(
    Body* A,
    Celestial* B,
    double restitution = 0.5,
    double frictionCoeff = 0.5,
    double penetrationPercent = 0.2,
    double slop = 1e-6,
    double eps2 = 1e-12
) {
    //B's invMass is zero, so we can optimize a bit
    Vec2 deltaPos = B->position - A->position;
    double dist2 = deltaPos.norm2();
    double radiusSum = A->getRadius() + B->getRadius();
    double radiusSum2 = radiusSum * radiusSum;
    // No collision or degenerate case
    if (!std::isfinite(dist2))
        return 0.0;

    if (dist2 >= radiusSum2 || dist2 < eps2)
        return 0.0;

    double dist = std::sqrt(dist2);
    Vec2 normal = deltaPos / dist;
    // Relative velocity
    Vec2 rv = B->velocity - A->velocity;
    double velAlongNormal = rv.dot(normal);
    // Separating
    if (velAlongNormal >= 0.0)
        return 0.0;
    const double invMassA = A->getInvMass();
    if (invMassA <= 0.0)
        return 0.0;
    // ----------------------------
    // Normal impulse
    // ----------------------------
    double jn = -(1.0 + restitution) * velAlongNormal;
    jn /= invMassA;
    Vec2 impulseN = normal * jn;
    A->velocity -= impulseN * invMassA;
    // ----------------------------
    // Tangential (friction) impulse
    // ----------------------------
    Vec2 vt = rv - normal * velAlongNormal;
    double vt2 = vt.norm2();
    if (vt2 > eps2) {
        Vec2 tangent = vt / std::sqrt(vt2);
        double jt = -rv.dot(tangent);
        jt /= invMassA;
        double maxFriction = frictionCoeff * jn;
        jt = std::clamp(jt, -maxFriction, maxFriction);
        Vec2 impulseT = tangent * jt;
        A->velocity -= impulseT * invMassA;
    }
    // ----------------------------
    // Positional correction
    // ----------------------------
    double penetration = radiusSum - dist;
    if (penetration > slop) {
        double corrMag =
            (penetration - slop) / invMassA * penetrationPercent;
        Vec2 correction = normal * corrMag;
        A->position -= correction * invMassA;
    }
    return -velAlongNormal;
}

#endif // COLLISIONS_H