#include "Body.h"

Body::Body(const string& name_, const Vec2& position_, const Vec2& velocity_, double mass_, double density_){
    name = name_;
    position = position_;
    velocity = velocity_;
    mass = mass_;
    density = density_;
    radius = cbrt((3.0 * mass) / (4.0 * M_PI * density));
}

void Body::applyForce(const Vec2& force) {

}

double Body::getKineticEnergy() const {
    double speed2 = velocity.norm2();
    return 0.5 * mass * speed2;
}

unordered_map<string, double> Body::getProperties() const {
    unordered_map<string, double> properties;
    properties["mass"] = mass;
    properties["position_x"] = position.x;
    properties["position_y"] = position.y;
    properties["velocity_x"] = velocity.x;
    properties["velocity_y"] = velocity.y;
    properties["density"] = density;
    properties["radius"] = radius;
    return properties;
}
