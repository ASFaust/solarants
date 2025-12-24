#include "Celestial.h"


Celestial::Celestial(System* system_, const string& name_, 
    const Vec2& position_, const Vec2& velocity_, 
    double mass_, double density_) 
    : Body(system_, position_, velocity_, mass_, density_) {
    name = name_;
}

