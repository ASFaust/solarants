#ifndef CELESTIAL_H
#define CELESTIAL_H

#include "Body.h"
#include <string>

//later we can add atmospheres, composition, etc

class Celestial : public Body {
    public:
        Celestial(System* system_, const string& name_, 
            const Vec2& position_, const Vec2& velocity_, 
            double mass_, double density_);
        Celestial* parent = nullptr;
};

#endif // CELESTIAL_H