#ifndef RESOURCE_H
#define RESOURCE_H

#include <string>
#include "Body.h"

//resources floating in space or lying on the ground that can be collected by agents

class Resource : public Body {
    public:
        Resource(System* system_, const Vec2& position_, 
            const Vec2& velocity_, double mass_, double density_);
            
};
    
#endif // RESOURCE_H