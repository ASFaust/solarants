// Resource.cpp
#include "Resource.h"

std::size_t Resource::global_id = 0;

Resource::Resource(System* system_,
                   const Vec2& position_,
                   const Vec2& velocity_,
                   double mass_,
                   double density_)
    : Body(system_, position_, velocity_, mass_, density_) {

    name = "Resource " + std::to_string(++global_id);
}
