#include "bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//we need to add string handling
#include <string>

#include "System.h"
#include "Vec2.h"
#include "Body.h"
#include "Agent.h"
#include "Resource.h"

PYBIND11_MODULE(solarants, m) {
    m.doc() = "Core bindings for SolarAnts simulation";

    pybind11::class_<Vec2>(m, "Vec2")
        .def(pybind11::init<>())
        .def(pybind11::init<double, double>())
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("y", &Vec2::y)
        .def("__add__", &Vec2::operator+)
        .def("__sub__", &Vec2::operator-)
        .def("__mul__", [](const Vec2 &v, double s) { return v * s; })
        .def("__truediv__", &Vec2::operator/)
        .def("dot", &Vec2::dot)
        .def("norm2", &Vec2::norm2)
        .def("norm", &Vec2::norm)
        .def("normalized", &Vec2::normalized);

    pybind11::class_<System>(m, "System")
        .def(pybind11::init<double, double>(),
             pybind11::arg("G"),
             pybind11::arg("deltaTime"))
        .def("initialize", &System::initialize)
        .def("step", &System::step, pybind11::arg("n") = 1)
        .def("addCelestial", &System::addCelestial,
                pybind11::arg("name"),
                pybind11::arg("position"),
                pybind11::arg("velocity"),
                pybind11::arg("mass"),
                pybind11::arg("density"),
                pybind11::arg("emitGravity"))
         .def("splitCelestial", &System::splitCelestial,     
             pybind11::arg("hostName"),
             pybind11::arg("splitCelestialName"),
             pybind11::arg("mass"),
             pybind11::arg("density"),
             pybind11::arg("emitGravity"),
             pybind11::arg("orbital_radius"),
             pybind11::arg("initial_angle"),
             pybind11::arg("ellipsity"),
             pybind11::arg("prograde")=true)
        .def("addAgent", &System::addAgent,
            pybind11::arg("hostName"),
            pybind11::arg("mass"),
            pybind11::arg("radius"),
            pybind11::arg("initial_angle"),
            pybind11::arg("collectionRadius"),
            pybind11::arg("maxControlForce"),
            pybind11::arg("cargoCapacity"))
        .def("addResourceInOrbit", &System::addResourceInOrbit,
            pybind11::arg("hostName"),
            pybind11::arg("mass"),
            pybind11::arg("density"),
            pybind11::arg("orbital_radius"),
            pybind11::arg("initial_angle"),
            pybind11::arg("ellipsity"),
            pybind11::arg("prograde"))
        .def("addResourceOnSurface", &System::addResourceOnSurface,
            pybind11::arg("hostName"),
            pybind11::arg("mass"),
            pybind11::arg("density"),
            pybind11::arg("initial_angle"))
        .def("getTotalEnergy", &System::getTotalEnergy)
        .def_readonly("bodies",&System::bodies, 
            pybind11::return_value_policy::reference_internal)
        .def_readonly("celestials",&System::celestials,
        pybind11::return_value_policy::reference_internal)
        .def_readonly("resources",&System::resources,
        pybind11::return_value_policy::reference_internal)
        .def_readonly("agents", &System::agents,
        pybind11::return_value_policy::reference_internal)
        //then the calculateGravity function which takes in a Vec2 and produces a Vec2
        //needs to be a lambda to aid template deduction
        .def("calculateGravity",
            py::overload_cast<const Vec2&>(&System::calculateGravity, py::const_));
        
    
    pybind11::class_<Body>(m, "Body")
        .def_property("mass", &Body::getMass, &Body::setMass)
        .def_property("density", &Body::getDensity, &Body::setDensity)
        .def_property_readonly("radius", &Body::getRadius)
        .def_readwrite("position", &Body::position)
        .def_readwrite("velocity", &Body::velocity)
        .def_readonly("name", &Body::name)
        .def_property_readonly("surfaceGravity", &Body::getSurfaceGravity);

    pybind11::class_<Resource, Body>(m, "Resource");
    pybind11::class_<Celestial, Body>(m, "Celestial");

    pybind11::class_<Agent, Body>(m, "Agent")
        .def("applyControlForce", &Agent::applyControlForce)
        .def("getSensorReadings", &Agent::getSensorReadings)
        .def("computeReward", &Agent::computeReward)
        .def_property_readonly("cargoStatus", &Agent::getCargoStatus);
}
