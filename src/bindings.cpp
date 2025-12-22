#include "bindings.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
//we need to add string handling
#include <string>

#include "System.h"
#include "Vec2.h"
#include "Body.h"

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
        .def("addBody", &System::addBody,
                pybind11::arg("name"),
                pybind11::arg("position"),
                pybind11::arg("velocity"),
                pybind11::arg("mass"),
                pybind11::arg("density"),
                pybind11::arg("emitGravity"))
         .def("splitBody", &System::splitBody,     
             pybind11::arg("name"),
             pybind11::arg("splitBodyName"),
             pybind11::arg("mass"),
             pybind11::arg("density"),
             pybind11::arg("emitGravity"),
             pybind11::arg("orbital_radius"),
             pybind11::arg("initial_angle"),
             pybind11::arg("ellipsity"),
             pybind11::arg("prograde")=true)
        .def("addAgent", &System::addAgent,
            pybind11::arg("hostBodyName"),
            pybind11::arg("agentName"),
            pybind11::arg("mass"),
            pybind11::arg("radius"),
            pybind11::arg("initial_angle"),
            pybind11::arg("emitGravity"))
        .def("getTotalEnergy", &System::getTotalEnergy)
        .def_property_readonly("bodies",
            [](System& self) -> std::vector<Body*>& {
                return self.allBodies;
            },
        py::return_value_policy::reference_internal)
        .def_property_readonly("agents", 
            [](System& self) -> std::vector<Agent*>& {
                return self.allAgents;
            },
        py::return_value_policy::reference_internal)
        //then the calculateGravity function which takes in a Vec2 and produces a Vec2
        .def("calculateGravity", [](System& self, const Vec2& location) {
            return self.calculateGravity(location);
        });
        
    
    pybind11::class_<Body>(m, "Body")
        .def_property("mass", &Body::getMass, &Body::setMass)
        .def_property("density", &Body::getDensity, &Body::setDensity)
        .def_property_readonly("radius", &Body::getRadius)
        .def_readwrite("position", &Body::position)
        .def_readwrite("velocity", &Body::velocity)
        .def_readonly("name", &Body::name)
        .def_property_readonly("surfaceGravity", &Body::getSurfaceGravity)
        .def_property_readonly("domainRadius", &Body::getDomainRadius);

    pybind11::class_<Agent, Body>(m, "Agent")
        .def_readonly("hostBody", &Agent::hostBody)
        .def_readwrite("controlForce", &Agent::controlForce)
        .def("applyControlForce", &Agent::applyControlForce)
        .def("getSensorReadings", &Agent::getSensorReadings);
}
