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
        .def(pybind11::init<double, double, double, double>(),
             pybind11::arg("sunMass"),
             pybind11::arg("sunDensity"),
             pybind11::arg("G"),
             pybind11::arg("deltaTime"))
        .def("initialize", &System::initialize)
        .def("step", &System::step)
        .def("addBody", &System::addBody,
             pybind11::arg("name"),
             pybind11::arg("mass"),
             pybind11::arg("density"),
             pybind11::arg("emitGravity"),
             pybind11::arg("orbital_radius"),
             pybind11::arg("initial_angle"),
             pybind11::arg("ellipsity"))
         .def("addMoon", &System::addMoon,     
             pybind11::arg("name"),
             pybind11::arg("hostName"),
             pybind11::arg("mass"),
             pybind11::arg("density"),
             pybind11::arg("emitGravity"),
             pybind11::arg("orbital_radius"),
             pybind11::arg("initial_angle"),
             pybind11::arg("ellipsity"))
        .def("getTotalEnergy", &System::getTotalEnergy)
        .def("getAllBodyProperties", &System::getAllBodyProperties)
        .def("drawGravityField", &System::drawGravityField,
             pybind11::arg("resolution"),
             pybind11::arg("radius"));

}
