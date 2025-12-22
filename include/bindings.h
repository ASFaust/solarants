#ifndef BINDINGS_H
#define BINDINGS_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "Vec2.h"

//we have to tell that py means pybind11
namespace py = pybind11;

namespace pybind11 { namespace detail {

template <>
struct type_caster<Vec2> {
public:
    PYBIND11_TYPE_CASTER(Vec2, _("Vec2"));

    // Python -> C++
    bool load(handle src, bool) {
        if (py::isinstance<py::tuple>(src) || py::isinstance<py::list>(src)) {
            py::sequence seq = py::reinterpret_borrow<py::sequence>(src);
            if (seq.size() != 2) return false;
            value = Vec2(
                seq[0].cast<double>(),
                seq[1].cast<double>()
            );
            return true;
        }

        if (py::isinstance<py::array>(src)) {
            py::array_t<double, py::array::c_style | py::array::forcecast> arr =
                py::reinterpret_borrow<
                    py::array_t<double, py::array::c_style | py::array::forcecast>
                >(src);

            if (arr.ndim() != 1 || arr.size() != 2)
                return false;

            auto r = arr.unchecked<1>();
            value = Vec2(r(0), r(1));
            return true;
        }

        return false;
    }

    // C++ -> Python
    static handle cast(const Vec2& v, return_value_policy, handle) {
        py::array_t<double> out(2);
        auto r = out.mutable_unchecked<1>();
        r(0) = v.x;
        r(1) = v.y;
        return out.release();
    }
};

}} // namespace pybind11::detail

#endif // BINDINGS_H