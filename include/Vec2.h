#ifndef SOLARANTS_VEC2_HPP
#define SOLARANTS_VEC2_HPP

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

struct Vec2 {
    double x;
    double y;

    constexpr Vec2() : x(0.0), y(0.0) {}
    constexpr Vec2(double x_, double y_) : x(x_), y(y_) {}

    // Basic arithmetic
    constexpr Vec2 operator+(const Vec2& v) const { return {x + v.x, y + v.y}; }
    constexpr Vec2 operator-(const Vec2& v) const { return {x - v.x, y - v.y}; }
    constexpr Vec2 operator*(double s) const { return {x * s, y * s}; }
    constexpr Vec2 operator/(double s) const { return {x / s, y / s}; }

    constexpr Vec2& operator+=(const Vec2& v) { x += v.x; y += v.y; return *this; }
    constexpr Vec2& operator-=(const Vec2& v) { x -= v.x; y -= v.y; return *this; }
    constexpr Vec2& operator*=(double s) { x *= s; y *= s; return *this; }
    constexpr Vec2& operator/=(double s) { x /= s; y /= s; return *this; }

    // Dot product
    constexpr double dot(const Vec2& v) const {
        return x * v.x + y * v.y;
    }

    // Squared magnitude (cheap, preferred)
    constexpr double norm2() const {
        return x * x + y * y;
    }

    // Magnitude (expensive)
    double norm() const {
        return std::sqrt(norm2());
    }

    // Safe normalization
    Vec2 normalized(double eps = 1e-12) const {
        double n2 = norm2();
        if (n2 < eps) return {0.0, 0.0};
        double inv = 1.0 / std::sqrt(n2);
        return {x * inv, y * inv};
    }

    inline void checkHealth(const std::string& context) const {
        std::string err;
        if (std::isnan(x)) err += "Vec2 x is NaN. ";
        if (std::isnan(y)) err += "Vec2 y is NaN. ";
        if (std::isinf(x)) err += "Vec2 x is Inf. ";
        if (std::isinf(y)) err += "Vec2 y is Inf. ";
        if (!err.empty()) {
            throw std::runtime_error(std::string("Vec2 health check failed (") + context + std::string("): ") + err);
        }
    }
};

// Scalar * vector
constexpr inline Vec2 operator*(double s, const Vec2& v) {
    return {s * v.x, s * v.y};
}

#endif
