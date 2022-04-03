#include <cmath>
#include "coords.h"

double coords_spherical::get_cos_of_angle_between(coords_spherical a, coords_spherical b) {
  return coords_spherical::product_dot(a, b) / (a.norm() * b.norm());
}

double coords_cartesian::get_cos_of_angle_between(coords_cartesian a, coords_cartesian b) {
  return coords_cartesian::product_dot(a, b) / (a.norm() * b.norm());
}

double coords_spherical::product_dot(coords_spherical a, coords_spherical b) {
  return a.r * b.r * (
    sin(a.theta) * sin(b.theta) * (sin(a.phi) * sin(b.phi) + cos(a.phi) * cos(b.phi)) +
    cos(a.theta) * cos(b.theta)
  );
}

double coords_cartesian::product_dot(coords_cartesian a, coords_cartesian b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double coords_cartesian::norm() const {
  return sqrt(coords_cartesian::product_dot(*this, *this));
}

double coords_spherical::norm() const {
  return sqrt(coords_spherical::product_dot(*this, *this));
}

coords_spherical coords_spherical::operator+(const coords_spherical& rhs) const {
  return (this->to_cartesian() + rhs.to_cartesian()).to_spherical();
}

coords_cartesian coords_cartesian::operator+(const coords_cartesian& rhs) const {
    return {
      this->x + rhs.x,
      this->y + rhs.y,
      this->z + rhs.z
    };
}

coords_spherical coords_spherical::operator-(const coords_spherical& rhs) const {
  return (this->to_cartesian() - rhs.to_cartesian()).to_spherical();
}

coords_cartesian coords_cartesian::operator-(const coords_cartesian& rhs) const {
    return {
      this->x - rhs.x,
      this->y - rhs.y,
      this->z - rhs.z
    };
}

coords_cartesian coords_spherical::to_cartesian() const {
  return {
    this->r * sin(this->theta) * cos(this->phi),
    this->r * sin(this->theta) * sin(this->phi),
    this->r * cos(this->theta)
  };
}

coords_spherical coords_cartesian::to_spherical() const {
  coords_spherical result;

  if (this->x == 0 && this->y == 0 && this->z == 0) {
    result.r = 0;
    result.theta = 0;
    result.phi = 0;
  } else {
    result.r = sqrt(pow(this->x, 2) + pow(this->y, 2) + pow(this->z, 2));
    result.theta = acos(this->z / result.r);
    result.phi = atan2(this->y, this->x);
  }

  return result;
}
