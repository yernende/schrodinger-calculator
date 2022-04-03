import numpy as np
import math

class Vector:
    def __init__(self, cartesian):
        x, y, z = cartesian

        if x == 0 and y == 0 and z == 0:
            module, theta, phi = 0.0, 0.0, 0.0
        else:
            module = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            theta = math.acos(z / module)
            phi = math.atan2(y, x)

        self.cartesian = np.array(cartesian)
        self.module = module
        self.theta = theta
        self.phi = phi

    def __add__(self, o):
        return Vector(self.cartesian + o.cartesian)

    def __sub__(self, o):
        return Vector(self.cartesian - o.cartesian)

    @staticmethod
    def product_dot(a, b):
        return Vector(np.dot(a.cartesian, b.cartesian))

    @staticmethod
    def get_cos_of_angle_between(a, b):
        return np.dot(a.cartesian, b.cartesian) / (a.module * b.module)

    @staticmethod
    def get_distance_between(a, b):
        return np.linalg.norm(a.cartesian - b.cartesian)
