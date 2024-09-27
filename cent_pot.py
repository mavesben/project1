import numpy as np
import matplotlib.pyplot as plt


def rk4(f, t, z, h, args=()):
    """
    fourth order Runga Kutta method
    inputs:
            function of dz/dt = f()
            t: time
            z: position and vecolity vecs concat
            h: timestep
    returns:
            z_new = z(t+h)
    """
    if not isinstance(args, tuple):
        args = (args,)
    k1 = f(t, z, *args)
    k2 = f(t + 0.5 * h, z + 0.5 * h * k1, *args)
    k3 = f(t + 0.5 * h, z + 0.5 * h * k2, *args)
    k4 = f(t + h, z + h * k3, *args)
    return z + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk2(f, t, z, h, args=()):
    """
    2nd order Runga Kutta method
    inputs:
            function of dz/dt = f()
            t: time
            z: position and vecolity vecs concat
            h: timestep
    returns:
            z_new = z(t+h)
    """
    if not isinstance(args, tuple):
        args = (args,)

    k1 = z + 0.5 * h * f(t, z, *args)
    return z + h * f(t + 0.5 * h, k1, *args)


def euler(f, t, z, h, args=()):
    """
    euler method
    inputs:
            function of dz/dt = f()
            t: time
            z: position and vecolity vecs concat
            h: timestep
    return:
            z_new = z(t+h)
    """
    if not isinstance(args, tuple):
        args = (args,)
    return z + h * f(t + h, z, *args)


def f(t, z, w):
    dzdt = np.zeros_like(z)
    dzdt[0] = w * z[1]
    dzdt[1] = -w * z[0]

    return dzdt


def kinetic_energy(v, m):
    return 0.5 * m * np.dot(v, v)


def potential_energy(x, m):
    r = np.linalg.norm(x)
    return -m / r


def total_energy(z, m):
    return potential_energy(z[0:2], m) + kinetic_energy(z[2:4], m)


def derivs(t, z, m):
    r = z[0:2]
    v = z[2:4]
    rad = np.linalg.norm(r)
    drdt = v
    dvdt = (-m / (rad**3)) * r
    dzdt = np.concatenate((drdt, dvdt))

    return dzdt


def initialize(a, m, e):
    eps0 = -m / (2 * a)
    Tperiod = np.sqrt((4 * a**3 * np.pi**2) / m)

    x0 = (1 + e) * a
    y0 = 0.0

    vx0 = 0.0
    vy0 = np.sqrt(2 * eps0 + 2 * m / x0)

    return np.array([x0, y0, vx0, vy0]), eps0, Tperiod


a = 1
e = 0
m = 1
t = 0
h = 0.1
w = 2 * np.pi
z, eps0, T = initialize(a, m, e)
x = rk2(f, t, z, h, args=w)
y = total_energy(x, m)
