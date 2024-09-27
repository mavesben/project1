import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from plotly.subplots import make_subplots


R = 4
dt = 0.1

t = np.linspace(0, 6, 1000)

x = R * np.cos(t) * np.sin(t**2)
y = R * np.sin(t) * np.cos(t)
z = dt * t * np.sin(t**3)

r = np.array([x, y, z])
# print(r)

fig = make_subplots()
fig.add_scatter3d(x=x, y=y, z=z, mode="lines")


def pos(R, dt):
    t = np.linspace(0, 6, 1000)
    x = R * np.cos(t) * np.sin(t**2)
    y = R * np.sin(t) * np.cos(t)
    z = dt * t * np.sin(t**3)
    r = np.array([x, y, z])
    return r, t


def for_eul(r, step):
    x1 = np.array([])
    y1 = np.array([])
    z1 = np.array([])
    for i in range(len(r[1]) - 1):
        r0 = r[:, i]
        r1 = r[:, i + 1]
        drdt = (r1 - r0) / step
        norm = drdt / np.linalg.norm(drdt)
        x1 = np.append(x1, norm[0])
        y1 = np.append(y1, norm[1])
        z1 = np.append(z1, norm[2])
    l_fin = np.array([x1, y1, z1])
    return l_fin


def back_eul(r, step):
    x1 = np.array([])
    y1 = np.array([])
    z1 = np.array([])
    for i in range(len(r[1]) - 1):
        r0 = r[:, i]
        r1 = r[:, i + 1]
        r2 = r[:, i + 2]
        drdt = (r2 - r1) / step
        norm = drdt / np.linalg.norm(drdt)
        x1.append(x1, norm[0])
        y1.append(y1, norm[0])
        z1.append(z1, norm[0])
    T = np.array([x1, y1, z1])
    return T


def sympectic(r, step):
    T1 = for_eul(r, step)
    T2 = back_eul(r, step)
    T = np.add(T1, T2) / 2
    return T


def binorm(T, N):
    B1 = np.array([])
    B2 = np.array([])
    B3 = np.array([])
    for i in range(len(N[0])):
        Bt = np.cross(T[:, i], N[:, i])
        B1 = np.append(B1, Bt[0])
        B2 = np.append(B2, Bt[1])
        B3 = np.append(B3, Bt[2])
    B = np.array([B1, B2, B3])
    return B


R = 4
dt = 0.1
t = np.linspace(0, 6, 1000)

x = R * np.cos(t) * np.sin(t**2)
y = R * np.sin(t) * np.cos(t)
z = dt * t * np.sin(t**3)

r = np.array([x, y, z])

T = for_eul(r, t[-1] / len(t))
N = for_eul(T, t[-1] / len(t))

B = binorm(T, N)

fig = make_subplots()
fig.add_scatter3d(x=x, y=y, z=z, mode="lines", marker_color="black")
for i in range((len(B[0]))):
    fig.add_scatter3d(
        x=[r[0][i], r[0][i] + T[0][i]],
        y=[r[1][i], r[1][i] + T[1][i]],
        z=[r[2][i], r[2][i] + T[2][i]],
        mode="lines",
        marker_color="red",
    )
    fig.add_scatter3d(
        x=[r[0][i], r[0][i] + N[0][i]],
        y=[r[1][i], r[1][i] + N[1][i]],
        z=[r[2][i], r[2][i] + N[2][i]],
        mode="lines",
        marker_color="green",
    )
    fig.add_scatter3d(
        x=[r[0][i], r[0][i] + B[0][i]],
        y=[r[1][i], r[1][i] + B[1][i]],
        z=[r[2][i], r[2][i] + B[2][i]],
        mode="lines",
        marker_color="blue",
    )
fig.write_image("figs/fig1.png")


print(T, N, B)
