import numpy as np
import matplotlib.pyplot as plt
import frenet_serret2 as fs
from plotly.subplots import make_subplots

m = 1
R = 4
dt = 0.01
t = np.arange(0, 6, dt)
v_rl = np.array([0, 0, 0])
pos_rl = np.array([R, 0, 0])
v = []
pos = []
for i in range(len(t)):
    v_rl, pos_rl = fs.riemann_l(dt, v_rl, pos_rl, m, R, t[i])
    v.append(v_rl)
    pos.append(pos_rl)

v_rr = np.array([0, 0, 0])
pos_rr = np.array([R, 0, 0])
v1 = []
pos1 = []
for i in range(len(t)):
    v_rr, pos_rr = fs.riemann_r(dt, v_rr, pos_rr, m, R, t[i])
    v1.append(v_rr)
    pos1.append(pos_rr)

v_trap = np.array([0, 0, 0])
pos_trap = np.array([R, 0, 0])
v2 = []
pos2 = []
for i in range(len(t)):
    v_trap, pos_trap = fs.trap(dt, v_trap, pos_trap, m, R, t[i])
    v2.append(v_trap)
    pos2.append(pos_trap)

v_simp = np.array([0, 0, 0])
pos_simp = np.array([R, 0, 0])
v3 = []
pos3 = []
for i in range(len(t)):
    v_simp, pos_simp = fs.simpson(dt, v_simp, pos_simp, m, R, t[i])
    v3.append(v_simp)
    pos3.append(pos_simp)
    print(v_simp, pos_simp)
v = np.vstack(v)
pos = np.vstack(pos)
v1 = np.vstack(v1)
pos1 = np.vstack(pos1)
v2 = np.vstack(v2)
pos2 = np.vstack(pos2)
v3 = np.vstack(v3)
pos3 = np.vstack(pos3)

force1 = fs.force(m, R, t, dt)
force1 = np.swapaxes(force1, 0, 1)

fig = make_subplots()
fig.add_scatter3d(
    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines", marker_color="black"
)

B = np.empty_like(v)
vx = np.empty_like(v)
force1x = np.empty_like(force1)
for i in range(len(v)):
    vx[i] = v[i] / np.linalg.norm(v[i])
    force1x[i] = force1[i] / np.linalg.norm(force1[i])
    B[i] = np.cross(vx[i], force1[i])

for i in range(len(t)):
    fig.add_scatter3d(
        x=[pos[i, 0], pos[i, 0] + vx[i, 0]],
        y=[pos[i, 1], pos[i, 1] + vx[i, 1]],
        z=[pos[i, 2], pos[i, 2] + vx[i, 2]],
        mode="lines",
        marker_color="red",
    )
    fig.add_scatter3d(
        x=[pos[i, 0], pos[i, 0] + force1x[i, 0]],
        y=[pos[i, 1], pos[i, 1] + force1x[i, 1]],
        z=[pos[i, 2], pos[i, 2] + force1x[i, 2]],
        mode="lines",
        marker_color="blue",
    )
    fig.add_scatter3d(
        x=[pos[i, 0], pos[i, 0] + B[i, 0]],
        y=[pos[i, 1], pos[i, 1] + B[i, 1]],
        z=[pos[i, 2], pos[i, 2] + B[i, 2]],
        mode="lines",
        marker_color="green",
    )
fig.write_image("fig1.png")


anax, anay, anaz = fs.pos(R, dt, t)
s_int_pos = []
ssumx, ssumy, ssumz = 0, 0, 0
for i in range(len(pos)):
    ssumx += anax[i]
    ssumy += anay[i]
    ssumz += anaz[i]
    sum_tot = ssumx + ssumy + ssumz
    s_int_pos.append(sum_tot)


plt.figure(figsize=(9, 9))
plt.subplot(3, 1, 1)
int_pos = []
int_v = []
sumx, sumy, sumz, sumvx, sumvy, sumvz = 0, 0, 0, 0, 0, 0
for i in range(len(pos)):
    sumx += pos[i, 0]
    sumy += pos[i, 1]
    sumz += pos[i, 2]
    sum_tot = sumx + sumy + sumz
    int_pos.append(sum_tot)
    sumvx += v[i, 0]
    sumvy += v[i, 1]
    sumvz += v[i, 2]
    sumv_tot = sumvx + sumvy + sumvz
    int_v.append(sumv_tot)
plt.plot(t, int_pos, label="left riemann position integral")

int_pos1 = []
int_v1 = []
sumx1, sumy1, sumz1, sumvx1, sumvy1, sumvz1 = 0, 0, 0, 0, 0, 0
for i in range(len(pos1)):
    sumx1 += pos1[i, 0]
    sumy1 += pos1[i, 1]
    sumz1 += pos1[i, 2]
    sum_tot1 = sumx1 + sumy1 + sumz1
    int_pos1.append(sum_tot1)
    sumvx1 += v1[i, 0]
    sumvy1 += v1[i, 1]
    sumvz1 += v1[i, 2]
    sumv_tot1 = sumvx1 + sumvy1 + sumvz1
    int_v1.append(sumv_tot1)
plt.plot(t, int_pos1, label="right riemann position integral")


int_pos2 = []
int_v2 = []
sumx2, sumy2, sumz2, sumvx2, sumvy2, sumvz2 = 0, 0, 0, 0, 0, 0
for i in range(len(pos2)):
    sumx2 += pos2[i, 0]
    sumy2 += pos2[i, 1]
    sumz2 += pos2[i, 2]
    sum_tot2 = sumx2 + sumy2 + sumz2
    int_pos2.append(sum_tot2)
    sumvx2 += v2[i, 0]
    sumvy2 += v2[i, 1]
    sumvz2 += v2[i, 2]
    sumv_tot2 = sumvx2 + sumvy2 + sumvz2
    int_v2.append(sumv_tot2)
plt.plot(t, int_pos2, label="trapezoidal position integral")

int_pos3 = []
int_v3 = []
sumx3, sumy3, sumz3, sumvx3, sumvy3, sumvz3 = 0, 0, 0, 0, 0, 0
for i in range(len(pos3)):
    sumx3 += pos3[i, 0]
    sumy3 += pos3[i, 1]
    sumz3 += pos3[i, 2]
    sum_tot3 = sumx3 + sumy3 + sumz3
    int_pos3.append(sum_tot3)
    sumvx3 += v3[i, 0]
    sumvy3 += v3[i, 1]
    sumvz3 += v3[i, 2]
    sumv_tot3 = sumvx3 + sumvy3 + sumvz3
    int_v3.append(sumv_tot3)
plt.plot(t, int_pos3, label="simpson position integral")
plt.plot(t, s_int_pos, label="analytic")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, int_v, label="left riemann velocity integral")
plt.plot(t, int_v1, label="right riemann velocity integral")
plt.plot(t, int_v2, label="trampezoidal velocity integral")
plt.plot(t, int_v3, label="simpson velocity integral")
plt.legend()
plt.subplot(3, 1, 3)

print(np.shape(s_int_pos), np.shape(int_pos))

plt.plot(t, np.subtract(s_int_pos, int_pos), label="left riemann error")
plt.plot(t, np.subtract(s_int_pos, int_pos1), label="right riemann error")
plt.plot(t, np.subtract(s_int_pos, int_pos2), label="trapezoidal error")
plt.plot(t, np.subtract(s_int_pos, int_pos3), label="simpson error")
plt.legend()
plt.savefig("figs/disc_int.png")
