import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

R = 4
dt = 0.01

m = 1
def force(m, R, t, dt):

#	fx = m * R * ((-4 * t **2 -1) * np.cos(t) * np.sin(t **2) + (2 * np.cos(t) - 4 * t * np.sin(t)) * np.cos(t**2))
#	fy = m * R * (-4 * np.cos(t) * np.sin(t))
#	fz = m  * (12 * t**2 * np.cos(t ** 3) - 9 * t ** 5 * np.sin(t ** 3))
	fx = m * R  * np.cos(t) * (-1)
	fy = m * R  * np.sin(t) * (-1)
	fz = m * R * 6  * t
	return  np.array([fx, fy, fz])

#def solution(m, R, t, dt):
#	x = R * np.cos(t) * np.sin(t**2)
#	y = R * np.sin(t) * np.cos(t)
#	z = R * dt * np.sin(t**3)
#	r = np.vstack(np.array([x,y,z]))
#	return r
def pos(R, dt, t):
#	x = R * np.cos(t) * np.sin(t**2)
#	y = R * np.sin(t) *  np.cos(t)
#	z = t * np.sin(t**3)
#	r = np.array([x,y,z]).reshape(len(t),3)
	x = R * np.cos(t) * m
	y = R * np.sin(t) * m
	z = R * t**3 * m
	return x,y,z

def riemann_l(dt, v1, pos1, m, R, t):
	force1 = force(m, R, t, dt)
	v = np.add(v1, force1/ m * dt)
	pos = np.add(pos1, v * dt)
	return v, pos
		

def riemann_r(dt, v1, pos1, m, R, t):
	force1 = force(m, R, t + dt, dt)
	v = np.add(v1, force1 /m * dt)
	pos = np.add(pos1, v * dt)
	return v, pos
	
def trap(dt,v1,pos1,m,R,t):
	vl, posl = riemann_l(dt, v1, pos1, m, R, t)
	vr, posr = riemann_r(dt, v1, pos1, m, R, t)
	v = np.add(vl,vr) / 2
	pos = np.add(posl, posr) / 2
	return v, pos

def simpson(dt, v1, pos1, m, R, t):
	force_a = force(m, R, t, dt)
	force_b = force(m, R, t + dt, dt)
	force_m = force(m, R, t + (dt / 2), dt)
	simpson = (1 / 6) * np.add(force_a, np.add(4 * force_m, force_b))
	v = np.add(v1, simpson / m * dt)
	pos = np.add(pos1, v * dt)	

	return v, pos











