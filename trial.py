import numpy as np
import cent_pot as cp
import matplotlib.pyplot as plt

def soln(t, w):
	return np.array([np.sin(w*t), np.cos(w*t)])

def do_one(method):
	# set initial conditions
	z = np.zeros(2)
	z[1] = 1.0

	# period is 1.0, frequency is 2*pi, stepsize is 1/100 of a period
	P = 1.0
	w = 2.0*np.pi/P
	h = P/100.0

	# we'll integrate from t = 0 to t = t_f = 2*P
	t = 0.0
	t_f = 2*P
	# Number of steps
	N = int(t_f/h)

	# check everything!
	print('integrating from t = {0} to t = {1} with {2} steps; h = {3:5.3f}\n'.\
		format(t,t_f,N,h))
    
	# print every 5th line
	PRINT_INTERVAL = 25
	# counter for steps
	cnt = 0
	# format for outputing results
	fmt = '{0:5d}{1:7.3f}   {2:7.3f}{3:7.3f}{4:9.2e}   {5:7.3f}{6:7.3f}{7:9.2e}'
	# format for header
	head_fmt = '{0:>5s}{1:>7s}   {2:>7s}{3:>7s}{4:>9s}   {5:>7s}{6:>7s}{7:>9s}'
	print(head_fmt.format(
		'step','t','z0','s0','|z0-s0|','z1','s1','|z1-s1|')
	)
	stepper = method
	for step in range(N):
		z = stepper(cp.f,t,z,h,args=w)
		t += h
		cnt += 1
		if (cnt % PRINT_INTERVAL == 0):
			zs = soln(t,w)
			resid = np.abs(z-zs)
			print(fmt.format(cnt,t,z[0],zs[0],resid[0],z[1],zs[1],resid[1]))

#print('\n====================Forward Euler====================')
#do_one(cp.euler)
#print('\n================2nd order Runge-Kutta================')
#do_one(cp.rk2)
#print('\n================4th order Runge-Kutta================')
#do_one(cp.rk4)

def integrate_orbit(z0,m,tend, h, method = cp.rk4):
	t = 0
	z = z0

	Nsteps = int(tend/h) +1
	
	ts = np.zeros(Nsteps)
	Xs = np.zeros(Nsteps)
	Ys = np.zeros(Nsteps)
	Ts = np.zeros(Nsteps)
	Vs = np.zeros(Nsteps)
	TEs = np.zeros(Nsteps)

	ts[0] = t
	Xs[0] = z[0]
	Ys[0] = z[1]
	Ts[0] = cp.kinetic_energy(z0[2:4],m)
	Vs[0] = cp.potential_energy(z0[0:2],m)
	TEs[0] = cp.total_energy(z0, m)

	advance_one = method
	
	for step in range(1, Nsteps):
		z = advance_one(cp.derivs, t,z,h, args = m)

		t += h
		ts[step] = t
		Xs[step] = z[0]
		Ys[step] = z[1]
		Ts[step] = cp.kinetic_energy(z[2:4], m)
		Vs[step] = cp.potential_energy(z[0:2], m)
		TEs[step] = cp.total_energy(z, m)

		continue
	return ts, Xs, Ys, Ts, Vs, TEs


a=1
m=1
e=0
z0 = cp.initialize(a,m,e)
T = z0[2]
tend = 50
h0 = 0.1 * 3 * T
h_list = [h0, h0/2, h0/4, h0/8, h0/16, h0/32, h0/64, h0/128, h0/256, h0/512, h0/ 1024]
zz = z0[0]

euler_list = []
rk2_list = []
rk4_list = []

for i in h_list:
	euler_list.append(integrate_orbit(zz,m,tend,i,method=cp.euler))
	rk2_list.append(integrate_orbit(zz,m,tend,i,method=cp.rk2))
	rk4_list.append(integrate_orbit(zz,m,tend,i,method=cp.rk4))



labels = ["h_0", "h_0/2", "h_0/4", "h_0/8", "h_0/16", "h_0/32", "h_0/64", "h_0/128", "h_0/256","h_0/512", "h_0/1024"]


# Euler Method as normal
plt.figure(figsize=(12,12))

plt.subplot(3,2,1)
j = 0
for i in euler_list:
	error = abs(i[5]-cp.total_energy(zz,m))
	plt.plot(i[0],error,label=labels[j])
	j+=1
plt.legend()
plt.ylabel("error")
plt.xlabel("h")
plt.title("Euler Total Energy Error")

# Euler Method as log scale
plt.subplot(3,2,2)
j=0
for i in euler_list:
	error = abs(i[5]-cp.total_energy(zz,m))
	plt.plot(i[0],error,label=labels[j])
	j+=1
plt.yscale("log")
plt.legend()
plt.ylabel("error")
plt.xlabel("h")
plt.title("Euler Total Energy Error as Log")



# RK2 Method as normal
plt.subplot(3,2,3)
j = 0
for i in rk2_list:
	error = abs(i[5]-cp.total_energy(zz,m))
	plt.plot(i[0],error,label=labels[j])
	j+=1
plt.legend()
plt.ylabel("error")
plt.xlabel("h")
plt.title("RK2 Total Energy Error")

# RK2 Method as log scale
plt.subplot(3,2,4)
j=0
for i in rk2_list:
	error = abs(i[5]-cp.total_energy(zz,m))
	plt.plot(i[0],error,label=labels[j])
	j+=1
plt.yscale("log")
plt.legend()
plt.ylabel("error")
plt.xlabel("h")
plt.title("RK2 Total Energy Error as Log")


# RK4 Method as normal
plt.subplot(3,2,5)
j = 0
for i in rk4_list:
	error = abs(i[5]-cp.total_energy(zz,m))
	plt.plot(i[0],error,label=labels[j])
	j+=1
plt.legend()
plt.ylabel("error")
plt.xlabel("h")
plt.title("RK4 Total Energy Error")

# RK4 Method as log scale
plt.subplot(3,2,6)
j=0
for i in rk4_list:
	error = abs(i[5]-cp.total_energy(zz,m))
	plt.plot(i[0],error,label=labels[j])
	j+=1
plt.yscale("log")
plt.legend()
plt.ylabel("error")
plt.xlabel("h")
plt.title("RK4 Total Energy Error as Log")

plt.tight_layout()
plt.savefig("figs/orbit_energies.png")
