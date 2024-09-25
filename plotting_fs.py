import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import frenet_serret as fs

r, t = fs.pos(4, 0.1)

T = fs.for_eul(r, t[-1]/len(t))
N = fs.for_eul(T, t[-1]/len(t))
B = fs.binorm(T, N)






for i in range(len(B[0])):
	ax = plt.figure().add_subplot(projection='3d')
	ax.plot(r[0,:],r[1,:],r[2,:])
	ax.plot(xs=[r[0][i],r[0][i]+T[0][i]],ys=[r[1][i],r[1][i]+T[1][i]],zs=[r[2][i],r[2][i]+T[2][i]], c = 'red')
	ax.plot(xs=[r[0][i],r[0][i]+N[0][i]],ys=[r[1][i],r[1][i]+N[1][i]],zs=[r[2][i],r[2][i]+N[2][i]], c = 'green')
	ax.plot(xs=[r[0][i],r[0][i]+B[0][i]],ys=[r[1][i],r[1][i]+B[1][i]],zs=[r[2][i],r[2][i]+B[2][i]], c = 'blue')
	plt.savefig(f"figs/evolve/{i}.png")
	plt.close()

images = [Image.open(f"figs/evolve/{n}.png") for n in range(len(B[0]))]

images[0].save('figs/vec.gif', save_all=True, append_images=images[1:], duration=1, loop=0)
