import numpy as np
import matplotlib.pyplot as plt


def initialize(r, cx, cy, nx, ny, dx, dy, Tcool, Thot):
	u0 = Tcool * np.ones((nx, ny))
	u = np.empty((nx, ny))
	r2 = r**2
	for i in range(nx):
		for j in range(ny):
			p2 = (i*dx-cx)**2 + (j*dy-cy)**2
			if p2 < r2:
				u0[i,j] = Thot
	return u0, u


def diffuse(u0, u, dt, dx2, dy2):
	# Propagate with forward-difference in time, central-difference in space
	u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * ((u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
											   + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2)
	u0 = u.copy()
	return u0, u


def simulate_data(nsteps, noise_std=0.1, w=10, h=10, dx=0.1, dy=0.1, D=4., Tcool=300, Thot=700):
	data = []
	nx, ny = int(w/dx), int(h/dy)
	dx2, dy2 = dx*dx, dy*dy
	dt = dx2 * dy2 / (2 * D * (dx2 + dy2))
	u0, u = initialize(r, cx, cy, nx, ny, dx, dy, Tcool, Thot)
	noise = np.random.normal(0, noise_std, (nx, ny))
	data.append(u0 + noise)
	for m in range(nsteps):
		u0, u = diffuse(u0, u, dt, dx2, dy2)
		noise = np.random.normal(0, noise_std, (nx, ny))
		data.append(u + noise)
	return data




# plate size
w = h = 10.
# intervals in x-, y- directions, mm
dx = dy = 0.1
# Thermal diffusivity of steel, mm2.s-1
D = 4.

Tcool, Thot = 300, 700

# Initial conditions - ring of inner radius r, width dr centred at (cx,cy) (mm)
r, cx, cy = 2, 5, 5

nsteps = 101
noise_std = 10

data = simulate_data(nsteps, noise_std, w, h, dx, dy, D, Tcool, Thot)
