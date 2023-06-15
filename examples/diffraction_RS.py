"""Demonstrate how to use the DiffractionFF class
"""


__author__ = 'Andrew Michaels'
__email__ = 'amichaels@berkeley.edu'

import sys
sys.path.append('../')

import diffraction as diff
import numpy as np
from math import pi
import matplotlib.pyplot as plt

L = 1e-2
wlen = 1e-6
N = 1000
dx = L/N

# Create the Diffraction object
d = diff.DiffractionRS(L, N, wlen)

# Setup the illumination pattern
# For this example, the grating is illuminated by a flat Gaussian beam
w0 = 2e-3
k = 2*pi/wlen
zR = pi*w0**2/wlen

w = lambda z : w0*np.sqrt(1 + (z/zR)**2)
invR = lambda z : z / (z**2 + zR**2)
psi = lambda z : np.arctan(z/zR)

# calculate gaussian beam in three different planes: z = -dz, z = 0, z = dz
E = lambda r,z : w0/w(z) * np.exp(-r**2/w(z)**2) * np.exp(-1j*(k*z + k*r**2/2.0*invR(z)-psi(z)))

x = np.linspace(-L/2,L/2,N)
y = np.linspace(-L/2,L/2,N)
X,Y = np.meshgrid(x,y)
r = np.sqrt(X**2 + Y**2)

d.u_in = E(r, 0)

# Setup the grating
# The grating focuses the beam
d.grating = np.ones((N,N), dtype=np.complex128)
kr = 2*pi/1e-3
d.grating *= np.exp(-1j*kr**2*r**2)

# Setup the ouput planes
dz = 4.1e-2
d.z_out = [dz, 2*dz, 3*dz]

# Propagate to the output planes
u_out = d.propagate()

# aggregate the input and outputs for plotting
u_plot = [d.u_in*d.grating] + u_out
L_plot = [L for _ in range(4)]

# plot the results
f = plt.figure()
for i in range(4):
    ax1 = f.add_subplot(2,4,i+1)
    ax2 = f.add_subplot(2,4,i+5)

    up = u_plot[i]
    Lp = L_plot[i]
    ax1.imshow(np.abs(up), extent=[0, Lp, 0, Lp], cmap='hot')

    mask = np.abs(up)/np.max(np.abs(up))
    im1 = ax2.imshow(np.angle(up)*mask, extent=[0, Lp, 0, Lp], vmin=-pi, vmax=pi, cmap=diff.phase_cmap())

plt.show()
