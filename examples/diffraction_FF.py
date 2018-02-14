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
d = diff.DiffractionFF(L, N, wlen)

# Setup the illumination pattern
# For this example, the grating is illuminated by a rect pattern in the center
d.u_in = np.zeros((N,N), dtype=np.complex128)
rect = slice(4*N/10, 6*N/10)
d.u_in[rect, rect] = 1.0

# Setup the grating
# The grating transmits the beam at an angle and expanded
d.grating = np.ones((N,N), dtype=np.complex128)
x = np.arange(0, L, dx)
X, Y = np.meshgrid(x,x)
R2 = (X-L/2)**2 + (Y-L/2)**2
k1 = 2*pi/1e-3
k2 = 2*pi/5e-5
d.grating *= np.exp(1j*k1**2*R2 + 1j*k2*X + 1j*k2*Y)

# Setup the ouput planes
d.z_out = [1.0, 2.0, 3.0]

# Propagate to the output planes
u_out, L_out = d.propagate()

# aggregate the input and outputs for plotting
u_plot = [d.u_in*d.grating] + u_out
L_plot = [L] + L_out

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
