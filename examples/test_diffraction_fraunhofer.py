"""Test forward Fresnel propagation and demonstrate its usage.
"""

__author__ = 'Andrew Michaels'
__email__ = 'amichaels@berkeley.edu'

import sys
sys.path.append('../')

from diffraction import prop_fraunhofer, prop_fraunhofer_inverse, phase_cmap
import numpy as np
from math import pi
import matplotlib.pyplot as plt

L = 1e-2
wlen = 1e-6
N = 1000

# Gaussian beam properties
w0 = 1e-4
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

z0 = 0
z1 = 2.0
E1 = E(r, z0)

# starting with E1, propagate over distance z1
E2_prop, L1 = prop_fraunhofer(E1, L, wlen, z1)

# calculate the theoretical fields using analytic equations
xff = np.linspace(-L1/2.0, L1/2.0, N)
Xff, Yff = np.meshgrid(xff, xff)
rff = np.sqrt(Xff**2 + Yff**2)

E2 = E(rff, z1)

# Calculate the total error in the propagated fields
err2 = np.linalg.norm(np.abs(E2_prop) - np.abs(E2)) / np.linalg.norm(np.abs(E2))
print('The error in the forward propagated field is %0.4E.' % err2)

# starting with E1, propagate over distance z1
E1_prop, L2 = prop_fraunhofer_inverse(E2_prop, L1, wlen, z1)

f = plt.figure()
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)

ax1.imshow(np.abs(E2_prop), extent=[0,L,0,L])
ax2.imshow(np.abs(E2), extent=[0,L,0,L])
plt.show()

# Calculate the total error in the propagated fields
err1 = np.linalg.norm(E1_prop - E1) / np.linalg.norm(E1)
print('The error in the inverse propagated field is %0.4E.' % err1)

