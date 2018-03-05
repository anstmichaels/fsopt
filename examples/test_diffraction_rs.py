"""Test forward Fresnel propagation and demonstrate its usage.
"""

__author__ = 'Andrew Michaels'
__email__ = 'amichaels@berkeley.edu'

import sys
sys.path.append('../')

from diffraction import prop_RS, prop_RS_inverse

import numpy as np
from math import pi
import matplotlib.pyplot as plt

L = 1e-2
N = 1000

# Gaussian beam properties
wlen = 1e-6
w0 = 1e-3
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

dz = 1e-2
E1 = E(r, 0)
E2 = E(r, dz)

# starting with E1, propagate over distance dz and 2*dz and compare to theoretical fields
E2_prop = prop_RS(E1, L, wlen, dz)

f = plt.figure()
ax = f.add_subplot(111)
ax.imshow(np.abs(E2_prop))
plt.show()

# Calculate the total error in the propagated fields
err2 = np.linalg.norm(E2_prop - E2) / np.linalg.norm(E2)
print('The error in the first propagated field is %0.4E.' % err2)

# starting with E1, propagate over distance z1
E1_prop = prop_RS_inverse(E2_prop, L, wlen, dz)

# Calculate the total error in the propagated fields
err1 = np.linalg.norm(E1_prop - E1) / np.linalg.norm(E1)
print('The error in the inverse propagated field is %0.4E.' % err1)

