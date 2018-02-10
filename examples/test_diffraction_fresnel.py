"""Test forward Fresnel propagation and demonstrate its usage.
"""

__author__ = 'Andrew Michaels'
__email__ = 'amichaels@berkeley.edu'

import sys
sys.path.append('../')

from diffraction import prop_Fresnel_TF

import numpy as np
from math import pi

L = 1e-3
dz = 3e-3
N = 1000

# Gaussian beam properties
wlen = 1e-6
w0 = 0.1e-4
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

E1 = E(r, -dz)
E2 = E(r, -dz+dz/10.0)
E3 = E(r, 0)

# starting with E1, propagate over distance dz and 2*dz and compare to theoretical fields
E2_prop = prop_Fresnel_TF(E1, L, wlen, dz/10.0)
E3_prop = prop_Fresnel_TF(E1, L, wlen, dz)

# Calculate the total error in the propagated fields
err2 = np.sum(np.abs(E2_prop - E2)**2) / np.sum(np.abs(E2)**2)
err3 = np.sum(np.abs(E3_prop - E3)**2) / np.sum(np.abs(E3)**2)

print('The error in the first propagated field is %0.4E.' % err2)
print('The error in the second propagated field is %0.4E.' % err3)
