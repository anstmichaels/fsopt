"""Demonstrate how to run an optimization with Rayleigh-Sommerfeld diffraction.

In this example, we optimize the phase of a grating to focus light on a desired
point. We do this by maximizing (minimizing the negative) the intensity at a
point in the grid.
"""

__author__ = 'Andrew Michaels'
__email__ = 'amichaels@berkeley.edu'

import sys
sys.path.append('../')

import diffraction as diff
import adjoint_method as am
import numpy as np
from math import pi
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class AMFocuser(am.AdjointMethod):

    def __init__(self, sim):
        super(AMFocuser, self).__init__(sim)

        self.x = int(0.25*sim.N)
        self.y = int(0.75*sim.N)

    def update_system(self, sim, params):
        N = sim.N
        phase = np.reshape(params, (N,N))
        sim.grating = np.exp(1j*phase)

    def calc_fom(self, sim, params):
        u_out = sim.u_out[0]

        i = self.y
        j = self.x

        return np.real(-1*u_out[i,j]*np.conj(u_out[i,j]))

    def calc_dFdu(self, sim, params):
        u_out = sim.u_out[0]
        N = sim.N

        i = self.y
        j = self.x

        dFdu = np.zeros((N,N), dtype=np.complex128)
        dFdu[i,j] = -1*np.conj(u_out[i,j])

L = 1e-2
wlen = 1e-6
N = 1000
dx = L/N

# Create the Diffraction object
d = diff.DiffractionRS(L, N, wlen)

# Setup the illumination pattern
# For this example, the grating is illuminated by a Gaussian beam
x = np.linspace(0, L, N)
X,Y = np.meshgrid(x,x)
d.u_in = diff.gaussian_beam(X, Y, 0, L/2, L/2, L/2, wlen)

# Setup the ouput planes
d.z_out = [1.0]

am = AMFocuser(d)
params = np.zeros(N**2)

am.check_gradient(param, skip=100000)

#result = minimize(am.fom, params, method='L-BFGS-B', jac=am.gradient)
