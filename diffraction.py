"""Propagate scalar fields forward and backwards with Fresnel diffraction.

In addition to providing the functionality needed to propagate a complex scalar field 
this module also allows you to propagate a field backwards (inverse propagation) and
to propagate a field using a transposed version of the discretized Fresnel diffraction
equations (useful for optimization).

Notes
-----
1. See "Computational Fourier Optics: A MATLAB Tutorial" by David G. Voelz for an example 
implementation of Fresnel propagation using the transfer function method.

2. Running this module as a script also demonstrates its usage.

Methods
-------
prop_Fresnel_TF(u1, L, wlen, z)
	Forward Fresnel propagation
inverse_prop_Fresnel_TF(u2, L, wlen, z)
	Inverse Fresnel propagation
transposed_prop_Fresnel_TF(u1, L, wlen, z)
	Transposed version of forward Fresnel propagation
phase_cmap()
	matplotlib colormap useful for plotting phase
"""

__author__ = 'Andrew Michaels'
__email__ = 'amichaels@berkeley.edu'

import numpy as np
from math import pi

def prop_Fresnel_TF(u1, L, wlen, z):
	"""Compute diffracted field using the Fresnel approximation of the Rayleigh-Sommerfeld equation.
	The Transfer Function approach is used.

	Parameters
	----------
	u1: numpy.array
		2D field in source plane
	L : float
		Length and Width of system
	wlen : float
		wavelength
	z : float
		distance over which field is propagated

	Returns
	-------
	numpy.array
		The propagated complex field at location z (relative to source plane)
	"""
	N,M = u1.shape

	dx = L/M
	k = 2 * pi / wlen

	# Define frequency domain
	fx = np.arange(-1/(2*dx), 1/(2*dx), 1/L)
	FX, FY = np.meshgrid(fx, fx)

	# Convolution kernel in frequency domain
	H = np.exp(1j*pi*wlen*z*(FX**2+FY**2))
	H = np.fft.fftshift(H)

	# Convolve by multiplying in frequency domain
	U1 = np.fft.fft2(np.fft.fftshift(u1));
	U2 = H*U1;
	u2 = np.fft.ifftshift(np.fft.ifft2(U2));

	return u2

def inverse_prop_Fresnel_TF(u2, L, wlen, z):
	"""Compute the inverse of the Fresnel propagation.

	Parameters
	----------
	u1 : numpy.array
		Field in source plane
	L : float
		Length and Width of system
	wlen : float
		Wavelength
	z : float
		Distance over which field is propagated

	Returns
	-------
	numpy.array
		The 2D propagated complex field at location z (relative to source plane)
	"""
	N,M = u2.shape

	dx = L/M
	k = 2 * pi / wlen

	# Fourier space
	fx = np.arange(-1/(2*dx), 1/(2*dx), 1/L)
	FX, FY = np.meshgrid(fx, fx)

	# convolution kernel
	H = np.exp(1j*pi*wlen*z*(FX**2+FY**2))
	H = np.fft.fftshift(H)

	# convolve
	U2 = np.fft.fft2(np.fft.fftshift(u2));
	U1 = U2/H;
	u1 = np.fft.ifftshift(np.fft.ifft2(U1));

	return u1

def transposed_prop_Fresnel_TF(u1, L, wlen, z):
	"""Perform a transposed Fresnel propagation.  This is needed for gradient calculations

	Notes
	-----
	It appears as if the transposed and normal equations produce the same result. 
	I think this makes some intuitive sense. Let's just leave this function here for 
	the sake of semantics and posterity.

	Parameters
	----------
	u1 : numpy.array
		Field in source plane
	L : float
		Length and Width of system
	wlen : float
		Wavelength
	z : float
		Distance over which field is propagated

	Returns
	-------
	numpy.array
		The 2D propagated complex field at location z (relative to source plane)
	"""
	N,M = u1.shape

	dx = L/M
	k = 2 * pi / wlen

	fx = np.arange(-1/(2*dx), 1/(2*dx), 1/L)
	FX, FY = np.meshgrid(fx, fx)

	H = np.exp(1j*pi*wlen*z*(FX**2+FY**2))
	H = np.fft.fftshift(H)

	U1 = np.fft.ifft2(np.fft.ifftshift(u1));
	U2 = H*U1;

	u2 = np.fft.fftshift(np.fft.fft2(U2));

	return u2

def phase_cmap():
	"""Define a matplotlib colormap that is useful for plotting phase.

	Define a new colormap for matplotlib which is useful for plotting phase. 
	This colormap follows a softened CMYK color palette and is cyclic such that 
	-pi and pi have the same color.

	Returns
	-------
	matplotlib.colors.LinearSegmentedColormap
		The matplotlib-compatible colormap. Returns None if matplotlib is not installed
	"""
	try:
		from matplotlib.colors import LinearSegmentedColormap
	except Exception as e:
		print(e)
		print('Matplotlib is not installed or not accessible')
		return None

	red = ( (0.0, 1.0, 1.0), (0.2, 179/255.0, 179/255.0), (0.4, 55/255.0, 55/255.0), \
			(0.6, 54/255.0, 54/255.0), (0.8, 1.0, 1.0), (1.0, 1.0, 1.0) )
	green = ( (0.0, 55/255.0, 55/255.0), (0.2, 126/255.0, 126/255.0), (0.4, 170/255.0, 170/255.0), \
              (0.6, 1.0, 1.0), (0.8, 243/255.0, 243/255.0), (1.0, 55/255.0, 55/255.0) )
	blue = ( (0.0, 68/255.0, 68/255.0), (0.2, 184/255.0, 184/255.0), (0.4, 1.0, 1.0), \
             (0.6, 151/255.0, 151/255.0), (0.8, 54/255.0, 54/255.0), (1.0, 68/255.0, 68/255.0) )

	cdict = {'red' : red, 'green' : green, 'blue' : blue}

	return LinearSegmentedColormap('PhaseCMap', cdict)

