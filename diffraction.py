"""Propagate scalar fields forward and backwards with either
Rayleigh-Sommerfeld diffraction or Fraunhofer diffraction.

In addition to providing the functionality needed to propagate a complex scalar field
this module also allows you to propagate a field backwards (inverse propagation) and
to propagate a field using a transposed version of the discretized
Rayleigh-Sommerfeld diffraction equations (useful for optimization).

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
from abc import ABCMeta, abstractmethod

class Diffraction(object):
    """Define a standard interface for implementing diffraction simulators.

    Currentently, this interface supports propagating a single input field to
    multiple output planes. The input field, output plane positions, and output
    fields are exposed.

    Notes
    -----
    The dimensions of the input and output fields should be the same.

    Attributes
    ----------
    u_in : numpy.ndarray
        The input field.
    u_out : list numpy.ndarray
        The list of numpy.ndarrays containing the output fields.
    z_out : list of float
        The list of z positions of the output planes.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._u_in = np.array([])
        self._u_out = []
        self._z_out = []

    @property
    def u_in(self):
        return self._u_in

    @u_in.setter
    def u_in(self, val):
        self._u_in = val

    @property
    def u_out(self):
        return self._u_out

    @u_out.setter
    def u_out(self, value):
        raise AttributeError('Diffraction.u_out cannot be modified directly in' \
                             ' this way.')

    @property
    def z_out(self):
        return self._z_out

    @z_out.setter
    def z_out(self, val):
        if(len(val) == 0):
            raise ValueError('At least one output plane position must be' \
                             ' specified.')
        else:
            self._z_out = val

    @abstractmethod
    def propagate(self):
        pass

    @abstractmethod
    def adjoint_propagate(self):
        pass

class DiffractionFF(Diffraction):
    """Simulate the illumination of a diffraction grating using Fraunhofer (far
    field) diffraction.

    The 'simulation' involves illuminating a amplitude/phase grating with a
    specified input scalar field and propagating the modified scalar field to
    one or more output planes.

    Notes
    -----
    Currently this class only supports square scalar fields.

    Attributes
    ----------
    L : float
        The width or height of the input scalar field
    dx : float
        The grid spacing of the input scalar field
    wavelength : float
        The wavelength of light.
    N : int
        The number of pixels along x or y
    u_in : numpy.ndarray
        The input field.
    u_out : list numpy.ndarray
        The list of numpy.ndarrays containing the output fields.
    z_out : list of float
        The list of z positions of the output planes.
    dx_out : list of float
        The list of grid spacings in the output planes
    L_out : list of float
        The list of widths of the output planes
    """
    def __init__(self, L, dx, wavelength):
        self._L = L
        self._dx = dx
        self._wlen = wavelength

        self._N = int(L/dx)
        self._uin = np.zeros(N,N, dtype=np.complex128)
        self._z_out = []
        self._u_out = []

        self._dx_out = []
        self._L_out = []

    def propagate(self):
        pass

    def adjoint_propagate(self):
        pass

class DiffractionRS(Diffraction):
    """Simulate the illumination of a diffraction grating using
    Rayleigh-Sommerfeld propagation.

    The 'simulation' involves illuminating a amplitude/phase grating with a
    specified input scalar field and propagating the modified scalar field to
    one or more output planes.

    Notes
    -----
    Currently this class only supports square scalar fields.

    A key limitation of the Rayleigh-Sommerfeld method is that all output
    planes must have the same size as the input plane.

    Attributes
    ----------
    L : float
        The width or height of the input scalar field
    dx : float
        The grid spacing of the input scalar field
    wavelength : float
        The wavelength of light.
    N : int
        The number of pixels along x or y
    u_in : numpy.ndarray
        The input field.
    u_out : list numpy.ndarray
        The list of numpy.ndarrays containing the output fields.
    z_out : list of float
        The list of z positions of the output planes.
    """
    def __init__(self, L, dx, wavelength):
        self._L = L
        self._dx = dx
        self._wlen = wavelength

        self._N = int(L/dx)
        self._uin = np.zeros(N,N, dtype=np.complex128)
        self._z_out = []
        self._u_out = []

    def propagate(self):
        pass

    def adjoint_propagate(self):
        pass

def prop_fraunhofer(us, L, wlen, z):
    """Compute diffracted field using the Fraunhofer diffraction.

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
    # Source plane parameters
    M,N = us.shape
    dxs = L/M

    k = 2*pi/wlen

    Lff = wlen * z / dxs
    dxff = wlen * z / L

    xff = np.arange(-Lff/2.0, Lff/2.0, dxff)
    Xff, Yff = np.meshgrid(xff, xff)

    # do the propagation
    c = np.exp(1j*k*z)/(1j * wlen * z) * np.exp(1j*k/(2*z) * (Xff**2+Yff**2))
    uff = c * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(us))) * dxs**2

    return uff, Lff

def prop_fraunhofer_adjoint(us, L, wlen, z):
    """Compute the adjoint of Fraunhofer diffraction.

    This is the same as normal Fraunhofer propagation. It is included for
    semantic purposes.

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
    # Source plane parameters
    M,N = us.shape
    dxs = L/M

    k = 2*pi/wlen

    Lff = wlen * z / dxs
    dxff = wlen * z / L

    xff = np.arange(-Lff/2.0, Lff/2.0, dxff)
    Xff, Yff = np.meshgrid(xff, xff)

    # do the propagation
    c = np.exp(1j*k*z)/(1j * wlen * z) * np.exp(1j*k/(2*z) * (Xff**2+Yff**2))
    uff = c * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(us))) * dxs**2

    return uff, Lff

def prop_fraunhofer_inverse(us, L, wlen, z):
    """Compute the inverse of Fraunhofer diffraction.

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
    # Source plane parameters
    M,N = us.shape
    k = 2*pi/wlen

    dx = L/M
    dxnf = wlen * z / L
    Lnf = dxnf*M

    xff = np.arange(-L/2.0, L/2.0, dx)
    Xff, Yff = np.meshgrid(xff, xff)

    # do the propagation
    c = np.exp(1j*k*z)/(1j * wlen * z) * np.exp(1j*k/(2*z) * (Xff**2+Yff**2))
    unf = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(us / dxnf**2 / c)))

    return unf, Lnf

def prop_RS(u1, L, wlen, z):
    """Compute diffracted field with the Rayleigh-Sommerfeld propagation using
    the transfer function.

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
    M,N = u1.shape

    dx = L/M
    k = 2 * pi / wlen

    # Define frequency domain
    fx = np.arange(-1/(2*dx), 1/(2*dx), 1.0/L)
    FX, FY = np.meshgrid(fx, fx)

    # Convolution kernel in frequency domain
    H = np.exp(1j*k*z*np.sqrt(1-(wlen*FX)**2-(wlen*FY)**2))
    H = np.fft.fftshift(H)

    # Convolve by multiplying in frequency domain
    U1 = np.fft.fft2(np.fft.fftshift(u1));
    U2 = H*U1;
    u2 = np.fft.ifftshift(np.fft.ifft2(U2));

    return u2

def prop_RS_adjoint(u1, L, wlen, z):
    """Perform a transposed Rayleighy-Sommerfeld propagation.  This is needed for gradient calculations

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

    H = np.exp(1j*k*z*np.sqrt(1-(wlen*FX)**2-(wlen*FY)**2))
    H = np.fft.fftshift(H)

    U1 = np.fft.ifft2(np.fft.ifftshift(u1));
    U2 = H*U1;

    u2 = np.fft.fftshift(np.fft.fft2(U2));

    return u2

def prop_RS_inverse(u2, L, wlen, z):
    """Compute the inverse of the Rayleigh-Sommerfeld propagation.

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
    H = np.exp(1j*k*z*np.sqrt(1-(wlen*FX)**2-(wlen*FY)**2))
    H = np.fft.fftshift(H)

    # convolve
    U2 = np.fft.fft2(np.fft.fftshift(u2));
    U1 = U2/H;
    u1 = np.fft.ifftshift(np.fft.ifft2(U1));

    return u1

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

