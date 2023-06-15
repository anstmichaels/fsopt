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

###############################################################################
# Core Diffraction Classes
###############################################################################

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
        self._grating = np.array([])
        self._u_out = []

        self._u_in_adj = []
        self._u_out_adj = []

        self._z_out = []
        self._Nz = 0

    @property
    def u_in(self):
        return self._u_in

    @u_in.setter
    def u_in(self, val):
        self._u_in = val

    @property
    def grating(self):
        return self._grating

    @grating.setter
    def grating(self, val):
        self._grating = val

    @property
    def u_out(self):
        return self._u_out

    @u_out.setter
    def u_out(self, value):
        raise AttributeError('Diffraction.u_out cannot be modified directly in' \
                             ' this way.')

    @property
    def u_out_adj(self):
        return self._u_out_adj

    @u_out_adj.setter
    def u_out_adj(self, value):
        raise AttributeError('Diffraction.u_out_adj cannot be modified directly in' \
                             ' this way.')

    @property
    def u_in_adj(self):
        return self._u_in_adj

    @u_in_adj.setter
    def u_in_adj(self, value):
        self._u_in_adj = value

    @property
    def Nz(self):
        return self._Nz

    @Nz.setter
    def Nz(self, value):
        raise AttributeError('Diffraction.Nz cannot be modified directly in' \
                             ' this way.')

    @property
    def z_out(self):
        return self._z_out

    @z_out.setter
    def z_out(self, val):
        Nz = len(val)
        self._Nz = Nz
        if (Nz == 0):
            raise ValueError('At least one output plane position must be' \
                                 ' specified.')
        self._z_out = val
        self._u_out = [
            np.zeros(self._uin.shape, dtype=np.complex128) for _ in range(len(val))
        ]

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
    def __init__(self, L, N, wavelength):
        super(DiffractionFF, self).__init__()

        self._L = L
        self._dx = L/N
        self._wlen = wavelength

        self._N = N
        self._u_in = np.zeros((N,N), dtype=np.complex128)
        self._grating = np.zeros((N,N), dtype=np.complex128)
        self._z_out = []
        self._u_out = []

        self._dx_out = []
        self._L_out = []

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, newN):
        self._N = newN

    @property
    def L_out(self):
        return self._L_out

    @L_out.setter
    def L_out(self, newL):
        raise AttributeError('DiffractionFF.L_out cannot be modified in this' \
                             ' way.')

    @property
    def z_out(self):
        return self._z_out

    @z_out.setter
    def z_out(self, val):
        Nz = len(val)
        self._Nz = Nz
        if (Nz == 0):
            raise ValueError('At least one output plane position must be' \
                                 ' specified.')
        self._z_out = val
        self._u_out = [
            np.zeros(self._u_in.shape, dtype=np.complex128) for _ in range(Nz)
        ]
        self._u_out_adj = [
            np.zeros(self._u_in.shape, dtype=np.complex128) for _ in range(Nz)
        ]
        self._dx_out = [0.0 for _ in range(Nz)]
        self._L_out = [0.0 for _ in range(Nz)]

    def propagate(self):
        """Propagate the input field through a diffraction grating to the
        output planes.

        Fraunhofer ("far field") diffraction is used to do the propagation. As
        a result, the lengths of the output planes depend on their distance
        from the input plane

        Returns
        -------
        list, list
            Two lists containing the ouput fields and the output plane sizes.
            These results can also be accessed through the DiffractionFF.u_out
            and DiffractionFF.L_out attributes. There is one output field for
            each output z position.
        """
        u0 = self._u_in * self._grating

        for i in range(self._Nz):
            uout, Lout = prop_fraunhofer(u0, self._L, self._wlen, self._z_out[i])
            self._u_out[i] = uout
            self._L_out[i] = Lout

        return self._u_out, self._L_out

    def adjoint_propagate(self):
        """Compute the adjoint fields by propagating the provided set of input
        adjoint field sources.

        Returns
        -------
        list
            The list containing the output adjoint fields. There is one output
            adjoint field for each input adjoint field.
        """
        for i in range(self._Nz):
            uout_adj, Lout_adj = prop_fraunhofer_adjoint(self._u_in_adj[i],
                                                         self._L_out[i],
                                                         self._wlen,
                                                         self._z_out[i])
            self._u_out_adj[i] = uout_adj

        return self._u_out_adj

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
    def __init__(self, L, N, wavelength):
        self._L = L
        self._dx = L/N
        self._wlen = wavelength

        self._N = N
        self._uin = np.zeros((N,N), dtype=np.complex128)
        self._z_out = []
        self._u_out = []

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, newN):
        self._N = newN

    def propagate(self):
        """Propagate the input field through a diffraction grating to the
        output planes.

        Rayleigh-Sommerfeld propagation is used. This is the most accurate
        method of scalar diffraction.

        Returns
        -------
        list
            The list of output scalar fields propagated from the input plane.
            This result can also be accessed via the DiffractionRS.u_out
            attribute. There is one output field for each z position.
        """
        u0 = self._u_in * self._grating

        for i in range(self._Nz):
            uout = prop_RS(u0, self._L, self._wlen, self._z_out[i])
            self._u_out[i] = uout

        return self._u_out

    def adjoint_propagate(self):
        """Compute the adjoint fields by propagating the provided set of input
        adjoint field sources.

        This is the adjoint of Rayleigh-Sommerfeld propagation.

        Returns
        -------
        list
            The list containing the output adjoint fields. There is one output
            adjoint field for each input adjoint field. This result can also be
            accessed via the DiffractionRS.u_out_adj attribute.
        """
        for i in range(self._Nz):
            uout_adj = prop_RS_adjoint(self._u_in_adj[i],
                                                 self._L,
                                                 self._wlen,
                                                 self._z_out[i])
            self._u_out_adj[i] = uout_adj

        return self._u_out_adj

###############################################################################
# Stand-Alone Diffraction Functions
###############################################################################

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
    FX = FX.astype(np.complex128)
    FY = FY.astype(np.complex128)

    # Convolution kernel in frequency domain
    H = np.exp(1j*k*z*np.sqrt(1-(wlen*FX)**2-(wlen*FY)**2))
    H = np.fft.fftshift(H)

    # Convolve by multiplying in frequency domain
    U1 = np.fft.fft2(np.fft.fftshift(u1));
    U2 = H*U1;
    return np.fft.ifftshift(np.fft.ifft2(U2))

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

    return np.fft.fftshift(np.fft.fft2(U2))

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
    return np.fft.ifftshift(np.fft.ifft2(U1))

def prop_kirchhoff(XS, YS, XF, YF, zf):
    pass

#####################################################################################
# Miscellaneous useful functions
#####################################################################################

def gaussian_beam(x, y, z, x0, y0, w0, wlen, M2=1.0):
    """Generate a slice of a Gaussian beam.

	The Guassian beam has the form

	..math:: A * \frac{w_0}{w(z_0)}e^{-\frac{ (x-x_0)^2 + (y-y_0)^2}{ w(z_0)^2}} e^{-i (k z_0 + k r^2 / 2 R(z_0) - \psi(z_0))

	Notes
	-----
	1. x and y must have the same shape

	2. The beam is assumed to propagate along the z direction.

	Parameters
	----------
	x : numpy.array
		The x coordinates where the Gaussian beam is evaluated
	y : numpy.array
		The y coordinates where the Gaussian beam is evaluated
	z : numpy.array
		The z position where the Gaussian beam is evaluated.
	x0 : float
		x coordinate of center
	y0 : float
		y coordinate of center
	w0 : float
		waist size of beam
	wlen : float
		wavelength of beam
	M2 : float
		M^2 value for modelling non-ideal beam

	Returns
	-------
	numpy.array
		Gaussian beam values at specified x,y coordinates
	"""
    k = 2*pi/wlen
    zR = pi*w0**2/wlen/M2

    w = lambda zz : w0*np.sqrt(1 + (zz/zR)**2)
    invR = lambda zz : z / (zz**2 + zR**2)
    psi = lambda zz : np.arctan(zz/zR)

    if(w0 == 0.0):
    	return np.zeros(x.shape)

    # calculate gaussian beam in three different planes: z = -dz, z = 0, z = dz
    E = lambda r,z : w0/w(z) * np.exp(-r**2/w(z)**2) * np.exp(-1j*(k*z + k*r**2/2.0*invR(z)-psi(z)))

    r = np.sqrt((x-x0)**2 + (y-y0)**2)

    return E(r, z)


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

