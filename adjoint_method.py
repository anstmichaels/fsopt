"""Calculate the gradient of a figure of merit in free space optics design
problems.
"""
import diffraction as diff
import numpy as np
from abc import ABCMeta, abstractmethod

class AdjointMethod(object):
    """Define an interface and support functions for calculating the gradient
    of a figure of merit in a free space optics design problem involving one or
    more diffraction gratings.

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    """
    __metaclass__ = ABCMeta

    def __init__(self, sim):
        self._sim = sim

    @property
    def sim(self):
        return self._sim

    @sim.setter
    def sim(self, new_sim):
        if new_sim is None:
            raise ValueError('New simulation object cannot be None...')
        else:
            self._sim = new_sim

    def fom(self, params):
        """Calculate the figure of merit.

        Parameters
        ----------
        params : numpy.ndarray
            The current set of design parameters.

        Returns
        -------
        float
            The figure of merit.
        """
        self.update_system(self._sim, params)
        self._sim.propagate()
        return self.calc_fom(self._sim, params)


    def gradient(self, params):
        """Calculate the gradient of the figure of merit.

        Parameters
        ----------
        params : numpy.ndarray
            The current set of design parameters

        Returns
        -------
        numpy.ndarray
            The gradient of the figure of merit with respect to the design
            parameters
        """
        # Run a forward simulation
        self.update_system(self._sim, params)
        self._sim.propagate()

        # Calculate "adjoint sources"
        dFdu = self.calc_dFdu(self._sim, params)

        # Run adjoint propagations
        self._sim.u_in_adj = dFdu
        self._sim.adjoint_propagate()
        u_adj = self._sim.u_out_adj

        # Calculate gradient
        grating = self._sim.grating
        u_in = self._sim._u_in
        grad_v = self.calc_grad_v # explicit dependence on design variables
        gradient = np.sum(u_adj, axis=0) *1j * grating * u_in

        # final gradient needs same dimensions and ordering as params
        # numpy.ndarray.ravel accomplishes this.
        return gradient.ravel()

    def check_gradient(params, skip=1000, step=1e-8):
        """Check that the accuracy of the gradient computed using the adjoint
        method.

        Parameters
        ----------
        params : numpy.array
            The list of design parameters at which the gradient accuracy is
            checked.
        skip : int (optional)
            Compare the adjoint method gradient computed with respect to every
            N=skip parameter. If there are 10,000 parameters and skip=100, then
            the derivative with respect to 100 values will be checked.
            (default=10000)
        step : float (optional)
            The step size to use when computing the "true" derivatives using
            finite differences (default = 1e-8)

        Return
        ------
        float
            The approximate error in the gradient.
        """
        gradient_am = self.gradient(params)

        fom_i = self.fom(params)
        fom_f = np.zeros(params.shape)
        inds = np.arange(0, len(params), skip)
        for i in inds:
            ppert = np.copy(params)
            params[i] += step
            fom_f[i] = self.fom(params)

        gradient_fd = (fom_f[inds] - fom_i)/step
        error_tot = np.linalg.norm(gradient_fd-gradient_am[inds])/np.linalg.norm(gradient_fd)
        errors = np.abs(gradient_fd-gradient_am[inds]) / np.abs(gradient_fd)

        import matplotlib.pyplot as plt
        f= plt.figure()
        ax1 = f.add_subplot(311)
        ax2 = f.add_subplot(312)
        ax3 = f.add_subplot(313)

        ax1.plot(inds, gradient_fd, '.-', markersize=8)
        ax2.plot(inds, gradient_am[inds], '.-', markersize=8)
        ax3.plot(inds, errors, '.-', markersize=8)

        print('The total gradient error = %0.4E' % (error_tot))

        return error_tot

    @abstractmethod
    def update_system(self, sim, params):
        """Update the simulation system based on the current design variables.

        This function must be implemented based on the specific problem that
        you are trying to solve. It should provide a mapping between the design
        parameters and the physical properties of the system.

        Parameters
        ----------
        sim : fsopt.diffraction.Diffraction
            The simulation object
        params : numpy.ndarray
            The list of design parameters
        """
        pass

    @abstractmethod
    def calc_fom(self, sim, params):
        """Defines how the figure of merit is calculated.

        This method must be implemented according to each unique optimization
        problem.

        Parameters
        ----------
        sim : fsopt.diffraction.Diffraction
            The simulation object
        params : numpy.ndarray
            The list of design parameters

        Returns
        -------
        float
            The figure of merit.
        """
        pass

    @abstractmethod
    def calc_dFdu(self, sim, params):
        """Defines how the derivative of the figure of merit with respect to
        the output fields is calculated.

        This method must be implemented according to each unique optimization
        problem.

        Parameters
        ----------
        sim : fsopt.diffraction.Diffraction
            The simulation object
        params : numpy.ndarray
            The list of design parameters

        Returns
        -------
        list of numpy.ndarray
            The derivatives of the figure of merit with respect to the output
            fields.
        """
        pass

    def calc_grad_v(self, sim, params):
        """Defines how the derivative of the figure of merit with respect to
        the design parameters when the figure of merit has an explicit
        functional dependence on the design parameters.

        Figures of merit may also explicitly depend on the design parameters.
        In such cases, the partial derivatives with respect to the design
        parameters (holding the output fields constant) must be taken.

        Parameters
        ----------
        sim : fsopt.diffraction.Diffraction
            The simulation object
        params : numpy.ndarray
            The list of design parameters

        Returns
        -------
        numpy.ndarray
            The derivative of the figure of merit with respect to the design
            parameters
        """
        return np.zeros(params.shape, dtype=np.complex128)
