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
        self._step = step

    @property
    def sim(self):
        return self._sim

    @sim.setter
    def sim(self, new_sim):
        if(new_sim == None):
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
        self._sim.solve_forward()
        return self.calc_fom()


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
        pass

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
        numpy.ndarray
            The derivative of the figure of merit with respect to the output
            fields.
        """
        pass

    @abstractmethod
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
        pass
