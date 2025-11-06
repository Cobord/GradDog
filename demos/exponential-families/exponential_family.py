"""
Exponential Family
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

import graddog as gd


# pylint:disable=too-few-public-methods
class BaseMeasure(ABC):
    """
    h(x) d\\mu_x
    """

    @abstractmethod
    def in_support(self, x: NDArray) -> bool:
        """
        Is this point in the support
        """


class BoxedContinuous(BaseMeasure):
    """
    h(x) d\\mu_x = dx
    in a box with a_i <= x_i <= b_i
    """

    def __init__(self, bounds: Iterable[Tuple[Optional[float], Optional[float]]]):
        super().__init__()
        self.bounds = bounds

    def in_support(self, x: NDArray):
        def in_bound(ai: Optional[float], xi: float, bi: Optional[float]) -> bool:
            match (ai, bi):
                case (None, None):
                    return True
                case (ai, None):
                    return ai <= xi
                case (None, bi):
                    return xi <= bi
                case (ai, bi):
                    return ai <= xi <= bi
                case _:
                    raise ValueError("Unreachable")

        return all((in_bound(ai, xi, bi) for xi, (ai, bi) in zip(x, self.bounds)))


class ExponentialFamily:
    """
    An exponential family with it's natural etas instead of thetas
    That is to say Canonical Form
    """

    # pylint:disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        num_etas: int,
        num_ts: int,
        t_function: Callable[[NDArray], NDArray],
        a_function: Callable[[NDArray], np.float64],
        dh: BaseMeasure,
        t_inv_function: Optional[Callable[[NDArray], NDArray]] = None,
    ):
        self.t_function = t_function
        self.t_inv_function = t_inv_function
        self.a_function = a_function
        self.num_etas = num_etas
        self.eta_params = np.random.rand(num_etas)
        self.dh = dh
        self.num_ts = num_ts

    def set_etas(self, new_etas: NDArray):
        """
        Set the current eta parameters
        """
        self.eta_params = new_etas

    def t_vars_exps_covariance(self) -> Tuple[NDArray, NDArray]:
        """
        Expectation values and covariances of the
        sufficient statistics T's
        """
        f_, f__ = gd.derivatives_and_hessians(
            f=self.a_function,
            seed=self.eta_params,
        )
        return (f_, f__)

    def in_support(self, x: NDArray) -> bool:
        """
        The support for the exponential family is the
        same as that of the base measure dh.
        """
        return self.dh.in_support(x)
