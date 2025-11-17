"""
Exponential Family
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numbers
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast
import numpy as np
from numpy.typing import NDArray

import graddog as gd
from graddog.functions import PossibleArgument
from graddog.trace import Trace

from .observed_information import FDistribution


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

    @abstractmethod
    def is_restricted_lebesgue(self, allow_constant: bool) -> bool:
        """
        is this just restricting Lebesgue measure (or possibly an overall scaling thereof)
        """

    @abstractmethod
    def restricted_lebesgue_scaling(self) -> Optional[float]:
        """
        if it is just restricting Lebesgue measure but with a scaling factor
        what is that scaling factor
        """

    @abstractmethod
    def rescaled(self, extra_scaling_factor: float) -> BaseMeasure:
        """
        Apply an extra nonzero scaling factor to this measure
        usually used to normalize
        """


class BoxedContinuous(BaseMeasure):
    """
    h(x) d\\mu_x = A dx
    in a box with a_i <= x_i <= b_i
    """

    def __init__(
        self,
        bounds: Iterable[Tuple[Optional[float], Optional[float]]],
        scale_factor: float,
    ):
        super().__init__()
        self.bounds = bounds
        self.scale_factor = scale_factor

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

    def is_restricted_lebesgue(self, allow_constant: bool) -> bool:
        if allow_constant:
            return True
        return self.scale_factor == 1.0

    def restricted_lebesgue_scaling(self) -> float:
        return self.scale_factor

    def rescaled(self, extra_scaling_factor: float) -> BaseMeasure:
        return BoxedContinuous(
            list(self.bounds), self.scale_factor * extra_scaling_factor
        )


class HTimesMu(BaseMeasure):
    """
    It is absolutely continuous with respect to mu
    and multiplied by the nonnegative valued function h
    """

    mu: BaseMeasure
    h: Callable[[NDArray], float]

    def __init__(self, mu, h):
        self.mu = mu
        self.h = h

    def in_support(self, x: NDArray) -> bool:
        return self.mu.in_support(x)

    def is_restricted_lebesgue(self, allow_constant: bool) -> bool:
        return False

    def restricted_lebesgue_scaling(self) -> None:
        return None

    def rescaled(self, extra_scaling_factor: float) -> BaseMeasure:
        return HTimesMu(self.mu.rescaled(extra_scaling_factor), self.h)


def one_gaussian_base_measure(mu: float, sigma: float) -> HTimesMu:
    """
    A gaussian measure on R with prescribed mu and sigma
    """
    return HTimesMu(
        mu=BoxedContinuous([(None, None)], 1 / np.sqrt(2 * np.pi) * 1 / sigma),
        h=lambda x: np.exp(-((x - mu) * (x - mu) / (2 * sigma * sigma))),
    )


class ExponentialFamily(FDistribution):
    """
    An exponential family with it's natural etas instead of thetas
    That is to say Canonical Form
    """

    # pylint:disable=too-many-arguments, too-many-positional-arguments
    def __init__(
        self,
        *,
        num_etas: int,
        num_xs: int,
        t_function: Callable[[NDArray], NDArray],
        a_function: Callable[[NDArray], np.float64],
        dh: BaseMeasure,
        t_inv_function: Optional[Callable[[NDArray], NDArray]] = None,
    ):
        self._t_function = t_function
        self._t_inv_function = t_inv_function
        self._a_function = a_function
        self._num_etas = num_etas
        self._eta_params = np.random.rand(num_etas)
        self._dh = dh
        self._num_xs = num_xs

    def set_etas(self, new_etas: NDArray):
        """
        Set the current eta parameters
        """
        if len(new_etas) != self._num_etas:
            raise TypeError("This is not the right number of eta parameters")
        self._eta_params = new_etas

    def t_vars_exps_covariance(self) -> Tuple[NDArray, NDArray]:
        """
        Expectation values and covariances of the
        sufficient statistics T's
        """
        f_, f__ = gd.derivatives_and_hessians(
            f=self._a_function,
            seed=self._eta_params,
        )
        return (f_, f__)

    def in_support(self, x: NDArray) -> bool:
        """
        The support for the exponential family is the
        same as that of the base measure dh.
        """
        return self._dh.in_support(x)

    def t_function(
        self, x: PossibleArgument
    ) -> Tuple[Union[List[PossibleArgument], NDArray], bool]:
        """
        The sufficient statistic Ts in terms of the xs
        """
        raise NotImplementedError

    def a_function(self, thetas: NDArray) -> np.float64:
        """
        the A function which does normalization in a parameter dependent manner
        """
        return self._a_function(thetas)

    @property
    def f_native(self) -> bool:
        return False

    def f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        # pylint:disable=useless-parent-delegation
        return super().f_function(x_i, theta)

    # pylint:disable=too-many-branches
    def log_f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        if not self._dh.is_restricted_lebesgue(allow_constant=True):
            raise TypeError(
                """This kind of exponenential family is not (manifestly)
                absolutely continuous with respect to lebesgue measure"""
            )
        if self._dh.is_restricted_lebesgue(allow_constant=False):
            scale_by = None
        else:
            scale_by = self._dh.restricted_lebesgue_scaling()
            assert (
                scale_by is not None
            ), "We already checked that it was restricted Lebesgue when allowing constant scaling"
        ts, ts_np_array = self.t_function(x_i)
        if ts_np_array:
            ts = cast(NDArray, ts)
            assert len(ts) == self.num_theta
        else:
            ts = cast(List[PossibleArgument], ts)
            assert len(ts) == self.num_theta
        etas = self.cur_thetas
        assert len(etas) == self.num_theta
        if ts_np_array:
            dotted = np.dot(cast(NDArray, ts), etas)
        else:
            dotted = cast(Union[Trace, numbers.Number], ts[0] * etas[0])
            for idx, (t_i, eta_i) in enumerate(zip(ts, etas)):
                if idx == 0:
                    continue
                dotted = cast(Union[Trace, numbers.Number], dotted + t_i * eta_i)
        try:
            _val = dotted.val  # type: ignore[reportAttributeAccessIssue]
            dotted = cast(Trace, dotted)
            if scale_by is None:
                return dotted - self.a_function(self.cur_thetas)
            return dotted - self.a_function(self.cur_thetas) + np.log(scale_by)
        except AttributeError:
            dotted = cast(numbers.Number, dotted)
            a_val = self.a_function(self.cur_thetas)
            if scale_by is None:
                return dotted - a_val  # type: ignore[reportOperatorIssue]
            return dotted - a_val + np.log(scale_by)  # type: ignore[reportOperatorIssue]

    @property
    def cur_thetas(self) -> NDArray:
        return self._eta_params

    def change_thetas(self, new_thetas: NDArray) -> NDArray:
        old_thetas = self.cur_thetas
        self.set_etas(new_thetas)
        return old_thetas

    @property
    def num_xi(self):
        return self._num_xs

    @property
    def num_theta(self):
        return self._num_etas

    def normalized_base_measure(self) -> BaseMeasure:
        """
        The integration of dh over R^{num_xs} is not
        necessarily normalized
        """
        a_val = self.a_function(np.array([0.0 for _ in range(self._num_etas)]))
        return self._dh.rescaled(np.exp(-a_val))


if __name__ == "__main__":
    z = ExponentialFamily(
        num_etas=1,
        num_xs=1,
        t_function=lambda x: x,
        a_function=lambda x: np.float64(1.0),
        dh=BoxedContinuous([(None, None)], 1.0),
        t_inv_function=None,
    )
