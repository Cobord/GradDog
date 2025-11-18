"""
Exponential Family
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numbers
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast
import numpy as np

import graddog as gd
from graddog.types import NumericNDArray, TracesNDArray
from graddog.functions import NumberSpecifics, PossibleArgument, as_np_array
from graddog.trace import Trace, Variable

from .observed_information import FDistribution


# pylint:disable=too-few-public-methods
class BaseMeasure(ABC):
    """
    h(x) d\\mu_x
    """

    @abstractmethod
    def in_support(self, x: NumericNDArray) -> bool:
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
        bounds: List[Tuple[Optional[float], Optional[float]]],
        scale_factor: float,
    ):
        super().__init__()
        self.bounds = bounds
        self.scale_factor = scale_factor

    def in_support(self, x: NumericNDArray):
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

        if x.shape == ():
            assert len(self.bounds) == 1, f"{x} {len(self.bounds)}"
            x = x.reshape(
                1,
            )
        assert x.shape == (len(self.bounds),), f"{x} {len(self.bounds)}"
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
    h: Callable[[NumericNDArray], float]

    def __init__(self, mu, h):
        self.mu = mu
        self.h = h

    def in_support(self, x: NumericNDArray) -> bool:
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


# pylint:disable=too-many-instance-attributes
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
        t_function: Callable[[NumericNDArray], NumericNDArray],
        t_function_traced: Callable[
            [Union[NumericNDArray, TracesNDArray]], Union[NumericNDArray, TracesNDArray]
        ],
        a_function: Callable[[NumericNDArray], np.number],
        a_function_traced: Callable[
            [Union[NumericNDArray, TracesNDArray]], Union[NumericNDArray, TracesNDArray]
        ],
        dh: BaseMeasure,
        t_inv_function: Optional[Callable[[NumericNDArray], NumericNDArray]] = None,
        description_with_parameters: Optional[Callable[[NumericNDArray], str]] = None,
    ):
        self._t_function = t_function
        self._t_function_traced = t_function_traced
        self._t_inv_function = t_inv_function
        self._a_function = a_function
        self._a_function_traced = a_function_traced
        self._num_etas = num_etas
        self._eta_params = np.random.rand(num_etas)
        self._dh = dh
        self._num_xs = num_xs
        self.description_with_parameters = description_with_parameters

    def set_etas(self, new_etas: NumericNDArray):
        """
        Set the current eta parameters
        """
        if new_etas.shape != (self._num_etas,):
            # pylint:disable=line-too-long
            raise TypeError(
                f"This is not the right number of eta parameters. We expect {self._num_etas} but we got {new_etas}"
            )
        self._eta_params = new_etas

    def t_vars_exps_covariance(self) -> Tuple[NumericNDArray, NumericNDArray]:
        """
        Expectation values and covariances of the
        sufficient statistics T's
        """
        f_, f__ = gd.derivatives_and_hessians(
            f=self._a_function,
            seed=self._eta_params,
        )
        return (f_, f__)

    def in_support(self, x: NumericNDArray) -> bool:
        """
        The support for the exponential family is the
        same as that of the base measure dh.
        """
        return self._dh.in_support(x)

    # pylint:disable = too-many-return-statements
    def in_support_traced(self, x_i: PossibleArgument) -> Optional[bool]:
        """
        The support for the exponential family is the
        same as that of the base measure dh.
        But this time we are allowing the x_i to have `Trace`s
        in which case the answer is indeterminant
        """
        if isinstance(x_i, Trace):
            return None
        if isinstance(x_i, (numbers.Number, float, int)):
            return self.in_support(np.array([x_i]))
        if isinstance(x_i, list):
            if all(isinstance(x, (numbers.Number, float, int)) for x in x_i):
                return self.in_support(np.array(x_i))
            return None
        xi_num_np_array = False
        xi_gen_np_array = False
        try:
            x_i = cast(np.typing.NDArray, x_i)
            _ = x_i.shape
            xi_gen_np_array = True
            xi_num_np_array = not x_i.dtype.hasobject
        except AttributeError:
            pass
        if xi_num_np_array:
            return self.in_support(cast(NumericNDArray, x_i))
        if xi_gen_np_array:
            return None
        x_i = cast(Iterable[NumberSpecifics | Trace], x_i)
        return self.in_support_traced(list(x_i))

    def t_function(
        self, x: PossibleArgument
    ) -> Tuple[Union[List[Trace], NumericNDArray], bool]:
        """
        The sufficient statistic Ts in terms of the xs
        """
        if isinstance(x, numbers.Number | float | int):
            x = np.array([x])
            returning = self._t_function(x)
            assert returning.shape == (self._num_etas,)
            return returning, True
        if isinstance(x, Trace):
            return_val = self._t_function_traced(np.array([x]))
        else:
            return_val = self._t_function_traced(np.array(x))
        if isinstance(return_val, numbers.Number):
            assert self._num_etas == 1
            return (np.array([return_val]), True)
        if isinstance(return_val, float):
            assert self._num_etas == 1
            return (np.array([return_val]), True)
        if isinstance(return_val, int):
            assert self._num_etas == 1
            return (np.array([return_val]), True)
        if isinstance(return_val, Trace):
            assert self._num_etas == 1
            return ([return_val], False)
        return_val = list(return_val)
        assert self._num_etas == len(return_val)
        if any(isinstance(x, Trace) for x in return_val):
            if all(isinstance(x, Trace) for x in return_val):
                return_val = cast(List[Trace], return_val)
                return (return_val, False)
            raise ValueError(
                "One of them was a Trace so all the outputs should have been Trace"
            )
        return_val = cast(List[NumberSpecifics], return_val)
        return (np.array(return_val), True)

    def a_function(self, thetas: NumericNDArray) -> np.number:
        """
        the A function which does normalization in a parameter dependent manner
        """
        assert thetas.shape == (self.num_theta,)
        return self._a_function(thetas)

    def a_function_traced(
        self, thetas: Union[NumericNDArray, TracesNDArray]
    ) -> Union[NumericNDArray, TracesNDArray]:
        """
        the A function which does normalization in a parameter dependent manner
        """
        assert thetas.shape == (self.num_theta,)
        return self._a_function_traced(thetas)

    @property
    def f_native(self) -> bool:
        return False

    def f_function(
        self,
        x_i: Union[NumericNDArray, TracesNDArray],
        theta: Union[NumericNDArray, TracesNDArray],
    ) -> Union[Trace, numbers.Number]:
        # pylint:disable=useless-parent-delegation
        return super().f_function(x_i, theta)

    # pylint:disable=too-many-branches, too-many-locals, too-many-statements
    def log_f_function(
        self,
        x_i: Union[NumericNDArray, TracesNDArray],
        theta: Union[NumericNDArray, TracesNDArray],
        debug: bool = False,
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
        assert self.in_support_traced(x_i)
        ts, ts_np_array = self.t_function(x_i)
        if ts_np_array:
            ts = cast(NumericNDArray, ts)
            assert ts.shape == (self.num_theta,), f"{x_i} -> {ts}"
        else:
            ts = cast(List[Trace], ts)
            assert len(ts) == self.num_theta
        etas, etas_np_array = as_np_array(theta, self.num_theta)
        if ts_np_array and etas_np_array:
            dotted = np.dot(cast(NumericNDArray, ts), cast(NumericNDArray, etas))
            try:
                dotted_shape = dotted.shape
                assert dotted_shape in ((1,), ())
                if dotted_shape == (1,):
                    dotted = cast(Union[numbers.Number, Trace], dotted[0])
                else:
                    dotted = cast(Union[numbers.Number, Trace], dotted)
            except AttributeError:
                assert isinstance(dotted, (numbers.Number, Trace))
        else:
            etas = cast(List[Union[Trace, NumberSpecifics]], etas)
            dotted = cast(Union[Trace, numbers.Number], ts[0] * etas[0])
            for idx, (t_i, eta_i) in enumerate(zip(ts, etas)):
                if idx == 0:
                    continue
                dotted = cast(Union[Trace, numbers.Number], dotted + t_i * eta_i)
        if debug:
            print(f"eta dot T gave {dotted}")
        try:
            _val = dotted.val  # type: ignore[reportAttributeAccessIssue]
            dotted = cast(Trace, dotted)
            a_val = self.a_function_traced(theta)
            if a_val.shape == (1,):
                a_val = a_val[0]
            if debug:
                print(f"A(eta) gave {a_val}")
            if scale_by is None:
                returning = dotted - a_val
            else:
                returning = dotted - a_val + np.log(scale_by)
            assert isinstance(returning, Trace)
        except AttributeError:
            dotted = cast(numbers.Number, dotted)
            a_val = self.a_function_traced(theta)
            if a_val.shape == (1,):
                a_val = a_val[0]
            if debug:
                print(f"A(eta) gave {a_val}")
            a_is_trace = isinstance(a_val, Trace)
            if scale_by is None:
                returning = dotted - a_val  # type: ignore[reportOperatorIssue]
            else:
                returning = dotted - a_val + np.log(scale_by)  # type: ignore[reportOperatorIssue]
            if a_is_trace:
                assert isinstance(returning, Trace)
            else:
                assert isinstance(returning, (numbers.Number, float, int))
                if not isinstance(returning, numbers.Number):
                    returning = cast(numbers.Number, returning)
        if debug:
            print(f"combined gave {returning}")
        return returning

    @property
    def cur_thetas(self) -> NumericNDArray:
        return self._eta_params

    def change_thetas(self, new_thetas: NumericNDArray) -> NumericNDArray:
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

    def __str__(self) -> str:
        if self.description_with_parameters is None:
            return str(super())
        return self.description_with_parameters(self.cur_thetas)


if __name__ == "__main__":
    z = ExponentialFamily(
        num_etas=1,
        num_xs=1,
        t_function=lambda x: x,
        t_function_traced=lambda x: x,
        a_function=lambda x: np.float64(1.0),
        a_function_traced=lambda x: np.array([Variable("1", 1.0)]),
        dh=BoxedContinuous([(None, None)], 1.0),
        t_inv_function=None,
    )
