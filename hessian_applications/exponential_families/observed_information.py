"""
Expected Information
"""

# pylint:disable=import-outside-toplevel

from abc import ABC, abstractmethod
from collections.abc import Iterable
import numbers
import random
import string
from typing import Callable, Union, cast
import numpy as np
from numpy.typing import NDArray

from graddog.functions import PossibleArgument, log
import graddog as gd
from graddog.trace import Trace, Variable


def _gen_name(length: int) -> str:
    """
    Produce a random variable name
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


class FDistribution(ABC):
    """
    f(x ; theta)
    """

    @property
    @abstractmethod
    def f_native(self) -> bool:
        """
        is f the one basicly implemented or log f
        """

    @property
    @abstractmethod
    def num_xi(self) -> int:
        """
        What is the dimension of the x_i
        """

    @property
    @abstractmethod
    def num_theta(self) -> int:
        """
        What is the dimension of the theta
        """

    @abstractmethod
    def f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        """
        density given the observed data ``x_i`` and the possibly
        unknown ``theta``
        """
        if self.f_native:
            raise NotImplementedError(
                """We were told f was natively implemented
                but this is the abstract super class without f"""
            )
        from graddog.functions import exp

        return cast(Union[Trace, numbers.Number], exp(self.log_f_function(x_i, theta)))

    @abstractmethod
    def log_f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        """
        log density given the observed data ``x_i`` and the possibly
        unknown ``theta``
        """
        if self.f_native:
            return cast(Union[Trace, numbers.Number], log(self.f_function(x_i, theta)))
        raise NotImplementedError(
            """We were told log f was natively implemented
            but this is the abstract super class without log f"""
        )

    def log_likelihood(
        self, x_is: Iterable[PossibleArgument], theta: PossibleArgument
    ) -> PossibleArgument:
        """
        sum_{x_i} log f(x_i; theta)
        """
        x_is = iter(x_is)
        try:
            to_return = self.log_f_function(next(x_is), theta)
            for item in x_is:
                to_return = to_return + self.log_f_function(
                    item, theta
                )  # type: ignore[reportOperatorIssue]
        except StopIteration:
            # pylint:disable = raise-missing-from
            raise ValueError("Empty Iterator. We need a nonzero number of observations")
        return to_return

    @property
    @abstractmethod
    def cur_thetas(self) -> NDArray:
        """
        value of theta
        """

    @abstractmethod
    def change_thetas(self, new_thetas: NDArray) -> NDArray:
        """
        change value of theta
        returning the previously stored ones
        """

    def observed_information_matrix(self, x_is: Iterable[PossibleArgument]) -> NDArray:
        """
        The hessian of ``log_likelihood``
        at ``cur_thetas``
        """

        def log_likelihood_to_trace(theta):
            return self.log_likelihood(x_is, theta)

        # pylint:disable= unused-variable
        f_, f__ = gd.derivatives_and_hessians(
            log_likelihood_to_trace,
            self.cur_thetas,
        )
        return -f__


class GeneralF(FDistribution):
    """
    Explicitly given f
    """

    def __init__(
        self,
        f_stored: Callable[
            [PossibleArgument, PossibleArgument], Union[Trace, numbers.Number]
        ],
        num_xi_known: int,
        thetas: NDArray,
    ):
        self.f_stored = f_stored
        self.thetas = thetas
        self._num_xi = num_xi_known
        self._num_theta = len(self.thetas)

    @property
    def f_native(self) -> bool:
        return True

    def f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        return self.f_stored(x_i, theta)

    def log_f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        # pylint:disable=useless-parent-delegation
        return super().log_f_function(x_i, theta)

    @property
    def cur_thetas(self) -> NDArray:
        return self.thetas

    def change_thetas(self, new_thetas: NDArray) -> NDArray:
        self.thetas, old_thetas = new_thetas, self.thetas
        return old_thetas

    @property
    def num_xi(self) -> int:
        return self._num_xi

    @property
    def num_theta(self) -> int:
        return self._num_theta


class GeneralLogF(FDistribution):
    """
    Explicitly given log f
    """

    def __init__(
        self,
        log_f_stored: Callable[
            [PossibleArgument, PossibleArgument], Union[Trace, numbers.Number]
        ],
        num_xi_known: int,
        thetas: NDArray,
    ):
        self.log_f_stored = log_f_stored
        self.thetas = thetas
        self._num_xi = num_xi_known
        self._num_theta = len(self.thetas)

    @property
    def f_native(self) -> bool:
        return False

    def log_f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        return self.log_f_stored(x_i, theta)

    def f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> Union[Trace, numbers.Number]:
        # pylint:disable=useless-parent-delegation
        return super().f_function(x_i, theta)

    @property
    def cur_thetas(self) -> NDArray:
        return self.thetas

    def change_thetas(self, new_thetas: NDArray) -> NDArray:
        self.thetas, old_thetas = new_thetas, self.thetas
        return old_thetas

    @property
    def num_xi(self) -> int:
        return self._num_xi

    @property
    def num_theta(self) -> int:
        return self._num_theta


if __name__ == "__main__":

    def f_fun(_x, theta):
        """junk f for making sure works"""
        return theta

    z = GeneralF(f_fun, num_xi_known=1, thetas=np.array([2.3940]))
    print(z.f_function(cast(numbers.Number, 0.2), cast(numbers.Number, 0.1)))
    print(z.log_f_function(cast(numbers.Number, 2.71828), cast(numbers.Number, 0.1)))
    print(z.log_likelihood([cast(numbers.Number, 2.71828)], cast(numbers.Number, 0.1)))
    print(z.f_function(cast(numbers.Number, 0.2), Variable("theta", 0.1)))
    print(z.log_f_function(cast(numbers.Number, 0.2), Variable("theta", 0.1)))
    print(z.log_likelihood([cast(numbers.Number, 2.71828)], Variable("theta", 0.1)))
    print(z.observed_information_matrix([0.2, Variable("x2", 0.3), 0.24, 0.28, 0.23]))

    def logf_fun(_x, theta):
        """junk log f for making sure works"""
        return cast(Union[Trace, numbers.Number], log(theta))

    z = GeneralLogF(logf_fun, num_xi_known=1, thetas=np.array([2.3940]))
    print(z.f_function(cast(numbers.Number, 0.2), cast(numbers.Number, 0.1)))
    print(z.log_f_function(cast(numbers.Number, 2.71828), cast(numbers.Number, 0.1)))
    print(z.log_likelihood([cast(numbers.Number, 2.71828)], cast(numbers.Number, 0.1)))
    print(z.f_function(cast(numbers.Number, 0.2), Variable("theta", 0.1)))
    print(z.log_f_function(cast(numbers.Number, 0.2), Variable("theta", 0.1)))
    print(z.log_likelihood([cast(numbers.Number, 2.71828)], Variable("theta", 0.1)))
    print(z.observed_information_matrix([0.2, Variable("x2", 0.3), 0.24, 0.28, 0.23]))
