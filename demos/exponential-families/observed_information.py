"""
Expected Information
"""

# pylint:disable=import-outside-toplevel

from abc import ABC, abstractmethod
from collections.abc import Iterable
from numpy.typing import NDArray

from graddog.functions import PossibleArgument
import graddog as gd


class FDistribution(ABC):
    """
    f(x ; theta)
    """

    @abstractmethod
    def f_native(self) -> bool:
        """
        is f the one basicly implemented or log f
        """

    def f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> PossibleArgument:
        """
        density given the observed data ``x_i`` and the possibly
        unknown ``theta``
        """
        if self.f_native():
            raise NotImplementedError
        from graddog.functions import exp

        return exp(self.f_function(x_i, theta))

    def log_f_function(
        self, x_i: PossibleArgument, theta: PossibleArgument
    ) -> PossibleArgument:
        """
        log density given the observed data ``x_i`` and the possibly
        unknown ``theta``
        """
        if self.f_native():
            from graddog.functions import log

            return log(self.f_function(x_i, theta))
        raise NotImplementedError

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
                to_return = to_return + self.log_f_function(item, theta)  # type: ignore
        except StopIteration:
            # pylint:disable = raise-missing-from
            raise ValueError("Empty Iterator. We need a nonzero number of observations")
        return to_return

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
        # pylint:disable= unused-variable
        f_, f__ = gd.derivatives_and_hessians(
            lambda theta: self.log_likelihood(x_is, theta), self.cur_thetas()
        )
        return -f__
