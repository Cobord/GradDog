"""
Amari Tensor Protocol
"""

# pylint:disable = assignment-from-no-return, unnecessary-ellipsis
from abc import ABC, abstractmethod
import itertools
import numbers
from typing import Generic, cast
from graddog.types import NumberSpecifics
from info_geometry.general_formulation.parameterizing_probability import (
    ThetaType,
    TangentThetaType,
    XType,
)
from info_geometry.general_formulation.fisher import FisherMetric


class AmariTensor(
    FisherMetric[ThetaType, TangentThetaType, XType],
    ABC,
    Generic[ThetaType, TangentThetaType, XType],
):
    """
    Amari Tensor Protocol
    Be able to compute the Amari Tensor at any particular theta
    Any implementor of this protocol must also implement FisherMetric
    with the same generics.
    """

    # pylint:disable=duplicate-code
    def amari_tensor(
        self,
        theta: ThetaType,
        d_theta_0: TangentThetaType,
        d_theta_1: TangentThetaType,
        d_theta_2: TangentThetaType,
    ) -> NumberSpecifics:
        """
        Compute the Amari Tensor at theta
        `C_{A,B,C} = E[ (∂_A log f)(∂_B log f)(∂_C log f) ]`
        """
        sample_count = self.min_sample_size_allowed()
        sum_p_i = 0.0
        to_return: NumberSpecifics = 0
        for p_i, x_i in itertools.islice(
            self.sample_dist(theta), None, sample_count, None
        ):
            sum_p_i += p_i
            from_x_i = self.amari_tensor_helper(
                theta, d_theta_0, d_theta_1, d_theta_2, x_i
            )
            if isinstance(to_return, float | int) and isinstance(from_x_i, float | int):
                to_return = to_return + (p_i * from_x_i)
            else:
                to_return = cast(
                    numbers.Number,
                    cast(numbers.Number, to_return)
                    + cast(numbers.Number, p_i * from_x_i),  # type: ignore
                )  # type: ignore
        assert sum_p_i > 0.0, "Sampled probabilities sum to zero"
        if isinstance(to_return, float | int):
            return to_return * (1.0 / sum_p_i)
        if isinstance(to_return, numbers.Number):
            return cast(numbers.Number, to_return * (1.0 / (sum_p_i)))  # type: ignore
        raise TypeError("Amari Tensor returned unknown type")

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    @abstractmethod
    def amari_tensor_helper(
        self,
        theta: ThetaType,
        d_theta_0: TangentThetaType,
        d_theta_1: TangentThetaType,
        d_theta_2: TangentThetaType,
        x_i: XType,
    ) -> NumberSpecifics:
        """
        Helper function to compute the Amari Tensor at theta
        The problem is because the calculation
        involves an expectation over the probability distribution
        over XType produced by theta.
        The default implementation of amari_tensor
        samples from that distribution
        and uses this helper function to compute
        the contribution to due to each sample point
        """
        ...
