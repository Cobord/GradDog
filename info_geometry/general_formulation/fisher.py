"""
Fisher Information Metric Protocol
"""

# pylint:disable = assignment-from-no-return, unnecessary-ellipsis

from abc import ABC, abstractmethod
import itertools
import numbers
from typing import Generic, cast
from graddog.types import NumberSpecifics

from info_geometry.general_formulation.parameterizing_probability import (
    ParameterizingProbability,
    ThetaType,
    TangentThetaType,
    XType,
)


class FisherMetric(
    ParameterizingProbability[ThetaType, TangentThetaType, XType],
    ABC,
    Generic[ThetaType, TangentThetaType, XType],
):
    """
    Fisher Information Metric Protocol
    Be able to compute the Fisher Information Metric at any particular theta
    Any implementor of this protocol must also implement ParameterizingProbability
    with the same generics.
    It is not necessarily a metric because
    the parameterization may not be identifiable
    (ie the map from ThetaType to distributions over XType
    may not be injective)
    but that is a predicate method implemented when implementing ParametrizingProbability
    so we do not duplicate it here.
    Some methods will raise exceptions if the parameterization is not identifiable.
    """

    def fisher_information(
        self,
        theta: ThetaType,
        d_theta_0: TangentThetaType,
        d_theta_1: TangentThetaType,
    ) -> NumberSpecifics:
        """
        Compute the Fisher Information Metric at theta
        """
        sample_count = self.min_sample_size_allowed()
        sum_p_i = 0.0
        to_return: NumberSpecifics = 0
        for p_i, x_i in itertools.islice(
            self.sample_dist(theta), None, sample_count, None
        ):
            sum_p_i += p_i
            from_x_i = self.fisher_information_helper(theta, d_theta_0, d_theta_1, x_i)
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
        raise TypeError("Fisher Information Metric returned unknown type")

    @abstractmethod
    def fisher_information_helper(
        self,
        theta: ThetaType,
        d_theta_0: TangentThetaType,
        d_theta_1: TangentThetaType,
        x_i: XType,
    ) -> NumberSpecifics:
        """
        Helper function to compute the Fisher Information Metric at theta
        """
        ...
