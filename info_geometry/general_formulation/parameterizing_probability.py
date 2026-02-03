"""
Protocol for something paratmeterizing
probability distributions on XType
"""

# pylint:disable = unnecessary-ellipsis

from abc import abstractmethod, ABC
from typing import Generator, Tuple, TypeVar, Generic

from graddog.types import NumericNDArray

# pylint:disable = invalid-name
ThetaType = TypeVar(
    "ThetaType", infer_variance=False, covariant=False, contravariant=False
)
# pylint:disable = invalid-name
TangentThetaType = TypeVar(
    "TangentThetaType", infer_variance=False, covariant=False, contravariant=False
)
# pylint:disable = invalid-name
XType = TypeVar("XType", infer_variance=False, covariant=False, contravariant=False)


class ParameterizingProbability(
    ABC, Generic[ThetaType, TangentThetaType, XType]  # type: ignore
):
    """
    A parameterization of probability distributions
    over XType indexed by ThetaType
    The tangent space at ThetaType is TangentThetaType
    """

    @abstractmethod
    def is_identifiable(self) -> bool:
        """
        Is the paramerization phi : ThetaType -> P(XType)
        identifiable?
        For example if it was not injective,
        then the kernel of dphi would be vectors
        in TangentThetaType
        that would be null directions
        so it would only be semi-definite
        """
        ...

    @abstractmethod
    def sample_size_allowed(self, sample_size: int) -> bool:
        """
        Does taking sample_size of sample_dist
        give the desired distribution
        """
        ...

    @abstractmethod
    def min_sample_size_allowed(self) -> int:
        """
        How many from sample_dist are needed
        to get the desired distribution
        """
        ...

    @abstractmethod
    def sample_dist(
        self,
        theta: ThetaType,  # type: ignore
    ) -> Generator[Tuple[float, XType], None, None]:  # type: ignore
        """
        Sample from the distribution at theta
        """
        ...

    @abstractmethod
    def log_p_theta_xi(
        self,
        theta: ThetaType,
        x_i: XType,
    ) -> float:
        """
        At theta, we get p_theta as a probability distribution over XType
        Return log p_theta(x_i)
        """
        ...

    @abstractmethod
    def theta_to_rn(self, theta: ThetaType) -> NumericNDArray:
        """
        Map theta to R^n
        where n is the dimension of the parameter space
        """
        ...

    @abstractmethod
    def dtheta_to_rn(
        self,
        theta: ThetaType,
        d_theta: TangentThetaType,
    ) -> NumericNDArray:
        """
        Map d_theta in the tangent space at theta
        to R^n
        where n is the dimension of the parameter space
        """
        ...

    def combine_derivative_with_direction(
        self,
        d_coordinates: NumericNDArray,
        theta: ThetaType,
        d_theta: TangentThetaType,
    ) -> float:
        """
        d_coordinates is the derivative
        with respect to the coordinates in R^n
        using the isomorphism of `theta_to_rn`
        """
        dtheta_in_rn = self.dtheta_to_rn(theta, d_theta)
        return d_coordinates.dot(dtheta_in_rn)

    def d_log_p_theta_xi(
        self,
        theta: ThetaType,
        d_theta: TangentThetaType,
        x_i: XType,
    ) -> float:
        """
        The directional derivative
        of log p_theta(x_i)
        in the direction d_theta
        """
        # pylint:disable=import-outside-toplevel
        from graddog import derivatives_only

        d_coordinates = derivatives_only(
            f=lambda th: self.log_p_theta_xi(th, x_i),
            seed=self.theta_to_rn(theta),
        )
        return self.combine_derivative_with_direction(d_coordinates, theta, d_theta)
