"""
Create Gaussian as incarnation
of ExponentialFamily
"""

import numbers
import random
from typing import Optional, Union, cast

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pytest
from graddog.functions import NumberSpecifics, abs_gd, log
from graddog.trace import Trace
from graddog.types import NumericNDArray, TracesNDArray
from hessian_applications.exponential_families.exponential_family import (
    ExponentialFamily,
    BoxedContinuous,
    one_gaussian_base_measure,
)


def create_oned_gaussian(known_variance: Optional[float]) -> ExponentialFamily:
    """
    Create Gaussian as incarnation
    of ExponentialFamily
    """

    def x_and_x_squared(x: Union[Trace, NumberSpecifics, NDArray]):
        """
        x -> [x,x^2]
        """
        if isinstance(x, Trace):
            return [x, x * x]
        if isinstance(x, (numbers.Number, float, int)):
            return np.array([x, x * x])  # type: ignore
        try:
            assert x.shape == (1,), x.shape
            x = x[0]
            return np.array([x, x * x])  # type: ignore
        except Exception as e:
            raise TypeError("Trace or number or NDArray of shape 1") from e

    def a_function(
        eta: Union[NumericNDArray, TracesNDArray],
    ) -> Union[NumericNDArray, TracesNDArray]:
        """
        a function for a Gaussian with both unknown mean and variance
        """
        assert eta.shape == (2,)
        if not eta.dtype.hasobject:
            assert isinstance(
                eta[0], numbers.Number
            ), f"{eta.dtype} not has object : {not eta.dtype.hasobject}"
            assert isinstance(eta[1], numbers.Number)
            return np.array(
                -eta[0] * eta[0] / (4 * eta[1])
                + 1 / 2 * np.log(np.abs(1 / (2 * eta[1])))
            )
        eta = cast(TracesNDArray, eta)
        z = eta[0] * eta[0] / (4 * eta[1]) + 1 / 2 * log(abs_gd(1 / (2 * eta[1])))
        return np.array([z])

    if known_variance is None:
        return ExponentialFamily(
            num_etas=2,
            num_xs=1,
            t_function=x_and_x_squared,  # type: ignore
            t_function_traced=x_and_x_squared,  # type: ignore
            a_function=lambda eta: a_function(eta)[0],
            a_function_traced=a_function,
            dh=BoxedContinuous([(None, None)], (1 / np.sqrt(2 * np.pi))),
            t_inv_function=lambda t: np.array(t[0]),
            # pylint:disable=line-too-long
            description_with_parameters=lambda etas: f"Gaussian as exponential family with both unknown and etas = {etas}",
        )

    def a_function_known(
        eta: Union[NumericNDArray, TracesNDArray],
    ) -> Union[NumericNDArray, TracesNDArray]:
        """
        a function for a Gaussian with unknown mean and known variance
        """
        assert eta.shape == (1,)
        if not eta.dtype.hasobject:
            return np.array([eta[0] * eta[0] / 2])
        eta = cast(TracesNDArray, eta)
        z = eta[0] * eta[0] / 2
        return np.array([z])

    return ExponentialFamily(
        num_etas=1,
        num_xs=1,
        t_function=lambda x: x / known_variance,
        t_function_traced=lambda x: x / known_variance,
        a_function=lambda eta: a_function_known(eta)[0],
        a_function_traced=lambda eta: a_function_known(eta)[0],
        dh=one_gaussian_base_measure(0, known_variance),
        t_inv_function=lambda t: t * known_variance,
        # pylint:disable=line-too-long
        description_with_parameters=lambda etas: f"Gaussian as exponential family with unknown mean but known variance {known_variance}.\neta is set to {etas}",
    )


def test_support():
    """
    test that the support of gaussians is the entire line
    """
    one_d_gaussian_unknown_both = create_oned_gaussian(None)
    for cur_pt in [-1.2, -0.2, 0.5, 1.3]:
        one_d_gaussian_unknown_both.in_support(np.array(cur_pt))
    one_d_gaussian_known_variance = create_oned_gaussian(1.0)
    for cur_pt in [-1.2, -0.2, 0.5, 1.3]:
        one_d_gaussian_known_variance.in_support(np.array([cur_pt]))


def test_logf_values_both():
    """
    Test that log f obtained from ExponentialFamily
    matches with explicit gaussian PDF
    """
    one_d_gaussian_unknown_both = create_oned_gaussian(None)
    x_mesh = 1.0
    for mu, sigma_squared in [(0.0, 1.0), (1.0, 1.0), (0.0, 2.0), (1.0, 2.0)]:
        eta_0 = mu / sigma_squared
        eta_1 = -1 / (2 * sigma_squared)
        log_f = one_d_gaussian_unknown_both.log_f_function(
            np.array([x_mesh]),
            np.array([eta_0, eta_1]),
            debug=True,
        )
        assert isinstance(log_f, numbers.Number)
        expected_value = np.exp(
            -((x_mesh - mu) * (x_mesh - mu)) / (2 * sigma_squared)
        ) / (np.sqrt(2 * np.pi) * np.sqrt(sigma_squared))
        exp_log_f = np.exp(np.array(log_f))
        # pylint:disable=line-too-long
        assert exp_log_f == pytest.approx(
            expected_value
        ), f"mu={mu}, sigma^2={sigma_squared}, x={x_mesh}, raw_gaussian={expected_value}, family={exp_log_f}"


# pylint:disable=too-many-locals
def main():
    """
    script version for gaussian examples
    """

    one_d_gaussian_unknown_both = create_oned_gaussian(None)
    one_d_gaussian_unknown_both.set_etas(np.array([0.0, -0.5]))
    print(f"Cur thetas {one_d_gaussian_unknown_both}")

    for mu, sigma_squared in [(0.0, 1.0), (1.0, 1.0), (-3.0, 1.0)]:
        x_mesh = np.linspace(-10, 10, 100)
        f_vals = np.exp(-((x_mesh - mu) * (x_mesh - mu)) / (2 * sigma_squared)) / (
            np.sqrt(2 * np.pi) * np.sqrt(sigma_squared)
        )
        g_vals = [
            cast(
                numbers.Number,
                one_d_gaussian_unknown_both.f_function(
                    x, np.array([mu / sigma_squared, -1 / (2 * sigma_squared)])
                ),
            )
            for x in x_mesh
        ]

        plt.plot(x_mesh, f_vals, color="red", label=f"gauss(x;{mu},{sigma_squared})")
        plt.plot(
            x_mesh,
            np.array(g_vals),
            color="blue",
            label=f"f exponential fam(x;{mu},{sigma_squared})",
        )
        plt.title("Gaussian")
        plt.xlabel("x")
        plt.ylabel("p(x)")
        plt.xlim(-10, 10)
        plt.legend()
        plt.show()

    mu, sigma_squared = np.meshgrid(np.linspace(0, 8), np.linspace(1, 5))
    prob_at_one = mu * 0
    rows, cols = prob_at_one.shape
    for idx in range(rows):
        for jdx in range(cols):
            cur_eta_0 = mu[idx][jdx] / sigma_squared[idx][jdx]
            cur_eta_1 = -1 / (2 * sigma_squared[idx][jdx])
            assert isinstance(cur_eta_0, float), cur_eta_0
            assert isinstance(cur_eta_1, float), cur_eta_1
            cur_z = cast(
                numbers.Number,
                one_d_gaussian_unknown_both.f_function(
                    np.array([1]), np.array([cur_eta_0, cur_eta_1])
                ),
            )
            prob_at_one[idx][jdx] = cast(float, cur_z)
    _h = plt.contourf(mu, sigma_squared, prob_at_one)
    plt.title("Probability at 1")
    plt.xlabel("mu")
    plt.ylabel("sigma^2")
    plt.show()

    mu_exact = 4.0
    sigma_squared_exact = 0.5

    mu, sigma_squared = np.meshgrid(np.linspace(2, 6), np.linspace(0.25, 5))
    sample = [random.gauss(mu_exact, np.sqrt(sigma_squared_exact)) for _ in range(100)]
    sample_np = np.array(sample)
    likelihood_of_seeing_sample = mu * 0
    rows, cols = likelihood_of_seeing_sample.shape
    for idx in range(rows):
        for jdx in range(cols):
            cur_eta_0 = mu[idx][jdx] / sigma_squared[idx][jdx]
            cur_eta_1 = -1 / (2 * sigma_squared[idx][jdx])
            assert isinstance(cur_eta_0, float), cur_eta_0
            assert isinstance(cur_eta_1, float), cur_eta_1
            cur_z = cast(
                numbers.Number,
                one_d_gaussian_unknown_both.log_likelihood(
                    sample_np, np.array([cur_eta_0, cur_eta_1])
                ),
            )
            likelihood_of_seeing_sample[idx][jdx] = cast(float, cur_z)
    _h = plt.contourf(mu, sigma_squared, likelihood_of_seeing_sample)
    plt.title("Likelihood of seeing this sample")
    plt.xlabel("mu")
    plt.ylabel("sigma^2")
    plt.show()


if __name__ == "__main__":
    main()
