"""
Create Gaussian as incarnation
of ExponentialFamily
"""

from typing import Optional

import numpy as np
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
    if known_variance is None:
        return ExponentialFamily(
            num_etas=2,
            num_xs=1,
            t_function=lambda x: np.array([x, x * x]),
            a_function=lambda eta: (eta * eta / 2)[0],
            dh=BoxedContinuous([(None, None)], (1 / np.sqrt(2 * np.pi))),
            t_inv_function=lambda t: t * known_variance,
        )
    return ExponentialFamily(
        num_etas=1,
        num_xs=1,
        t_function=lambda x: x / known_variance,
        a_function=lambda eta: (eta * eta / 2)[0],
        dh=one_gaussian_base_measure(0, known_variance),
        t_inv_function=lambda t: t * known_variance,
    )
