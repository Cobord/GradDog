"""
Quadratic form
Hessian should give back A
"""

# pylint:disable=invalid-name

from typing import cast
from numpy.typing import NDArray
import numpy as np
import pytest

import graddog as gd
from graddog.compgraph import CompGraph
from graddog.trace import Trace


def make_quadratic(a_matrix: NDArray):
    """
    make the function x -> 1/2 x^TAx
    """

    rows, cols = a_matrix.shape
    if rows != cols:
        raise TypeError(f"{rows} != {cols} so the quadratic form is not valid")
    if rows >= 14:
        raise TypeError("This is too big")

    def f(*args) -> np.float64 | Trace:
        if len(args) < rows:
            raise TypeError(
                f"There are {len(args)} arguments but this is a quadratic form in {rows} variables"
            )
        to_return = np.float64(0.0)
        for idx in range(rows):
            for jdx in range(cols):
                to_return += a_matrix[idx][jdx] * args[idx] * args[jdx]
        to_return = cast(np.float64 | Trace, to_return)
        to_return = to_return * 0.5
        return to_return

    return f


def test_smaller_quadratic():
    """
    a 4 by 4 1/2 x^T A x
    """
    CompGraph.reset()
    LEN = 4
    a = np.random.rand(LEN, LEN)
    a = (a + a.transpose()) / 2
    seed = np.random.rand(LEN)

    derivatives, hessian = gd.derivatives_and_hessians(
        f=make_quadratic(a),
        seed=seed,
    )
    # pylint:disable=unused-variable
    f_ = derivatives
    f__ = hessian
    assert f__ == pytest.approx(a), f"Differentiated as\n{f__}\n, vs mades as\n{a}\n"


def test_small_quadratic():
    """
    a 7 by 7 1/2 x^T A x
    """
    CompGraph.reset()
    LEN = 7
    a = np.random.rand(LEN, LEN)
    a = (a + a.transpose()) / 2
    seed = np.random.rand(LEN)

    derivatives, hessian = gd.derivatives_and_hessians(
        f=make_quadratic(a),
        seed=seed,
    )
    # pylint:disable=unused-variable
    f_ = derivatives
    f__ = hessian
    assert f__ == pytest.approx(a), f"Differentiated as\n{f__}\n, vs mades as\n{a}\n"
