"""
Quadratic form
Hessian should give back A
"""

from numpy.typing import NDArray
import numpy as np

import graddog as gd


def make_quadratic(a_matrix: NDArray):
    """
    make the function x -> 1/2 x^TAx
    """

    rows, cols = a_matrix.shape
    if rows != cols:
        raise TypeError(f"{rows} != {cols} so the quadratic form is not valid")
    if rows >= 14:
        raise TypeError("This is too big")

    def f(*args) -> np.float64:
        if len(args) < rows:
            raise TypeError(
                f"There are {len(args)} arguments but this is a quadratic form in {rows} variables"
            )
        to_return = np.float64(0.0)
        for idx in range(rows):
            for jdx in range(cols):
                to_return += a_matrix[idx][jdx] * args[idx] * args[jdx]
        return to_return * 0.5

    return f


if __name__ == "__main__":
    LEN = 10
    a = np.random.rand(LEN, LEN)
    seed = np.random.rand(LEN)

    derivatives, hessian = gd.derivatives_and_hessians(
        f=make_quadratic(a),
        seed=seed,
    )
    f_ = derivatives.round(2)
    f__ = hessian.round(2)
    print(f_)
    print(f__)
