"""
Useful type aliases
that are not strictly enforced
but help with DX
"""

# pylint:disable=invalid-name

import numbers
from typing import Any
import numpy as np
from numpy.typing import NDArray

type NumericNDArray = NDArray[np.number]
type TracesNDArray = NDArray[Any]

type NumberSpecifics = float | int | numbers.Number
