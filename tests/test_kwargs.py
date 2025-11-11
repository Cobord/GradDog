"""
gd.trace with keyword arguments on the functions
"""

# pylint:disable=ungrouped-imports, missing-function-docstring, unnecessary-dunder-call, protected-access, invalid-name
from enum import IntEnum, auto
import pytest
import numpy as np
import graddog as gd

# pylint:disable=unused-import
from graddog.functions import (
    sin,
    arcsin,
    sinh,
    cos,
    arccos,
    cosh,
    tan,
    arctan,
    tanh,
    exp,
    log,
    sigmoid,
    sqrt,
)

class Choices(IntEnum):
    """
    two choices of which function to do
    """
    FUNCTION1 = auto()
    FUNCTION2 = auto()

def two_choices(xs,*,choice: Choices=Choices.FUNCTION1):
    """
    two different functions
    each one can differentiate but there is no sense of
    being differentiable in the keyword choice
    """
    match choice:
        case Choices.FUNCTION1:
            return sin(xs)
        case Choices.FUNCTION2:
            return cos(xs)


def test_derivatives_choice_of_two_RtoR():
    """
    there is a discrete choice that is fixed and we are
    only ever differentiating with that choice fixed
    """

    inputs = [1,2,3,4]
    num_inputs = len(inputs)

    x = gd.derivatives_only(two_choices, inputs)
    assert x.shape == (num_inputs,num_inputs)
    for idx in range(num_inputs):
        for jdx in range(num_inputs):
            if idx != jdx:
                assert x[idx][jdx] == 0.0
            else:
                assert x[idx][jdx] == pytest.approx(np.cos(inputs[idx]))
    x = gd.derivatives_only(two_choices, inputs, choice=Choices.FUNCTION1)
    assert x.shape == (num_inputs,num_inputs)
    for idx in range(num_inputs):
        for jdx in range(num_inputs):
            if idx != jdx:
                assert x[idx][jdx] == 0.0
            else:
                assert x[idx][jdx] == pytest.approx(np.cos(inputs[idx]))
    x = gd.derivatives_only(two_choices, inputs, choice=Choices.FUNCTION2)
    assert x.shape == (num_inputs,num_inputs)
    for idx in range(num_inputs):
        for jdx in range(num_inputs):
            if idx != jdx:
                assert x[idx][jdx] == 0.0
            else:
                assert x[idx][jdx] == pytest.approx(-np.sin(inputs[idx]))

def test_hessians_choice_of_two_RtoR():
    """
    there is a discrete choice that is fixed and we are
    only ever hessians with that choice fixed
    """

    inputs = [5.0]
    _,x = gd.derivatives_and_hessians(two_choices, inputs)
    assert x.shape == (1,1)
    assert x[0][0] == pytest.approx(-np.sin(inputs[0]))
    _,x = gd.derivatives_and_hessians(two_choices, inputs, choice=Choices.FUNCTION1)
    assert x.shape == (1,1)
    assert x[0][0] == pytest.approx(-np.sin(inputs[0]))
    _,x = gd.derivatives_and_hessians(two_choices, inputs, choice=Choices.FUNCTION2)
    assert x.shape == (1,1)
    assert x[0][0] == pytest.approx(-np.cos(inputs[0]))
