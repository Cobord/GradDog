"""
Any implementable unary (one_parent) or binary (two_parent) operations can be added here
"""

# pylint:disable=pointless-string-statement
import numbers
from collections.abc import Iterable
from typing import Union, cast
import numpy as np

# pylint:disable=consider-using-from-import
import graddog.math as math

# pylint:disable=unused-import
from graddog.trace import Trace, one_parent

type NumberSpecifics = float | int | numbers.Number

type PossibleArgument = Union[
    Trace, NumberSpecifics, Iterable[Union[Trace, NumberSpecifics]]
]


# pylint:disable=raise-missing-from
def sin(t: PossibleArgument):
    """
    This allows to create sin().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues sin() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.sin)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.sin(t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sin(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


def arcsin(t: PossibleArgument):
    """
    This allows to creat arcsin(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)

    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues arcsin() elementary function
    """
    try:
        t_val = t.val # type: ignore
        t = cast(Trace, t)
        if math.in_domain(t_val, math.Ops.arcsin):
            return one_parent(t, math.Ops.arcsin)
        raise ValueError("Input out of domain")
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.arcsin):
                return np.arcsin(t) # type: ignore
            raise ValueError("Input out of domain")
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([arcsin(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def cos(t: PossibleArgument):
    """
    This allows to create cos().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues cos() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.cos)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.cos(t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([cos(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def arccos(t: PossibleArgument):
    """
    This allows to creat arccos(). ValueError is caught if the input Trace
    instance has value not in the domain of (-1, 1)

    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues arccos() elementary function
    """
    try:
        t_val = t.val # type: ignore
        if math.in_domain(t_val, math.Ops.arccos):
            return one_parent(cast(Trace,t), math.Ops.arccos)
        raise ValueError("Input out of domain")
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.arccos):
                return np.arccos(t) # type: ignore
            raise ValueError("Input out of domain")
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([arccos(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def tan(t: PossibleArgument):
    """
    This allows to create tan().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues tan() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.tan)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.tan(t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([tan(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def arctan(t: PossibleArgument):
    """
    This allows to creat arctan()

    Parameters:
        t: (Trace instance, scalar, or vector/list of Traces/scalars)

    Return Trace that constitues arctan() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.arctan)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.arctan(t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([arctan(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from, protected-access
def exp(t: PossibleArgument, base: Union[float, int] = np.e):
    """
    This allows to create exp().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
        base (int, or float)
    Return Trace that constitues exp() elementary function with input base (default=e)
    """
    try:
        _t_val = t.val # type: ignore
        t = cast(Trace, t)
        formula = None
        if base != np.e:
            formula = f"{base}^{t._trace_name}"
        return one_parent(t, math.Ops.exp, base, formula)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.power(base, t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([exp(t_, base) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from, protected-access
def log(t: PossibleArgument, base: Union[float, int] = np.e):
    """
    This allows to create log().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
        base (int, or float)
    Return Trace that constitues log() elementary function with input base (default=e)
    """
    try:
        t_val = t.val # type: ignore
        t = cast(Trace, t)
        formula = None
        if base != np.e:
            formula = f"log_{base}({t._trace_name})"
        if math.in_domain(t_val, math.Ops.log, base):
            return one_parent(t, math.Ops.log, base, formula)
        raise ValueError("Input out of domain")
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.log, base):
                return np.log(t) / np.log(base) # type: ignore
            raise ValueError("Input out of domain")
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([log(t_, base) for t_ in t])
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def sinh(t: PossibleArgument):
    """
    This allows to create sinh().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
        base (int, or float)
    Return Trace that constitues sinh() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.sinh)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.sinh(t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sinh(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def cosh(t: PossibleArgument):
    """
    This allows to create cosh().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues cosh() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.cosh)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.cosh(t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([cosh(t_) for t_ in t]) # type: ignore
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def tanh(t: PossibleArgument):
    """
    This allows to create tanh().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues tanh() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.tanh)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return np.tanh(t) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([tanh(t_) for t_ in t])
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def sqrt(t: PossibleArgument):
    """
    This allows to create sqrt().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues sqrt() elementary function
    """
    try:
        t_val = t.val # type: ignore
        t = cast(Trace,t)
        if math.in_domain(t_val, math.Ops.sqrt):
            return one_parent(t, math.Ops.sqrt)
        raise ValueError("Input out of domain")
    except AttributeError:
        if isinstance(t, numbers.Number):
            if math.in_domain(t, math.Ops.sqrt):
                return t**0.5 # type: ignore
            raise ValueError("Input out of domain")
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sqrt(t_) for t_ in t])
        raise TypeError("Input(s) must be Trace or scalar")


# pylint:disable=raise-missing-from
def sigmoid(t: PossibleArgument):
    """
    This allows to create sigmoid().
    Parameters:
        t (Trace instance, scalar, or vector/list of Traces/scalars)
    Return Trace that constitues sigmoig() elementary function
    """
    try:
        _t_val = t.val # type: ignore
        return one_parent(cast(Trace,t), math.Ops.sigm)
    except AttributeError:
        if isinstance(t, numbers.Number):
            return 1 / (1 + np.exp(-t)) # type: ignore
        if isinstance(t, Iterable) and not isinstance(t, str):
            return np.array([sigmoid(t_) for t_ in t])
        raise TypeError("Input(s) must be Trace or scalar")
