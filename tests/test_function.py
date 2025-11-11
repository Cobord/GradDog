"""
mathematical functions
"""

# pylint:disable=ungrouped-imports, missing-function-docstring, unnecessary-dunder-call, protected-access, invalid-name
import numbers
import re
from typing import List, cast
import pytest
import numpy as np
from numpy.typing import NDArray
import graddog as gd
from graddog.trace import Trace, Variable
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


def test_sin():
    value = 0.5
    x = Variable("x", value)
    cast_value = cast(numbers.Number,value)
    a = sin(cast_value)
    b = arcsin(cast_value)
    c = sinh(cast_value)
    assert a == pytest.approx(np.sin(value))
    assert b == pytest.approx(np.arcsin(value))
    assert c == pytest.approx(np.sinh(value))
    g = cast(Trace,sin(x))
    h = cast(Trace,arcsin(x))
    i = cast(Trace,sinh(x))
    assert g.val == pytest.approx(np.sin(value))
    assert h.val == pytest.approx(np.arcsin(value))
    assert i.val == pytest.approx(np.sinh(value))


def test_cos():
    value = 0.5
    x = Variable("x", value)
    a = cos(cast(numbers.Number,value))
    b = arccos(cast(numbers.Number,value))
    c = cosh(cast(numbers.Number,value))
    assert a == pytest.approx(np.cos(value))
    assert b == pytest.approx(np.arccos(value))
    assert c == pytest.approx(np.cosh(value))
    g = cast(Trace,cos(x))
    h = cast(Trace,arccos(x))
    i = cast(Trace,cosh(x))
    assert g.val == pytest.approx(np.cos(value))
    assert h.val == pytest.approx(np.arccos(value))
    assert i.val == pytest.approx(np.cosh(value))


def test_tan():
    value = 0.5
    x = Variable("x", value)
    cast_value = cast(numbers.Number, value)
    a = tan(cast_value)
    b = arctan(cast_value)
    c = tanh(cast_value)
    assert a == pytest.approx(np.tan(value))
    assert b == pytest.approx(np.arctan(value))
    assert c == pytest.approx(np.tanh(value))
    g = cast(Trace,tan(x))
    h = cast(Trace,arctan(x))
    i = cast(Trace,tanh(x))
    assert g.val == pytest.approx(np.tan(value))
    assert h.val == pytest.approx(np.arctan(value))
    assert i.val == pytest.approx(np.tanh(value))


def test_sigmoid():
    value = 0.5
    x = Variable("x", value)
    cast_value = cast(numbers.Number, value)
    g = cast(Trace,sigmoid(x))
    a = sigmoid(cast_value)
    assert a == pytest.approx(1 / (1 + np.exp(-value)))
    assert g.val == pytest.approx(1 / (1 + np.exp(-value)))


def test_sqrt():
    value = 9
    x = Variable("x", value)
    cast_value = cast(numbers.Number, value)
    a = sqrt(cast_value)
    assert a == pytest.approx(np.sqrt(value))
    g = cast(Trace,sqrt(x))
    assert g.val == pytest.approx(np.sqrt(value))


def test_log_base2():
    x = Variable("x", 32)
    base = 2
    f = cast(Trace,log(x, base=base))
    assert f._val == pytest.approx(5)
    assert f._der["v1"] == 1 / (x._val * np.log(base))


def test_exp_base2():
    x = Variable("x", 5)
    base = 2
    f = cast(Trace,exp(x, base=base))
    assert f._val == pytest.approx(32)
    assert f._der["v1"] == (base**x._val) * np.log(base) # type: ignore


def test_log():
    value = 4
    x = Variable("x", value)
    f = log(x)
    a = log(value)
    assert a == pytest.approx(np.log(value))
    assert f._val == np.log(value) # type: ignore
    assert f._der["v1"] == pytest.approx(0.25) # type: ignore


def test_exp():
    value = 67
    x = Variable("x", value)
    f = exp(x)
    a = exp(value)
    assert a == pytest.approx(np.exp(value))
    assert f._val == pytest.approx(np.exp(value), rel=1e-5) # type: ignore
    assert f._der["v1"] == f._val # type: ignore


def test_composition_val():
    value = np.pi / 6
    x = Variable("x", value)
    c = cos(x)
    s = sin(x)
    t = tan(x)
    e = exp(x)
    f = c * t + e
    _g = c + s
    assert isinstance(f, Trace)
    assert f._val == np.cos(value) * np.tan(value) + np.exp(value)


def test_basic_der():
    # Decorator function maker that can be used to create function variables
    def fm(f):
        def fun(x):
            return f(x)

        return fun

    value = 0.5
    assert gd.trace(fm(sin), value) == np.cos(value)
    assert gd.trace(fm(cos), value) == -np.sin(value)
    assert gd.trace(fm(tan), value) == 1 / (np.cos(value) * np.cos(value))


def test_composition_der():
    def f(x):
        return cos(x) * tan(x) + exp(x)

    value = 0.5
    der = cast(NDArray,gd.trace(f, value))
    assert der[0] == -1 * np.sin(value) * np.tan(value) + 1 / np.cos(value) + np.exp(
        value
    )


def test_string_input():
    matcher = re.escape("Input(s) must be Trace or scalar")
    with pytest.raises(TypeError, match=matcher):
        _f = sin("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = cos("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = tan("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = sinh("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = cosh("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = tanh("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = arcsin("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = arccos("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = arctan("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = sqrt("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = sigmoid("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = log("test") # type: ignore
    with pytest.raises(TypeError, match=matcher):
        _f = exp("test") # type: ignore


def test_arc_domains():
    x = Variable("x", 2)
    y = 2
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = arcsin(x)
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = arccos(x)
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = arcsin(y)
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = arccos(y)


def test_other_domains():
    x = Variable("x", -2)
    y = -2
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = log(x)
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = log(y)
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = sqrt(x)
    with pytest.raises(ValueError, match="Input out of domain"):
        _f = sqrt(y)


# pylint:disable=too-many-locals
def test_array_input():
    vals = [0.5, 0.2, 0.999]
    arr = [Variable("x", val) for val in vals]
    t1 = cast(List[Trace],sin(arr))
    t2 = cast(List[Trace],arcsin(arr))
    t3 = cast(List[Trace],cos(arr))
    t4 = cast(List[Trace],arccos(arr))
    t5 = cast(List[Trace],tan(arr))
    t6 = cast(List[Trace],arctan(arr))
    t7 = cast(List[Trace],exp(arr))
    t8 = cast(List[Trace],log(arr))
    t9 = cast(List[Trace],sinh(arr))
    t10 = cast(List[Trace],cosh(arr))
    t11 = cast(List[Trace],tanh(arr))
    t12 = cast(List[Trace],sqrt(arr))
    t13 = cast(List[Trace],sigmoid(arr))
    for idx, val in enumerate(vals):
        assert t1[idx].val == pytest.approx(np.sin(val))
        assert t2[idx].val == pytest.approx(np.arcsin(val))
        assert t3[idx].val == pytest.approx(np.cos(val))
        assert t4[idx].val == pytest.approx(np.arccos(val))
        assert t5[idx].val == pytest.approx(np.tan(val))
        assert t6[idx].val == pytest.approx(np.arctan(val))
        assert t7[idx].val == pytest.approx(np.exp(val))
        assert t8[idx].val == pytest.approx(np.log(val))
        assert t9[idx].val == pytest.approx(np.sinh(val))
        assert t10[idx].val == pytest.approx(np.cosh(val))
        assert t11[idx].val == pytest.approx(np.tanh(val))
        assert t12[idx].val == pytest.approx(np.sqrt(val))
        assert t13[idx].val == pytest.approx(1 / (1 + np.exp(-val)))
