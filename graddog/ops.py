"""
# :)
"""

from typing import Any, Callable, Dict, NewType, Optional, Tuple
import numpy as np


# pylint:disable=too-few-public-methods
class OneParentOp:
    """
    A one parent operation possibly using a parameter
    """

    __slots__ = ("val", "der", "double_der", "in_domain")

    def __init__(
        self,
        *,
        val: Callable[[Any, Any], Any],
        der: Optional[Callable[[Any, Any], Any]] = None,
        double_der: Optional[Callable[[Any, Any], Any]] = None,
        in_domain: Optional[Callable[[Any, Any], bool]] = None,
    ):
        self.val = val
        self.der = der
        self.double_der = double_der
        self.in_domain = in_domain


# pylint:disable=too-few-public-methods
class TwoParentOp:
    """
    A two parent operation
    """

    __slots__ = ("val", "der", "double_der", "in_domain")

    def __init__(
        self,
        *,
        val: Callable[[Any, Any], Any],
        der: Optional[Callable[[Any, Any], Tuple[Any, Any]]] = None,
        double_der: Optional[Callable[[Any, Any], np.ndarray[Any]]] = None,
        in_domain: Optional[Callable[[Any, Any], bool]] = None,
    ):
        self.val = val
        self.der = der
        self.double_der = double_der
        self.in_domain = in_domain


OpName = NewType("OpName", str)


# pylint:disable=chained-comparison, too-few-public-methods
class Ops:
    """
    all of the math operations currently implemented
    include a value and derivative rule for each operation
    can include UNARY operators (with or without a numerical parameters)
      which are called 'one_parent'
    can include BINARY operators which are called 'two_parent'

    _R denotes that the operator is applied with a trace to the RIGHT
    certain _R operators like __radd__ and __rmul__ actually have no difference
      from __add__ and __mul__
    but __rsub__ and __rdiv__, for example, have different derivatives
      than __sub__ and __div__
    """

    #### used internally
    __VAL_KEY = "val"
    __DER_KEY = "der"
    __DOUBLE_DER_KEY = "double_der"
    __IN_DOMAIN_KEY = "in_domain"

    add = OpName("+")
    sub = OpName("-")
    sub_R = OpName("-R")
    mul = OpName("*")
    div = OpName("/")
    div_R = OpName("/R")
    power = OpName("^")
    sin = OpName("sin")
    cos = OpName("cos")
    tan = OpName("tan")
    exp = OpName("exp")
    log = OpName("log")
    sqrt = OpName("sqrt")
    sigm = OpName("sigm")
    sinh = OpName("sinh")
    cosh = OpName("cosh")
    tanh = OpName("tanh")
    arcsin = OpName("arcsin")
    arccos = OpName("arccos")
    arctan = OpName("arctan")
    abs = OpName("abs")

    one_parent_rules: Dict[OpName, OneParentOp] = {
        add: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: t + param,
                __DER_KEY: lambda t, param: 1.0,
                __DOUBLE_DER_KEY: lambda t, param: 0,
            }
        ),
        sub: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: t - param,
                __DER_KEY: lambda t, param: 1.0,
                __DOUBLE_DER_KEY: lambda t, param: 0,
            }
        ),
        sub_R: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: param - t,
                __DER_KEY: lambda t, param: -1.0,
                __DOUBLE_DER_KEY: lambda t, param: 0,
            }
        ),
        mul: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: t * param,
                __DER_KEY: lambda t, param: param,
                __DOUBLE_DER_KEY: lambda t, param: 0,
            }
        ),
        div: OneParentOp(
            **{
                __IN_DOMAIN_KEY: lambda t, param: param != 0,
                __VAL_KEY: lambda t, param: t / param,
                __DER_KEY: lambda t, param: 1 / param,
                __DOUBLE_DER_KEY: lambda t, param: 0,
            }
        ),
        div_R: OneParentOp(
            **{
                __IN_DOMAIN_KEY: lambda t, param: t != 0,
                __VAL_KEY: lambda t, param: param / t,
                __DER_KEY: lambda t, param: -param / (t**2),
                __DOUBLE_DER_KEY: lambda t, param: 2 * param / (t**3),
            }
        ),
        power: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: t**param,
                __DER_KEY: lambda t, param: param * t ** (param - 1),
                __DOUBLE_DER_KEY: lambda t, param: param
                * (param - 1)
                * (t ** (param - 2)),
            }
        ),
        sin: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: np.sin(t),
                __DER_KEY: lambda t, param: np.cos(t),
                __DOUBLE_DER_KEY: lambda t, param: -np.sin(t),
            }
        ),
        arcsin: OneParentOp(
            **{
                __IN_DOMAIN_KEY: lambda t, param: t >= -1 and t <= 1,
                __VAL_KEY: lambda t, param: np.arcsin(t),
                __DER_KEY: lambda t, param: 1 / (np.sqrt(1 - t**2)),
            }
        ),
        cos: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: np.cos(t),
                __DER_KEY: lambda t, param: -np.sin(t),
                __DOUBLE_DER_KEY: lambda t, param: -np.cos(t),
            }
        ),
        arccos: OneParentOp(
            **{
                __IN_DOMAIN_KEY: lambda t, param: t >= -1 and t <= 1,
                __VAL_KEY: lambda t, param: np.arccos(t),
                __DER_KEY: lambda t, param: -1 / (np.sqrt(1 - t**2)),
            }
        ),
        tan: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: np.tan(t),
                __DER_KEY: lambda t, param: 1 / (np.cos(t) ** 2),
            }
        ),
        arctan: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: np.arctan(t),
                __DER_KEY: lambda t, param: 1 / (1 + t**2),
            }
        ),
        exp: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: np.power(param, t),
                __DER_KEY: lambda t, param: np.power(param, t) * np.log(param),
                __DOUBLE_DER_KEY: lambda t, param: np.power(param, t)
                * np.log(param)
                * np.log(param),
            }
        ),
        log: OneParentOp(
            **{
                __IN_DOMAIN_KEY: lambda t, param: t > 0 and param > 0,
                __VAL_KEY: lambda t, param: np.log(t) / np.log(param),
                __DER_KEY: lambda t, param: 1 / (np.log(param) * t),
                __DOUBLE_DER_KEY: lambda t, param: -1 / (np.log(param) * t**2),
            }
        ),
        sqrt: OneParentOp(
            **{
                __IN_DOMAIN_KEY: lambda t, param: t >= 0,
                __VAL_KEY: lambda t, param: t**0.5,
                __DER_KEY: lambda t, param: 1 / (2 * t**0.5),
                __DOUBLE_DER_KEY: lambda t, param: -1 / (4 * t**1.5),
            }
        ),
        sigm: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: 1 / (1 + np.exp(-t)),
                __DER_KEY: lambda t, param: np.exp(-t) / ((1 + np.exp(t)) ** 2),
            }
        ),
        sinh: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: (np.exp(t) - np.exp(-t)) / 2,
                __DER_KEY: lambda t, param: (np.exp(t) + np.exp(-t)) / 2,
                __DOUBLE_DER_KEY: lambda t, param: (np.exp(t) - np.exp(-t)) / 2,
            }
        ),
        cosh: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: (np.exp(t) + np.exp(-t)) / 2,
                __DER_KEY: lambda t, param: (np.exp(t) - np.exp(-t)) / 2,
                __DOUBLE_DER_KEY: lambda t, param: (np.exp(t) + np.exp(-t)) / 2,
            }
        ),
        tanh: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: (np.exp(t) - np.exp(-t))
                / (np.exp(t) + np.exp(-t)),
                __DER_KEY: lambda t, param: 4 / ((np.exp(t) + np.exp(-t)) ** 2),
            }
        ),
        abs: OneParentOp(
            **{
                __VAL_KEY: lambda t, param: t if t >= 0 else -t,
                __DER_KEY: lambda t, param: 1 if t >= 0 else -1,
                __DOUBLE_DER_KEY: lambda t, param: 0,
            }
        ),
    }

    two_parent_rules: Dict[OpName, TwoParentOp] = {
        add: TwoParentOp(
            **{
                __VAL_KEY: lambda t1, t2: t1 + t2,
                __DER_KEY: lambda t1, t2: (1.0, 1.0),
                __DOUBLE_DER_KEY: lambda t1, t2: np.array([[0, 0], [0, 0]]),
            }
        ),
        sub: TwoParentOp(
            **{
                __VAL_KEY: lambda t1, t2: t1 - t2,
                __DER_KEY: lambda t1, t2: (1.0, -1.0),
                __DOUBLE_DER_KEY: lambda t1, t2: np.array([[0, 0], [0, 0]]),
            }
        ),
        mul: TwoParentOp(
            **{
                __VAL_KEY: lambda t1, t2: t1 * t2,
                __DER_KEY: lambda t1, t2: (t2, t1),
                __DOUBLE_DER_KEY: lambda t1, t2: np.array([[0, 1], [1, 0]]),
            }
        ),
        div: TwoParentOp(
            **{
                __VAL_KEY: lambda t1, t2: t1 / t2,
                __DER_KEY: lambda t1, t2: (1 / t2, -t1 / (t2**2)),
                __DOUBLE_DER_KEY: lambda t1, t2: np.array(
                    [[0, -1 / (t2**2)], [-1 / (t2**2), 2 * t1 / (t2**3)]]
                ),
            }
        ),
        power: TwoParentOp(
            **{
                __VAL_KEY: lambda t1, t2: t1**t2,
                __DER_KEY: lambda t1, t2: (
                    t2 * (t1 ** (t2 - 1)),
                    (t1**t2) * np.log(t1),
                ),
                __DOUBLE_DER_KEY: lambda t1, t2: np.array(
                    [
                        [
                            t2 * (t2 - 1) * (t1 ** (t2 - 2)),
                            (t1 ** (t2 - 1)) * (t2 * np.log(t1) + 1),
                        ],
                        [
                            (t1 ** (t2 - 1)) * (t2 * np.log(t1) + 1),
                            np.log(t1) * np.log(t1) * t1**t2,
                        ],
                    ]
                ),
            }
        ),
    }

    @classmethod
    def _deriv_one_parent(cls, op: OpName, cur_val, cur_param):
        """derivative of a trace with one parent"""
        rule = cls.one_parent_rules[op]
        if rule.der is None:
            raise AttributeError(f"No der in {op}")
        return rule.der(cur_val, cur_param)

    @classmethod
    def _deriv_two_parents(cls, op: OpName, val_t1, val_t2):
        """derivative of a trace with two parents"""
        rule = cls.two_parent_rules[op]
        if rule.der is None:
            raise AttributeError(f"No der in {op}")
        return rule.der(val_t1, val_t2)

    @classmethod
    def _val_one_parent(cls, op: OpName, cur_val, cur_param):
        """value of a trace with one parent and optional scalar parameter"""
        rule = cls.one_parent_rules[op]
        if rule.val is None:
            raise AttributeError(f"No val in {op}")
        return rule.val(cur_val, cur_param)

    @classmethod
    def _val_two_parents(cls, op: OpName, val_t1, val_t2):
        """value of a trace with two parents"""
        rule = cls.two_parent_rules[op]
        if rule.val is None:
            raise AttributeError(f"No val in {op}")
        return rule.val(val_t1, val_t2)

    @classmethod
    def _double_deriv_one_parent(cls, op: OpName, cur_val, cur_param):
        """
        Give the full second derivatives of this one parent
        operation with parent t and optional parameter param
        """
        rule = cls.one_parent_rules[op]
        if rule.double_der is None:
            raise AttributeError(f"No double_der in {op}")
        return rule.double_der(cur_val, cur_param)

    @classmethod
    def _double_deriv_two_parents(cls, op: OpName, val_t1, val_t2):
        """
        Give the full second derivatives of this two parent
        operation with parents t1 and t2
        """
        rule = cls.two_parent_rules[op]
        if rule.double_der is None:
            raise AttributeError(f"No double_der in {op}")
        return rule.double_der(val_t1, val_t2)

    @classmethod
    def _in_domain_one_parent(cls, op: OpName, cur_val, cur_param):
        """
        in the domain of this op
        """
        rule = cls.one_parent_rules[op]
        if rule.in_domain is None:
            raise AttributeError(f"No in_domain in {op}")
        return rule.in_domain(cur_val, cur_param)

    @classmethod
    def _in_domain_two_parents(cls, op: OpName | str, val_t1, val_t2):
        """
        in the domain of this op
        """
        if isinstance(op, str):
            op = OpName(op)
        rule = cls.two_parent_rules[op]
        if rule.in_domain is None:
            raise AttributeError(f"No in_domain in {op}")
        return rule.in_domain(val_t1, val_t2)
