"""
# :)
"""

from graddog.ops import Ops, OpName


# pylint:disable=protected-access
def deriv_one_parent(t, op: OpName | str, param=None):
    """derivative of a trace with one parent"""
    if isinstance(op, str):
        op = OpName(op)
    try:
        d_op_dt = Ops._deriv_one_parent(op, t.val, param)
        return {t._trace_name: d_op_dt}
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement derivative of operation {op}")


# pylint:disable=protected-access
def deriv_two_parents(t1, op, t2):
    """derivative of a trace with two parents"""
    try:
        d_op_dt1, d_op_dt2 = Ops._deriv_two_parents(op, t1.val, t2.val)
        return {t1._trace_name: d_op_dt1, t2._trace_name: d_op_dt2}
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement derivative of operation {op}")


def val_one_parent(t, op, param=None):
    """value of a trace with one parent and optional scalar parameter"""
    try:
        # t is a trace
        return Ops._val_one_parent(op, t.val, param)
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement value of operation {op}")


def val_two_parents(t1, op, t2):
    """value of a trace with two parents"""
    try:
        return Ops._val_two_parents(op, t1.val, t2.val)
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement value of operation {op}")


def deriv(t, op, other=None):
    """
    derivative of a trace
    """
    if other is None:
        return deriv_one_parent(t, op)
    try:
        # if other is a trace
        _other_val = other.val
        return deriv_two_parents(t, op, other)
    except AttributeError:
        # if other is a scalar
        return deriv_one_parent(t, op, other)


def val(t, op, other=None):
    """value of a trace"""
    if other is None:
        return val_one_parent(t, op)
    try:
        # if other is a trace
        _other_val = other.val
        return val_two_parents(t, op, other)
    except AttributeError:
        # if other is a scalar
        return val_one_parent(t, op, other)


def double_deriv(t1, t2, t3):
    """
    Returns double derivative of t1 w.r.t. both t2 and t3

    References the parent traces of t1 because the order of the arguments matters
    """
    parents = t1._parents
    if len(parents) == 1:
        return new_double_deriv_one_parent(t2, t1._op, t1._param)
    double_deriv_t2t3 = new_double_deriv_two_parents(parents[0], t1._op, parents[1])
    if t2._trace_name != t3._trace_name:
        return double_deriv_t2t3[0, 1]
    if t2._trace_name == parents[0]._trace_name:
        return double_deriv_t2t3[0, 0]
    return double_deriv_t2t3[1, 1]


def new_double_deriv_one_parent(t, op, param=None):
    """
    Give the full second derivatives of this one parent
    operation with parent t and optional parameter param
    """
    try:
        return Ops._double_deriv_one_parent(op, t.val, param)
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement double derivative of operation {op}")


def new_double_deriv_two_parents(t1, op, t2):
    """
    Give the full second derivatives of this two parent
    operation with parents t1 and t2
    """
    try:
        return Ops._double_deriv_two_parents(op, t1.val, t2.val)
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement double derivative of operation {op}")


def in_domain(query_val, op, param=None):
    """
    in the domain of this op
    """
    try:
        return Ops._in_domain_one_parent(op, query_val, param)
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement in domain of operation {op}")


def in_domain_two(query_val, op, query_val_other):
    """
    in the domain of this op
    """
    try:
        return Ops._in_domain_two_parents(op, query_val, query_val_other)
    except (KeyError, AttributeError):
        # pylint:disable=raise-missing-from
        raise ValueError(f"need to implement in domain of operation {op}")
