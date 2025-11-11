""":)"""

from typing import Literal, Optional, Tuple, cast
import numpy as np
from numpy.typing import NDArray
from graddog.modes import Mode
from graddog.trace import Variable
from graddog.compgraph import CompGraph


# pylint:disable=invalid-name, too-many-statements, too-many-branches
def trace(
    f,
    seed,
    mode: Optional[Mode] | Literal["reverse"] | Literal["forward"] = None,
    return_second_deriv=False,
    verbose=False,
    **kwargs,
):
    """
    f : a function
    seed: a vector/list of scalars. If f is single-dimensional, seed can be a scalar

    Optional parameter mode
        When mode = None, this function infers the more efficient mode
        from the number of input and output variables

    Optional parameter return_second_deriv (default is False)
        When return_second_deriv = True, this function returns f' AND f''

    f can be
    f: R --> R using explicit single-variable input
    f: Rm --> R using explicit multi-variable input
    f: R --> Rn using explicit single-variable input and explicit vector output
    f: Rm --> R using explicit vector input
    f: Rm --> Rn using explicit vector input and explicit vector output
    f: Rm --> Rn using explicit multi-variable input and explicit vector output
    f: Rm --> Rn using IMPLICIT vector input and IMPLICIT vector output
    """

    if mode == "forward":
        mode = Mode.FORWARD
    if mode == "reverse":
        mode = Mode.REVERSE

    ######################## make your Variable objects #########################
    # for now, always reset the CompGraph when tracing a new function
    CompGraph.reset()

    # infer the dimensionality of the input
    try:  # if multidimensional input
        M = len(seed)  # get the dimension of the input
        seed = np.array(seed)
    except TypeError:  # if single-dimensional input
        M = 1
        seed = np.array([seed])
    if verbose:
        print(f"Inferred {M}-dimensional input")

    # create new variables
    names = [f"v{i+1}" for i in range(M)]
    new_variables = np.array([Variable(names[i], seed[i]) for i in range(M)])
    #############################################################################

    ################ Trace the function ##############
    if verbose:
        print("Scanning the computational graph...")
    # Apply f to the new variables
    # Infer the way f was meant to be applied
    if M > 1:
        # multi-variable input

        try:
            # as a vector
            output = f(new_variables, **kwargs)
            if verbose:
                print("...inferred the input is a vector...")
        except TypeError:
            # as variables
            output = f(*new_variables, **kwargs)
            if verbose:
                print("...inferred the inputs are variables...")

    else:
        # single-variable input
        output = f(new_variables[0], **kwargs)
        if verbose:
            print("...inferred the input is a variable...")
    if verbose:
        print("...finished")
    ############################################

    ################ Get Outputs #################
    try:
        N = len(output)
    except AttributeError:
        N = 1
        output = [output]
    except TypeError:
        N = 1
        output = [output]
    if verbose:
        print(f"Inferred {N}-dimensional output")
        print(output)
    ##############################################

    ##################### Second Derivative #########################
    if return_second_deriv:
        if mode is not None and mode != Mode.REVERSE:
            raise ValueError(
                "Second derivative is automatically calculated in reverse mode"
            )
        if N > 1:
            raise ValueError("Can only compute second derivative for scalar output f")
        if verbose:
            print("Computing reverse mode first AND second derivative...")
        return CompGraph.hessian(output, verbose)
    ######################################################

    ####### get user-defined mode or infer the more efficient mode ##########
    if mode is None:
        if M > N:
            mode = Mode.REVERSE
        else:
            mode = Mode.FORWARD
    elif isinstance(mode, Mode):
        pass
    elif isinstance(mode, str):
        mode = Mode(mode.lower())
    else:
        raise ValueError("Didnt recognize mode, should be forward or reverse")
    ######################################################################

    ############## First Derivative ####################
    if verbose:
        print(f"Computing {mode} mode derivative...")
    if mode == Mode.FORWARD:
        return CompGraph.forward_mode(output, verbose)
    if mode == Mode.REVERSE:
        return CompGraph.reverse_mode(output, verbose)
    raise ValueError("Didnt recognize mode, should be forward or reverse")
    ########################################################


def derivatives_only(f, seed, **kwargs) -> NDArray:
    """
    f : a function
    seed: a vector/list of scalars. If f is single-dimensional, seed can be a scalar

    this function returns f' only

    f can be
    f: R --> R using explicit single-variable input
    f: Rm --> R using explicit multi-variable input
    f: R --> Rn using explicit single-variable input and explicit vector output
    f: Rm --> R using explicit vector input
    f: Rm --> Rn using explicit vector input and explicit vector output
    f: Rm --> Rn using explicit multi-variable input and explicit vector output
    f: Rm --> Rn using IMPLICIT vector input and IMPLICIT vector output
    """
    return cast(
        NDArray,
        trace(
            f=f,
            seed=seed,
            return_second_deriv=False,
            verbose=False,
            **kwargs,
        ),
    )


def derivatives_and_hessians(f, seed, **kwargs) -> Tuple[NDArray, NDArray]:
    """
    f : a function
    seed: a vector/list of scalars. If f is single-dimensional, seed can be a scalar

    this function returns f' AND f''

    f can be
    f: R --> R using explicit single-variable input
    f: Rm --> R using explicit multi-variable input
    f: R --> Rn using explicit single-variable input and explicit vector output
    f: Rm --> R using explicit vector input
    f: Rm --> Rn using explicit vector input and explicit vector output
    f: Rm --> Rn using explicit multi-variable input and explicit vector output
    f: Rm --> Rn using IMPLICIT vector input and IMPLICIT vector output
    """
    traced = trace(
        f=f,
        seed=seed,
        return_second_deriv=True,
        verbose=False,
        **kwargs,
    )
    match traced:
        case (derivatives, hessian):
            return (derivatives, hessian)
        case None:
            raise ValueError("Gave back nothing during trace")
        case f_:  # pylint:disable = unused-variable
            raise ValueError(
                "Only gave back the expectation values but not the covariances"
            )
