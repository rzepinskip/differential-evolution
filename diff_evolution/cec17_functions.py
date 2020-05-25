import os
from ctypes import CDLL, POINTER, c_double, c_int


# Based on https://github.com/lacerdamarcelo/cec17_python/blob/master/cec17_functions.py
def cec17_test_func(
    x,
    dims=2,
    objectives=1,
    func_num=1,
    pdf_numeration=True,
    dll_path=CDLL(os.path.abspath("diff_evolution/cec17_test_func.so")),
):
    # The 2nd function was removed from CEC-2017. Because of that an inconsistency in numbering has occurred. In `cec17_test_func.c` the original numbering has been preserved.
    # In https://github.com/P-N-Suganthan/CEC2017-BoundContrained/blob/master/Definitions%20of%20%20CEC2017%20benchmark%20suite%20final%20version%20updated.pdf
    # the 2nd function was completely removed and other functions have been renumbered (3 -> 2, 4 -> 3, etc.)
    if pdf_numeration and func_num > 1:
        func_num +=1

    if func_num not in (set(range(1, 31)) - set([2])):
        raise ValueError(f"Function {func_num} is not defined")

    if dims not in {2, 10, 30, 50}:
        raise ValueError(f"There is no data for {dims} dimension")

    functions = dll_path
    x_pointer_type = POINTER(c_double * dims)
    f_pointer_type = POINTER(c_double * objectives)
    dims_type = c_int
    objectives_type = c_int
    func_num_type = c_int
    functions.cec17_test_func.argtypes = [
        x_pointer_type,
        f_pointer_type,
        dims_type,
        objectives_type,
        func_num_type,
    ]
    functions.cec17_test_func.restype = None
    x_ctype = (c_double * dims)()
    for i, value in enumerate(x):
        x_ctype[i] = value
    f_ctype = (c_double * objectives)()
    for i in range(objectives):
        f_ctype[i] = 0
    functions.cec17_test_func(
        x_pointer_type(x_ctype), f_pointer_type(f_ctype), dims, objectives, func_num
    )

    f = [0] * objectives
    for i in range(len(f)):
        f[i] = f_ctype[i]
    return f
