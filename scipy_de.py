from scipy.optimize import differential_evolution, rosen

from diff_evolution.cec17_functions import cec17_test_func

dims = 10
bounds = [(-100, 100)] * dims

tested_funcs = [30]
# tested_funcs = range(2, 30)
for func_num in tested_funcs:
    global_opt_raw = open(f"data/cec2017/shift_data_{func_num}.txt").readline()
    global_opt = [float(v) for v in global_opt_raw.split()]
    global_opt = global_opt[:dims]

    def call_cec(x):
        fitness = cec17_test_func(x, dims=dims, func_num=func_num)
        return fitness[0]

    print(f"[{func_num}]")
    # print(f"Global optimum: {global_opt}")
    print(f"Global optimum value: {call_cec(global_opt)}")
    result = differential_evolution(call_cec, bounds)
    target_value = 100 * (func_num)
    print(f"Predicted {result.fun} vs truth {target_value}")
    print()
