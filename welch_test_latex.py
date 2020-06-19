import os

import click
import numpy as np
from scipy import stats


def get_results(alg1_results_path, alg2_results_path):
    alg1_results = np.genfromtxt(alg1_results_path, delimiter=",")[13]
    alg2_results = np.genfromtxt(alg2_results_path, delimiter=",")[13]

    alg1_mean = np.mean(alg1_results)
    alg2_mean = np.mean(alg2_results)

    return alg1_mean, alg2_mean


def compare_algorithms_sinlge_problem(
    alg1_results_path, alg2_results_path, confidence=0.95
):
    """
    Checks if alg1 is better (gives better result) then alg2 with given confidence
    Returns:
        True if alg1 is better than alg2
        False if alg2 is better than alg1
        None if H0 can't be rejected with given confidence
    """

    alg1_results = np.genfromtxt(alg1_results_path, delimiter=",")[13]
    alg2_results = np.genfromtxt(alg2_results_path, delimiter=",")[13]

    alg1_mean = np.mean(alg1_results)
    alg2_mean = np.mean(alg2_results)

    pvalue = stats.ttest_ind(alg1_results, alg2_results, equal_var=False)[1]

    if pvalue / 2 <= (1-confidence):
        return True if alg1_mean < alg2_mean else False

    return None


@click.command()
@click.option(
    "--alg1-dir",
    required=True,
    help="Algorithm 1 results folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--alg2-dir",
    required=True,
    help="Algorithm 2 results folder (assuming that its content is analogous to Algorithm 1 results folder)",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--dimension", "-d", required=True, help="Dimension compared", type=int,
)
@click.option(
    "--confidence",
    "-c",
    default=0.95,
    show_default=True,
    help="Welch's t-test confidence",
    type=float,
)
def compare_algorithms(alg1_dir, alg2_dir, dimension, confidence=0.95):
    counter = {True: 0, False: 0, None: 0}
    alg1_name = None
    alg2_name = None

    alg1_name = os.path.basename(os.path.normpath(alg1_dir))
    alg2_name = os.path.basename(os.path.normpath(alg2_dir))
    latex_lines = []
    for function in range(1, 31):
        f1 = os.path.join(alg1_dir, f"{alg1_name}_{function}_{dimension}.txt")
        f2 = os.path.join(alg2_dir, f"{alg2_name}_{function}_{dimension}.txt")
        result = compare_algorithms_sinlge_problem(f1, f2, confidence,)
        counter[result] += 1
        alg1_mean, alg2_mean = get_results(f1, f2)
        func_str = f"{function} " if function < 10 else f"{function}"
        alg_str = f"{alg1_mean:.2E}          & {alg2_mean:.2E}         "
        if result is True:
            alg_str = f"\\textbf{{{alg1_mean:.2E}}} & {alg2_mean:.2E}         "
        elif result is False:
            alg_str = f"{alg1_mean:.2E}          & \\colorbox{{pink}}\\textbf{{{alg2_mean:.2E}}}"
        latex_lines += [f"{func_str} & {alg_str}"]

    counter[f"{alg1_name} better"] = counter.pop(True)
    counter[f"{alg2_name} better"] = counter.pop(False)
    counter[f"Not resolved"] = counter.pop(None)
    print(counter)
    print()
    print("\\toprule")
    print(f"{{Function}} & {{{alg1_name}}} & {{{alg2_name}}} \\\\ \\midrule")
    print(" \\\\ \n".join(latex_lines) + " \\\\ \\bottomrule")


if __name__ == "__main__":
    compare_algorithms()
