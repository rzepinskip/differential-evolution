import os

from scipy import stats
import numpy as np
import click

def compare_algorithms_sinlge_problem(alg1_results_path, alg2_results_path, confidence = 0.95):
    '''
    Checks if alg1 is better (gives better result) then alg2 with given confidence
    Returns:
        True if alg1 is better than alg2
        False if alg2 is better than alg1
        None if H0 can't be rejected with given confidence
    '''

    alg1_results = np.genfromtxt(alg1_results_path, delimiter=',')[13]
    alg2_results = np.genfromtxt(alg2_results_path, delimiter=',')[13]

    alg1_mean = np.mean(alg1_results)
    alg2_mean = np.mean(alg2_results)

    pvalue = stats.ttest_ind(alg1_results, alg2_results, equal_var=False)[1]

    if pvalue <= (1-confidence)/2:
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
    "--confidence", "-c", default=0.95, show_default=True, help="Welch's t-test confidence", type=float
)
def compare_algorithms(alg1_dir, alg2_dir, confidence = 0.95):
    counter = {
        True: 0,
        False: 0,
        None: 0
    }
    alg1_name = None
    alg2_name = None
    for f1 in os.listdir(alg1_dir):
        if f1.endswith(".txt"):
            if not alg1_name:
                alg1_name = f1[:f1.find('_')]
            file_subname = f1[f1.find('_'):]
            for f2 in os.listdir(alg2_dir):
                if f2.endswith(file_subname):
                    if not alg2_name:
                        alg2_name = f2[:f2.find('_')]
                    result = compare_algorithms_sinlge_problem(
                        os.path.join(alg1_dir, f1),
                        os.path.join(alg2_dir, f2),
                        confidence
                    )
                    counter[result] += 1


    counter[f'{alg1_name} better'] = counter.pop(True)
    counter[f'{alg2_name} better'] = counter.pop(False)
    counter[f'Not resolved'] = counter.pop(None)
    print(counter)

if __name__ == "__main__":
    compare_algorithms()