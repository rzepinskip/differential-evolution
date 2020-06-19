import os
import csv

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

    result = None

    if pvalue / 2 <= (1-confidence):
        if alg1_mean < alg2_mean:
            result = True
        else:
            result = False

    return alg1_mean, alg2_mean, result


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
    "--output",
    required=True,
    help="Output file folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "--confidence", "-c", default=0.95, show_default=True, help="Welch's t-test confidence", type=float
)
def compare_algorithms(alg1_dir, alg2_dir, output, confidence = 0.95):
    counter = {
        True: 0,
        False: 0,
        None: 0
    }

    alg1_name = os.path.basename(alg1_dir)
    alg2_name = os.path.basename(alg2_dir)

    out_file_path = os.path.join(output, f"{alg1_name}_vs_{alg2_name}.csv")

    alg1_better_name = f"{alg1_name} better than {alg2_name}"
    fieldnames = ['Func.', alg1_name, alg2_name, alg1_better_name]

    results = []

    for f1 in os.listdir(alg1_dir):
        if f1.endswith(".txt"):
            file_subname = f1[f1.find('_'):]
            for f2 in os.listdir(alg2_dir):
                if f2.endswith(file_subname):
                    alg1_result, alg2_result, alg1_better = compare_algorithms_sinlge_problem(
                        os.path.join(alg1_dir, f1),
                        os.path.join(alg2_dir, f2),
                        confidence
                    )
                    counter[alg1_better] += 1
                    function_num = int(file_subname[1:file_subname.rfind('_')])

                    res = {
                        'Func.': function_num,
                        alg1_name: f"{alg1_result:.2E}",
                        alg2_name: f"{alg2_result:.2E}",
                        alg1_better_name: alg1_better
                    }
                    results.append(res)

    results.sort(key = lambda x: x['Func.'])

    with open(out_file_path, mode="w", newline="\n") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for res in results:
            writer.writerow(res)

    counter[f'{alg1_name} better'] = counter.pop(True)
    counter[f'{alg2_name} better'] = counter.pop(False)
    counter[f'Not resolved'] = counter.pop(None)
    print(counter)

if __name__ == "__main__":
    compare_algorithms()