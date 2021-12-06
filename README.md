# Confident off-policy evaluation and selection through self-normalized importance weighting

The package provided here contains implementation of commonly used estimators
for off-policy evaluation, corresponding high probability lower bounds on the
value, and a new Efron-Stein type bound for the self-normalized estimator
described in ["Kuzborskij, I., Vernade, C., Gyorgy, A., & Szepesvári, C. (2021,
March). Confident off-policy evaluation and selection through self-normalized
importance weighting. In International Conference on Artificial Intelligence and
Statistics (pp. 640-648). PMLR."](https://arxiv.org/abs/2006.10460). The package
also contains an off-policy selection benchmark used in the paper above and a
setup to reproduce most of the results.

## Setup

To install necessary packages, execute:

```
python3 -m venv /tmp/eslb_venv
source /tmp/eslb_venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r offpolicy_selection_eslb/requirements.txt
```

## Usage

To run the benchmark on the UCI datasets considered in the paper, execute:

```
python3 demo/benchmark.py --dataset_type=uci_all --n_trials=10 --delta=0.01
```

This will run a full benchmark. You can replace `uci_all` in the above with
`uci_medium` or `uci_small` for smaller subsets of UCI datasets.

## Examples

In `colabs/eslb_synthetic_example.ipynb` you can find a standalone example
demonstrating the usage of our estimator on synthetic data.

## Usage as a library

Module `estimators` contains several classes which implement estimators
which can be used in a standalone fashion.

In particular:

* `ESLB` implements an Efron-Stein high probability bound for off-policy
evaluation (Theorem 1 and Algorithm 1).

* `IWEstimator` implements the standard importance weighted estimator (IW).

* `SNIWEstimator` implements a self-normalized version of IW.

* `IWLambdaEmpBernsteinEstimator` implements a high probability empirical
Bernstein bound for λ-corrected IW (the estimator is stabilized by adding λ
to the denominator) with appropriate tuning of λ (see Proposition 1).

See `colabs/eslb_synthetic_example.ipynb` for a usage example.


## Citing this work

If you use this code, please cite our work:

```
@InProceedings{pmlr-v130-kuzborskij21a,
  title =        {Confident Off-Policy Evaluation and Selection through Self-Normalized Importance Weighting },
  author =       {Kuzborskij, Ilja and Vernade, Claire and Gyorgy, Andras and Szepesvari, Csaba},
  booktitle =    {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
  year =         {2021}
}
```

## Disclaimer

This is not an official Google product.
