# Support Vector Machines from the Dual Optimization Problem

An executable study of support vector machines (SVMs), built around the dual
quadratic program rather than a black-box estimator. The notebook implements
kernelized binary classification, extends it to multiclass prediction, and
evaluates one-vs-rest and one-vs-one strategies with confusion matrices.

## Results

On the notebook's held-out dataset, the recorded runs achieve:

| Strategy | Training accuracy | Test accuracy |
| --- | ---: | ---: |
| One-vs-rest | 88.08% | 84.70% |
| One-vs-one | 89.83% | 87.20% |

These are captured outputs from the committed notebook and may change if the
data split or hyperparameters are modified.

## Engineering highlights

- Formulates SVM training as a constrained quadratic program with CVXOPT
- Supports nonlinear decision boundaries through kernel functions
- Decomposes multiclass prediction into independently testable binary models
- Visualizes classification quality with scikit-learn confusion matrices
- Exposes the effect of the soft-margin parameter \(C\)

## Quick start

```bash
git clone https://github.com/takakhoo/Classification-via-Support-Vector-Machines.git
cd Classification-via-Support-Vector-Machines
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter lab "Classification Support Vector Machine.ipynb"
```

[Open the executed notebook](Classification%20Support%20Vector%20Machine.ipynb)

## Scope

This repository prioritizes inspectable mathematics and experimentation. For a
production classifier, add deterministic data splits, automated tests, model
serialization, and a stable inference interface.
