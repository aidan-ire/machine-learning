# Machine learning (self-directed)

This is self-directed study, not a university module.

| Project | Contents |
|---|---|
| [01_linear_regression_gradient_descent/lab1.ipynb](01_linear_regression_gradient_descent/lab1.ipynb) | linear regression on an hours-studied vs scores dataset. The cost function and gradient descent are written from scratch with NumPy, then the fitted line is compared against scikit-learn's `LinearRegression` |
| [02_logistic_regression_diabetes/log_reg_diabetes.ipynb](02_logistic_regression_diabetes/log_reg_diabetes.ipynb) | logistic regression on the Pima Indians diabetes dataset with a 75/25 train/test split, followed by a confusion matrix heatmap, a classification report and a ROC curve with AUC |
| [03_char_transformer_shakespeare/](03_char_transformer_shakespeare/) | a character-level GPT-style transformer trained on Shakespeare's complete works, built by following Andrej Karpathy's "Let's build GPT" tutorial. `doc.py` trains it and `generate.py` loads the weights and writes new text |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Data

- `01_linear_regression_gradient_descent/lab1.csv` — `Hours` and `Scores` columns, read from the
  notebook's own folder.
- `02_logistic_regression_diabetes/data/pima_diabetes.csv` — a local copy of the dataset. The notebook
  downloads it from
  `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`.

## Known issues

- `lab1.ipynb`: the last cell uses `LinearRegression` without importing it. Add
  `from sklearn.linear_model import LinearRegression` before running that cell.
- `log_reg_diabetes.ipynb`: the first cell saves the download to `../LAB 2/data/`, which is not this
  folder's name, so running it creates a separate `LAB 2` folder next to this one. The rest of the
  notebook reads from that path and runs normally.
