# Wheat Seed Classification

A machine learning project that classifies wheat kernels into one of three varieties (Kama, Rosa, Canadian) from seven geometric measurements. The work covers exploratory data analysis, outlier handling, a scikit-learn pipeline (imputation, Yeo-Johnson transform, scaling, PCA, Random Forest), hyperparameter tuning, and a saved model for inference.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Making Predictions](#making-predictions)
- [Limitations and Notes](#limitations-and-notes)
- [Possible Improvements](#possible-improvements)

## Dataset

The data is the UCI **Seeds** dataset: 210 samples, 70 for each of three wheat varieties. The kernels were measured with a soft X-ray technique.

| Feature | Description |
|---|---|
| `Area` | Kernel area |
| `Perimeter` | Kernel perimeter |
| `Compactness` | 4πA / P² |
| `Length of kernel` | Kernel length |
| `Width of kernel` | Kernel width |
| `Asymmetry coefficient` | Kernel asymmetry |
| `Length of kernel groove` | Length of the kernel groove |
| `Class (1, 2, 3)` | **Target**: wheat variety |

The raw CSV (`seeds_dataset (2).csv`) has two empty trailing columns, `Unnamed: 8` and `Unnamed: 9`, which the notebook drops. There are no missing values and no duplicate rows, and the classes are balanced.

## Project Structure

```
.
├── README.md
├── requirements.txt        # Python dependencies
└── wheat seed/
    ├── seeds_dataset (2).csv   # Raw dataset (210 rows)
    ├── wheat3.ipynb            # EDA, preprocessing, training, tuning, model export
    ├── wheat3_pred.ipynb       # Loads the saved model and predicts on new samples
    └── best_model.pkl          # Serialized best pipeline from GridSearchCV
```

## Methodology

### 1. Exploratory Data Analysis (`wheat3.ipynb`)

- Inspected the data with `head`, `describe`, `info`, and null and duplicate checks.
- Plotted the target distribution, per-feature distributions, and an Area vs. Length of kernel scatter plot.
- Drew box plots for every feature. `Compactness` and `Asymmetry coefficient` showed a few outliers.

### 2. Outlier Removal

Rows outside the 1.5 × IQR fences were removed for `Compactness` and then for `Asymmetry coefficient`. This takes the dataset from 210 to **205 rows**.

### 3. Multicollinearity Check

The correlation matrix and Variance Inflation Factors show very strong collinearity. `Area` and `Perimeter` correlate at about 0.99, and VIFs run from roughly 1,300 to over 20,000. Only `Asymmetry coefficient` is low, at about 9.9. This motivated using PCA.

### 4. Model Pipeline

An 80/20 train/test split (`random_state=42`) feeds a single scikit-learn `Pipeline`:

| Step | Component | Purpose |
|---|---|---|
| 1 | `SimpleImputer(strategy='mean')` | Fills missing values at inference time |
| 2 | `PowerTransformer(method='yeo-johnson')` | Makes feature distributions more Gaussian |
| 3 | `StandardScaler()` | Standardizes features |
| 4 | `PCA(n_components=0.95)` | Keeps 95% of variance and removes collinearity |
| 5 | `RandomForestClassifier()` | Classifier |

### 5. Hyperparameter Tuning

`GridSearchCV` (5-fold CV, accuracy scoring, 9 candidates, 45 fits) searches:

- `classifier__n_estimators`: `[50, 100, 150]`
- `classifier__max_depth`: `[None, 10, 20]`

The best estimator is saved to `best_model.pkl` with `pickle`.

## Results

Evaluation is on the held-out test set of 41 samples.

**Baseline pipeline (default Random Forest)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | 1.00 | 0.85 | 0.92 | 13 |
| 2 | 1.00 | 1.00 | 1.00 | 12 |
| 3 | 0.89 | 1.00 | 0.94 | 16 |
| **Accuracy** | | | **0.95** | 41 |

**Tuned model (`best_model.pkl`)**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 1 | 0.92 | 0.92 | 0.92 | 13 |
| 2 | 1.00 | 1.00 | 1.00 | 12 |
| 3 | 0.94 | 0.94 | 0.94 | 16 |
| **Accuracy** | | | **0.95** | 41 |

Tuning did not change overall accuracy. It did make the per-class scores more even.

## Getting Started

### Prerequisites

- Python 3.11 (the notebooks were run on 3.11)
- Jupyter Notebook or JupyterLab

### Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the versions the project was developed with. `best_model.pkl` was saved with scikit-learn 1.5.0, so keep that version to load it reliably, because pickled models can break across versions. If loading fails, re-run `wheat3.ipynb` to regenerate the file.

### Running

1. Open `wheat3.ipynb` and run all cells to reproduce the analysis, training, and `best_model.pkl`.
2. Open `wheat3_pred.ipynb` to run inference with the saved model.

Run the notebooks from inside the `wheat seed/` folder (`cd "wheat seed"`, then start Jupyter). They use relative paths for the CSV and the pickle file.

## Making Predictions

The saved pipeline takes the seven features in this order: `Area`, `Perimeter`, `Compactness`, `Length of kernel`, `Width of kernel`, `Asymmetry coefficient`, `Length of kernel groove`.

```python
import pickle
import numpy as np

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

sample = [[15.0, 14.5, 0.85, 5.6, 3.4, 2.2, 5.0]]
print("Predicted class:", model.predict(sample))   # -> [1]
```

Because the pipeline starts with an imputer, missing values can be passed as `np.nan`:

```python
sample_missing = [[18.27, np.nan, 0.8870, 6.173, np.nan, 2.443, 6.197]]
print(model.predict(sample_missing))               # -> [2]
```

`wheat3_pred.ipynb` also predicts on a few rows taken from the dataset, and the outputs match their true labels. A scikit-learn warning about missing feature names appears when you pass plain lists. It is harmless. To silence it, pass a `pandas.DataFrame` with the original column names.

## Limitations and Notes

- **Small test set.** 41 test samples means each misclassification moves accuracy by about 2.4 points, so treat the 0.95 as approximate.
- **Non-deterministic results.** `RandomForestClassifier` is created without a `random_state`, so re-running gives slightly different numbers.
- **Outlier removal before the split.** The IQR filtering uses the whole dataset before the train/test split. This is a minor leak, and the effect is small (5 rows removed).
- **Pickle security.** Only load `.pkl` files from sources you trust, since unpickling can run arbitrary code.
- **Class labels.** The target is encoded as 1, 2, 3. In the UCI documentation these correspond to Kama, Rosa, and Canadian.

## Possible Improvements

- Add `random_state` to the classifier and pipeline for reproducibility.
- Use stratified k-fold cross-validation for the final evaluation instead of a single small split.
- Compare other models (SVM, logistic regression, gradient boosting) and report the tuned hyperparameters.
- Add a small CLI or web app for inference.

## Acknowledgements

Dataset: M. Charytanowicz, J. Niewczas, P. Kulczycki, P. A. Kowalski, S. Łukasik, S. Żak, *Seeds* dataset, UCI Machine Learning Repository.
