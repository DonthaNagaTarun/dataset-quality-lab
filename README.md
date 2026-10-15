# AI-Powered Dataset Quality Assessment and Its Impact on Machine Learning Model Performance

A local Streamlit application for inspecting CSV data quality, generating controlled
synthetic classification data, exporting a cleaned dataset, and comparing Logistic
Regression and Random Forest performance before and after robust preprocessing.

## Features

- Upload a CSV file or generate synthetic data with adjustable missingness,
  duplicates, numeric outliers, and class imbalance.
- Detect missing values, exact duplicate rows, numeric IQR outliers, and class
  imbalance.
- Calculate a transparent composite quality score from 0 to 100.
- Clean and download a CSV with duplicates removed, missing features imputed, and
  numeric outliers capped.
- Compare Accuracy and macro F1 for Logistic Regression and Random Forest on a
  shared stratified holdout split.
- Download a Markdown research report with the methodology, results, and limitations.
- Process uploaded data locally in the app; the application does not send it to an
  external service.

## Windows installation

Install Python 3.11 or newer from [python.org](https://www.python.org/downloads/).
During installation, enable **Add Python to PATH**.

Open PowerShell in this project folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in a
browser. To stop the app, press `Ctrl+C` in the terminal.

## Using the application

1. Choose **Upload CSV** or **Generate synthetic** in the sidebar.
2. Select the column that represents the classification target.
3. Review the quality overview and column-level checks.
4. Download the cleaned CSV from **Data cleaning**.
5. In **Model impact**, choose a holdout size and run the comparison.
6. Download the report from **Research report**.

The uploaded CSV must have unique column names and at least one feature column plus
one target column. The model comparison supports classification targets with 2–30
classes, where every class has at least two records after duplicate removal.

## Quality score

The heuristic starts at 100 and subtracts a weighted penalty for:

- Missing cells: up to 25 points, scaled by the fraction of missing cells.
- Duplicate rows: up to 15 points, scaled by the fraction of duplicate rows.
- Rows with one or more numeric IQR outliers: up to 25 points, scaled by their
  fraction of all rows.
- Target class imbalance: up to 35 points, scaled by `1 - minority_count / majority_count`.

The final value is limited to the 0–100 range. This is an interpretable project
heuristic, not a universal or standardized data-quality score.

## Evaluation and data-leakage safeguards

The comparison removes exact duplicate rows and rows with a missing target before
making one stratified train/test split. Both conditions use the same holdout set.
Imputers, one-hot encoders, scalers, and outlier-clipping thresholds are fitted
inside scikit-learn pipelines on training data only. The test set is used only for
final predictions and metrics.

The "before cleaning" condition still imputes missing data so the estimators can
train. The "after cleaning" condition additionally clips numeric features using
training-only IQR bounds and applies balanced class weights. The downloadable
cleaned CSV is a separate full-dataset export and is **not** used to fit the test
evaluation.

Accuracy and macro F1 are reported. Macro F1 gives every class equal weight, which
is useful when classes are imbalanced. The app uses one fixed random seed and one
holdout split; for research conclusions, repeat the experiment across seeds or use
cross-validation and review whether the chosen cleaning strategy makes sense for
the domain.
