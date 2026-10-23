# ImbLLM-style Credit Card Fraud Oversampling

This project implements a research-paper-inspired tabular oversampling workflow for the Credit Card Fraud Detection dataset. It uses numerical interpolation and a fine-tuned DistilGPT2 language model to generate synthetic minority/fraud transactions.

The main educational deliverable is:

```text
ImbLLM_CreditCard_DistilGPT2.ipynb
```

The notebook compares the proposed synthetic oversampling approach with:

- Original imbalanced training
- Random oversampling
- SMOTE
- DistilGPT2-inspired synthetic oversampling

All methods are evaluated on the same untouched test set.

## Project structure

```text
ML_paper_implent/
|
|-- ImbLLM_CreditCard_DistilGPT2.ipynb
|-- compare_original_vs_synthetic.py
|-- run_experiment.py
|-- requirements.txt
|-- ML_paper.pdf
|
|-- imbllm_pipeline/
|   |-- __init__.py
|   |-- config.py
|   |-- data.py
|   |-- text.py
|   `-- metrics.py
|
|-- creditcard.csv/
|   `-- creditcard.csv
|
`-- results/
```

The `creditcard.csv` file is the original dataset and is intentionally not included in Git-style source control workflows because it is large. Keep it at:

```text
creditcard.csv/creditcard.csv
```

The code also supports a directly placed file named:

```text
creditcard.csv
```

## Requirements

Recommended Python version:

```text
Python 3.10, 3.11, or 3.12
```

Python 3.11 is recommended for broad compatibility with PyTorch and Transformers.

The project uses:

- Python
- Jupyter Notebook
- pandas and NumPy
- scikit-learn
- imbalanced-learn
- XGBoost
- PyTorch
- Hugging Face Transformers
- SciPy
- matplotlib and seaborn

## Installation

Open PowerShell in the project directory:

```powershell
Set-Location "C:\path\to\ML_paper_implent"
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start Jupyter:

```powershell
jupyter notebook
```

Open `ImbLLM_CreditCard_DistilGPT2.ipynb` and select the virtual-environment Python kernel.

## Quick smoke test

Before running the full notebook, verify that the data preparation package works:

```powershell
python run_experiment.py
```

This performs the leakage-safe split, constructs the minority partitions, creates interpolated rows, and writes:

```text
results/partition_summary.csv
```

The smoke test does not fine-tune DistilGPT2.

## Running the notebook

Run the notebook cells from top to bottom.

For the first run, keep:

```python
DEBUG_MODE = True
```

Debug mode:

- Uses fewer fine-tuning epochs
- Generates approximately 100 synthetic rows
- Helps verify model loading, generation, parsing, and classifier execution
- Is suitable for a CPU smoke test, although CPU generation can still be slow

After the debug run succeeds, set:

```python
DEBUG_MODE = False
```

Full mode trains for the configured number of epochs and generates enough synthetic rows to approximately balance the majority class.

The first model-loading run requires internet access because DistilGPT2 is downloaded from Hugging Face:

```python
MODEL_NAME = "distilgpt2"
```

The model is normally cached locally after downloading.

## GPU support

The notebook automatically selects CUDA when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

A GPU is recommended for the fine-tuning and generation stages, but it is not required. CPU execution is supported and may take substantially longer.

## Research workflow

The notebook follows this leakage-safe workflow:

```text
Original dataset
        |
        v
80/20 stratified train/test split
        |
        +-------------------+
        |                   |
        v                   v
   D_train              D_test
        |
        v
D_major + D*_minor
        |
        v
20% visible D_minor
        |
        +------------------------+
        |                        |
        v                        v
Numerical interpolation      fix_y text conversion
        |                        |
        v                        v
   D_inter              D_minor + D_inter
                                  |
                                  v
                            DistilGPT2
                                  |
                                  v
                           condition_yx
                                  |
                                  v
                         D_hat_minor synthetic
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
          XGBoost classifier              Quality analysis
                  |                       KS, PCA, DCR,
                  v                       Coverage, Close Probability
               D_test
```

Notation:

- `D_major`: majority-class training rows
- `D*_minor`: all original fraud rows in the training set
- `D_minor`: visible 20% subset of `D*_minor`
- `D_inter`: interpolated minority rows created only from `D_minor`
- `D_hat_minor`: validated synthetic minority rows generated by DistilGPT2
- `D_test`: completely held-out test set

## Important leakage safeguards

The test set is created before oversampling and is never used for:

- DistilGPT2 fine-tuning
- Minority interpolation
- Synthetic prompt construction
- Scaler fitting
- PCA fitting
- Synthetic quality reference
- Model selection or tuning

The unseen portion of `D*_minor` is used only for synthetic-data quality analysis. It is not used for LLM training.

## Comparing original and synthetic training

After running the notebook through the generation stage, the notebook writes:

```text
results/synthetic_tabular.csv
```

This is the classifier-ready synthetic dataset. It contains numerical feature columns and `Class=1`.

Run:

```powershell
python compare_original_vs_synthetic.py
```

This trains and evaluates:

1. XGBoost using the original training data
2. XGBoost using original training data plus synthetic minority rows

Both models use the same untouched test set.

The comparison is saved to:

```text
results/original_vs_synthetic.csv
```

Focus primarily on:

- F1
- PR-AUC
- Recall
- Precision
- ROC-AUC
- False positives and false negatives

Accuracy alone is not sufficient because the dataset is extremely imbalanced.

## Generated outputs

The notebook writes outputs to `results/`, including:

```text
classification_results.csv
original_vs_synthetic.csv
ablation_results.csv
synthetic_tabular.csv
raw_llm_generations.csv
synthetic_quality_metrics.csv
partition_summary.csv

class_distribution.png
loss_curve.png
pca_comparison.png
correlation_real.png
correlation_synthetic.png
dcr_distribution.png
roc_curves.png
pr_curves.png
f1_comparison.png
auc_comparison.png

distilgpt2_model/
```

The two synthetic files have different purposes:

- `synthetic_tabular.csv`: parsed numerical records for classifier training
- `raw_llm_generations.csv`: raw prompts, generated text, validation status, and rejection reasons

Do not pass `raw_llm_generations.csv` directly to XGBoost.

## Interpretation of quality metrics

The notebook reports:

- Per-feature KS statistics
- Mean distributional difference
- Correlation heatmaps
- PCA visualization
- Distance to Closest Record (DCR)
- Approximate Close Probability
- Approximate Coverage

Close Probability and Coverage are explicitly labelled approximations when the paper does not provide an executable definition. They should not be interpreted as identical to an official implementation unless the exact paper definition is available.

## Research-paper deviations

This is not claimed to be an exact reproduction of every paper experiment. The notebook clearly documents these deviations:

- The Credit Card Fraud Detection dataset is used instead of the paper's original dataset.
- DistilGPT2 is used as the language model.
- Practical debug/full-mode settings are used.
- The parser accepts the notebook's natural-language row format.
- Close Probability and Coverage may use documented distance-based approximations.
- The PCA-transformed dataset features remain numerical during interpolation.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the virtual environment is activated and dependencies are installed:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dataset not found

Confirm that one of these paths exists:

```text
creditcard.csv/creditcard.csv
creditcard.csv
```

### DistilGPT2 download fails

Check internet access and rerun the model-loading cell. Hugging Face may also require additional time for the first download.

### Notebook is too slow

Set:

```python
DEBUG_MODE = True
```

Use a GPU if available, and avoid generating the full dataset until the debug run succeeds.

### No valid generated rows

Inspect:

```text
results/raw_llm_generations.csv
```

The notebook records parser rejection reasons. In debug mode it uses an interpolation fallback only to keep downstream analysis executable; this fallback is clearly reported and must not be confused with successful LLM generation.

## Reproducibility

The default seed is:

```python
SEED = 42
```

The notebook also evaluates multiple seeds:

```python
SEEDS = [42, 123, 456]
```

The experiment configuration is printed near the beginning of the notebook.

