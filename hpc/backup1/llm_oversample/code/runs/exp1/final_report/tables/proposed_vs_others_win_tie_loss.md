# ImbLLM-IAFP (proposed) versus every other method

W/T/L = datasets where IAFP is better/equal/worse (per-dataset mean over seeds). Wilcoxon signed-rank p-value over datasets (p < 0.05 = consistent difference).

| metric | versus | n_datasets | W/T/L | mean_delta | wilcoxon_p |
|---|---|---|---|---|---|
| f1 | Imbalanced | 22 | 17/1/4 | +0.1107 | 0.001 |
| f1 | RandomOverSampler | 22 | 3/3/16 | -0.0187 | 0.053 |
| f1 | SMOTE* | 21 | 4/3/14 | -0.0165 | 0.085 |
| f1 | ImbLLM-random | 22 | 9/3/10 | -0.0013 | 0.629 |
| f1 | ADASYN | 12 | 1/2/9 | -0.0274 | 0.084 |
| f1 | Original | 22 | 2/3/17 | -0.1077 | 0.001 |
| g_score | Imbalanced | 22 | 17/1/4 | +0.1355 | 0.001 |
| g_score | RandomOverSampler | 22 | 4/3/15 | +0.0025 | 0.136 |
| g_score | SMOTE* | 21 | 4/3/14 | +0.0057 | 0.184 |
| g_score | ImbLLM-random | 22 | 9/3/10 | -0.0010 | 0.658 |
| g_score | ADASYN | 12 | 1/2/9 | -0.0024 | 0.084 |
| g_score | Original | 22 | 2/3/17 | -0.0787 | 0.009 |
| pr_auc | Imbalanced | 22 | 7/4/11 | -0.0052 | 0.349 |
| pr_auc | RandomOverSampler | 22 | 7/4/11 | +0.0013 | 0.777 |
| pr_auc | SMOTE* | 21 | 6/4/11 | -0.0021 | 0.381 |
| pr_auc | ImbLLM-random | 22 | 7/4/11 | -0.0048 | 0.078 |
| pr_auc | ADASYN | 12 | 4/3/5 | +0.0042 | 0.652 |
| pr_auc | Original | 22 | 0/4/18 | -0.0458 | 0.000 |
| roc_auc | Imbalanced | 22 | 5/4/13 | +0.0017 | 0.446 |
| roc_auc | RandomOverSampler | 22 | 9/4/9 | +0.0061 | 0.744 |
| roc_auc | SMOTE* | 21 | 5/4/12 | +0.0009 | 0.356 |
| roc_auc | ImbLLM-random | 22 | 6/4/12 | -0.0036 | 0.184 |
| roc_auc | ADASYN | 12 | 4/3/5 | +0.0020 | 1.000 |
| roc_auc | Original | 22 | 0/4/18 | -0.0255 | 0.000 |
