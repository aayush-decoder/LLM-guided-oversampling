# ImbLLM-IAFP (proposed) versus every other method

W/T/L = datasets where IAFP is better/equal/worse (per-dataset mean over seeds). Wilcoxon signed-rank p-value over datasets (p < 0.05 = consistent difference).

| metric | versus | n_datasets | W/T/L | mean_delta | wilcoxon_p |
|---|---|---|---|---|---|
| f1 | Imbalanced | 22 | 15/1/6 | +0.0843 | 0.007 |
| f1 | RandomOverSampler | 22 | 3/3/16 | -0.0451 | 0.016 |
| f1 | SMOTE* | 21 | 4/3/14 | -0.0445 | 0.028 |
| f1 | ImbLLM-random | 22 | 8/1/13 | +0.0246 | 0.876 |
| f1 | ADASYN | 12 | 1/2/9 | -0.0590 | 0.064 |
| f1 | Original | 22 | 2/3/17 | -0.1340 | 0.001 |
| g_score | Imbalanced | 22 | 16/1/5 | +0.1121 | 0.003 |
| g_score | RandomOverSampler | 22 | 4/3/15 | -0.0209 | 0.053 |
| g_score | SMOTE* | 21 | 4/3/14 | -0.0182 | 0.094 |
| g_score | ImbLLM-random | 22 | 10/1/11 | +0.0218 | 0.498 |
| g_score | ADASYN | 12 | 1/2/9 | -0.0315 | 0.084 |
| g_score | Original | 22 | 2/3/17 | -0.1021 | 0.009 |
| pr_auc | Imbalanced | 22 | 4/4/14 | -0.0130 | 0.071 |
| pr_auc | RandomOverSampler | 22 | 4/4/14 | -0.0064 | 0.078 |
| pr_auc | SMOTE* | 21 | 6/4/11 | -0.0094 | 0.210 |
| pr_auc | ImbLLM-random | 22 | 7/2/13 | -0.0038 | 0.135 |
| pr_auc | ADASYN | 12 | 3/3/6 | -0.0049 | 0.570 |
| pr_auc | Original | 22 | 1/4/17 | -0.0535 | 0.000 |
| roc_auc | Imbalanced | 22 | 6/4/12 | -0.0053 | 0.094 |
| roc_auc | RandomOverSampler | 22 | 6/4/12 | -0.0009 | 0.396 |
| roc_auc | SMOTE* | 21 | 4/4/13 | -0.0064 | 0.044 |
| roc_auc | ImbLLM-random | 22 | 6/2/14 | -0.0061 | 0.014 |
| roc_auc | ADASYN | 12 | 2/3/7 | -0.0082 | 0.074 |
| roc_auc | Original | 22 | 0/4/18 | -0.0325 | 0.000 |
