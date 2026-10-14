# Overall summary

| method | role | n_datasets | f1 | g_score | pr_auc | roc_auc | recall | precision | accuracy | rank_f1 | rank_g_score | rank_pr_auc | rank_roc_auc | f1_gap_closed_% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Original | upper bound | 22 | 0.720 | 0.779 | 0.776 | 0.898 | 0.688 | 0.770 | 0.902 | – | – | – | – | – |
| Imbalanced | floor | 22 | 0.502 | 0.565 | **0.736** | 0.871 | 0.415 | **0.822** | 0.877 | 4.17 | 4.26 | 2.29 | 2.43 | 0% |
| RandomOverSampler | benchmark | 22 | **0.631** | **0.698** | 0.729 | 0.867 | **0.575** | 0.754 | 0.881 | 2.00 | 2.00 | 3.00 | 3.29 | 58% |
| SMOTE* | benchmark | 21 | 0.619 | 0.683 | 0.723 | 0.866 | 0.550 | 0.771 | **0.885** | 2.05 | 2.05 | 2.62 | 2.52 | 57% |
| ADASYN | benchmark | 12 | 0.625 | 0.681 | 0.726 | 0.860 | 0.570 | 0.752 | 0.882 | – | – | – | – | 62% |
| ImbLLM-random | original paper | 22 | 0.561 | 0.655 | 0.727 | **0.872** | 0.505 | 0.754 | 0.866 | 3.31 | 3.36 | 3.33 | 3.05 | 24% |
| ImbLLM-IAFP | PROPOSED | 22 | 0.586 | 0.677 | 0.723 | 0.866 | 0.545 | 0.750 | 0.865 | 3.48 | 3.33 | 3.76 | 3.71 | 42% |

Averages are over the datasets each method ran on (n_datasets). Ranks (1 = best) use only the 21 datasets where all of Imbalanced, RandomOverSampler, SMOTE*, ImbLLM-random, ImbLLM-IAFP exist; Original/ADASYN are not ranked. f1_gap_closed = share of the Original-minus-Imbalanced F1 gap recovered, over datasets where that gap > 0.01.
