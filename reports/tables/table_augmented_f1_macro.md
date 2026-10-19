# Data Augmentation: Real Only vs. (Real + Synthetic) on Held-Out Test Set
Metric: f1_macro (XGBoost) mean (std) over seeds

| dataset   | Real Only (Original)   | v2:pred_llm_v2@distilgpt2   | v3:pred_llm_v3@distilgpt2   | v3:v3_cfg_negative@distilgpt2   | v3:v3_no_curation@distilgpt2   | v3:v3_no_negative@distilgpt2   | v3:v3_no_outlier_range@distilgpt2   | v3:v3_plain@distilgpt2   |
|:----------|:-----------------------|:----------------------------|:----------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:------------------------------------|:-------------------------|
| bank      | 0.6704 (0.000)         | 0.6949 (0.000)              | nan                         | nan                             | nan                            | nan                            | nan                                 | nan                      |
| blood     | 0.5630 (0.000)         | nan                         | 0.5842 (0.000)              | nan                             | nan                            | nan                            | nan                                 | nan                      |
| iris      | 0.9553 (0.032)         | nan                         | 0.9553 (0.016)              | 0.9553 (0.016)                  | 0.9435 (0.042)                 | 0.9553 (0.016)                 | 0.9666 (0.000)                      | 0.9666 (0.000)           |
