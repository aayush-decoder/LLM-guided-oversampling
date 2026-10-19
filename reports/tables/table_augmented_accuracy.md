# Data Augmentation: Real Only vs. (Real + Synthetic) on Held-Out Test Set
Metric: accuracy (XGBoost) mean (std) over seeds

| dataset   | Real Only (Original)   | v2:pred_llm_v2@distilgpt2   | v3:pred_llm_v3@distilgpt2   | v3:v3_cfg_negative@distilgpt2   | v3:v3_no_curation@distilgpt2   | v3:v3_no_negative@distilgpt2   | v3:v3_no_outlier_range@distilgpt2   | v3:v3_plain@distilgpt2   |
|:----------|:-----------------------|:----------------------------|:----------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:------------------------------------|:-------------------------|
| bank      | 0.8884 (0.000)         | 0.8961 (0.000)              | nan                         | nan                             | nan                            | nan                            | nan                                 | nan                      |
| blood     | 0.7067 (0.000)         | nan                         | 0.6800 (0.000)              | nan                             | nan                            | nan                            | nan                                 | nan                      |
| iris      | 0.9555 (0.031)         | nan                         | 0.9556 (0.016)              | 0.9556 (0.016)                  | 0.9444 (0.042)                 | 0.9556 (0.016)                 | 0.9667 (0.000)                      | 0.9667 (0.000)           |
