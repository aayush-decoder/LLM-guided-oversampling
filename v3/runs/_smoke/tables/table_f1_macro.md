# TSTR f1_macro (XGBoost)  mean (std) over seeds

| dataset   | Original(ours)   | v3:original@tiny-gpt2   | v3:pred_llm_v3@tiny-gpt2   | v3:v3_no_negative@tiny-gpt2   | v3:v3_no_outlier_range@tiny-gpt2   |
|:----------|:-----------------|:------------------------|:---------------------------|:------------------------------|:-----------------------------------|
| iris      | 0.9327 (0.000)   | 0.9327 (0.000)          | 0.5308 (0.000)             | 0.5308 (0.000)                | 0.4046 (0.000)                     |

## Average over datasets

| col                              |   mean |
|:---------------------------------|-------:|
| v3:original@tiny-gpt2            | 0.9327 |
| v3:pred_llm_v3@tiny-gpt2         | 0.5308 |
| v3:v3_no_negative@tiny-gpt2      | 0.5308 |
| v3:v3_no_outlier_range@tiny-gpt2 | 0.4046 |