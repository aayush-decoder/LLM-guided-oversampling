# TSTR accuracy (XGBoost)  mean (std) over seeds

| dataset   | Original(ours)   | v3:original@tiny-gpt2   | v3:pred_llm_v3@tiny-gpt2   | v3:v3_no_negative@tiny-gpt2   | v3:v3_no_outlier_range@tiny-gpt2   |   paper:original |   paper:copulagan |   paper:ctgan |   paper:tvae |   paper:great |   paper:taptap |   paper:pred_llm |
|:----------|:-----------------|:------------------------|:---------------------------|:------------------------------|:-----------------------------------|-----------------:|------------------:|--------------:|-------------:|--------------:|---------------:|-----------------:|
| iris      | 0.9333 (0.000)   | 0.9333 (0.000)          | 0.6333 (0.000)             | 0.6333 (0.000)                | 0.5000 (0.000)                     |           0.9555 |            0.3445 |        0.4444 |          0.9 |        0.3667 |         0.8778 |           0.9889 |

## Average over datasets

| col                              |   mean |
|:---------------------------------|-------:|
| v3:original@tiny-gpt2            | 0.9333 |
| v3:pred_llm_v3@tiny-gpt2         | 0.6333 |
| v3:v3_no_negative@tiny-gpt2      | 0.6333 |
| v3:v3_no_outlier_range@tiny-gpt2 | 0.5    |