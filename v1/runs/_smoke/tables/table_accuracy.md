# TSTR accuracy (XGBoost)  mean (std) over seeds

| dataset   | Original(ours)   | v1:ctgan@distilgpt2   | v1:great@distilgpt2   | v1:original@distilgpt2   | v1:pred_llm@distilgpt2   | v1:taptap@distilgpt2   |   paper:original |   paper:copulagan |   paper:ctgan |   paper:tvae |   paper:great |   paper:taptap |   paper:pred_llm |
|:----------|:-----------------|:----------------------|:----------------------|:-------------------------|:-------------------------|:-----------------------|-----------------:|------------------:|--------------:|-------------:|--------------:|---------------:|-----------------:|
| iris      | 0.9333 (0.000)   | 0.4333 (0.000)        | 0.5333 (0.000)        | 0.9333 (0.000)           | 1.0000 (0.000)           | 0.9333 (0.000)         |           0.9555 |            0.3445 |        0.4444 |          0.9 |        0.3667 |         0.8778 |           0.9889 |

## Average over datasets

| col                    |   mean |
|:-----------------------|-------:|
| v1:ctgan@distilgpt2    | 0.4333 |
| v1:great@distilgpt2    | 0.5333 |
| v1:original@distilgpt2 | 0.9333 |
| v1:pred_llm@distilgpt2 | 1      |
| v1:taptap@distilgpt2   | 0.9333 |