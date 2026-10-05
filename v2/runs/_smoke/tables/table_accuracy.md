# TSTR accuracy (XGBoost)  mean (std) over seeds

| dataset   | Original(ours)   | v2:original@tiny-gpt2   | v2:pred_llm_v2@tiny-gpt2   | v2:v2_encoded@tiny-gpt2   | v2:v2_plain_loss@tiny-gpt2   |   paper:original |   paper:copulagan |   paper:ctgan |   paper:tvae |   paper:great |   paper:taptap |   paper:pred_llm |
|:----------|:-----------------|:------------------------|:---------------------------|:--------------------------|:-----------------------------|-----------------:|------------------:|--------------:|-------------:|--------------:|---------------:|-----------------:|
| iris      | 0.9333 (0.000)   | 0.9333 (0.000)          | 0.4000 (0.000)             | 0.5333 (0.000)            | 0.4000 (0.000)               |           0.9555 |            0.3445 |        0.4444 |          0.9 |        0.3667 |         0.8778 |           0.9889 |

## Average over datasets

| col                        |   mean |
|:---------------------------|-------:|
| v2:original@tiny-gpt2      | 0.9333 |
| v2:pred_llm_v2@tiny-gpt2   | 0.4    |
| v2:v2_encoded@tiny-gpt2    | 0.5333 |
| v2:v2_plain_loss@tiny-gpt2 | 0.4    |