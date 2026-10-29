# TSTR f1_macro (XGBoost)  mean (std) over seeds

| dataset   | Original(ours)   | v2:original@tiny-gpt2   | v2:pred_llm_v2@tiny-gpt2   | v2:v2_encoded@tiny-gpt2   | v2:v2_plain_loss@tiny-gpt2   |
|:----------|:-----------------|:------------------------|:---------------------------|:--------------------------|:-----------------------------|
| iris      | 0.9327 (0.000)   | 0.9327 (0.000)          | 0.3576 (0.000)             | 0.4908 (0.000)            | 0.3576 (0.000)               |

## Average over datasets

| col                        |   mean |
|:---------------------------|-------:|
| v2:original@tiny-gpt2      | 0.9327 |
| v2:pred_llm_v2@tiny-gpt2   | 0.3576 |
| v2:v2_encoded@tiny-gpt2    | 0.4908 |
| v2:v2_plain_loss@tiny-gpt2 | 0.3576 |