# TSTR accuracy (XGBoost)  mean (std) over seeds

| dataset   | Original(ours)   | v1:great@distilgpt2   | v1:pred_llm@distilgpt2   | v1:pred_llm@qwen2.5-0.5b   | v1:pred_llm@qwen3-0.6b   | v1:taptap@distilgpt2   | v2:pred_llm_v2@distilgpt2   | v2:pred_llm_v2@qwen2.5-0.5b   | v2:pred_llm_v2@qwen2.5-1.5b   | v2:pred_llm_v2@qwen3-0.6b   | v2:pred_llm_v2@qwen3-1.7b   | v2:v2_encoded@distilgpt2   | v2:v2_plain_loss@distilgpt2   | v3:pred_llm_v3@distilgpt2   | v3:v3_cfg_negative@distilgpt2   | v3:v3_no_curation@distilgpt2   | v3:v3_no_negative@distilgpt2   | v3:v3_no_outlier_range@distilgpt2   | v3:v3_plain@distilgpt2   |   paper:original |   paper:copulagan |   paper:ctgan |   paper:tvae |   paper:great |   paper:taptap |   paper:pred_llm |
|:----------|:-----------------|:----------------------|:-------------------------|:---------------------------|:-------------------------|:-----------------------|:----------------------------|:------------------------------|:------------------------------|:----------------------------|:----------------------------|:---------------------------|:------------------------------|:----------------------------|:--------------------------------|:-------------------------------|:-------------------------------|:------------------------------------|:-------------------------|-----------------:|------------------:|--------------:|-------------:|--------------:|---------------:|-----------------:|
| blood     | 0.7067 (0.000)   | nan                   | nan                      | nan                        | nan                      | nan                    | nan                         | nan                           | nan                           | nan                         | nan                         | nan                        | nan                           | 0.3000 (0.000)              | nan                             | nan                            | nan                            | nan                                 | nan                      |           0.7444 |            0.7178 |        0.7355 |       0.7445 |        0.6866 |         0.7422 |           0.7489 |
| iris      | 0.9555 (0.031)   | 0.4222 (0.042)        | 0.9778 (0.016)           | 0.9333 (0.027)             | 0.9556 (0.042)           | 0.9111 (0.083)         | 0.9333 (0.054)              | 0.9444 (0.042)                | 0.9556 (0.042)                | 0.9556 (0.042)              | 0.9556 (0.042)              | 0.9333 (0.047)             | 0.9555 (0.031)                | 0.9667 (0.000)              | 0.9667 (0.000)                  | 0.9333 (0.054)                 | 0.9667 (0.000)                 | 0.9667 (0.027)                      | 0.9667 (0.000)           |           0.9555 |            0.3445 |        0.4444 |       0.9    |        0.3667 |         0.8778 |           0.9889 |

## Average over datasets

| col                               |     mean |
|:----------------------------------|---------:|
| v1:great@distilgpt2               | 0.422233 |
| v1:pred_llm@distilgpt2            | 0.9778   |
| v1:pred_llm@qwen2.5-0.5b          | 0.933333 |
| v1:pred_llm@qwen3-0.6b            | 0.955567 |
| v1:taptap@distilgpt2              | 0.9111   |
| v2:pred_llm_v2@distilgpt2         | 0.933333 |
| v2:pred_llm_v2@qwen2.5-0.5b       | 0.944433 |
| v2:pred_llm_v2@qwen2.5-1.5b       | 0.955567 |
| v2:pred_llm_v2@qwen3-0.6b         | 0.955567 |
| v2:pred_llm_v2@qwen3-1.7b         | 0.955567 |
| v2:v2_encoded@distilgpt2          | 0.933333 |
| v2:v2_plain_loss@distilgpt2       | 0.955533 |
| v3:pred_llm_v3@distilgpt2         | 0.800025 |
| v3:v3_cfg_negative@distilgpt2     | 0.9667   |
| v3:v3_no_curation@distilgpt2      | 0.933333 |
| v3:v3_no_negative@distilgpt2      | 0.9667   |
| v3:v3_no_outlier_range@distilgpt2 | 0.966667 |
| v3:v3_plain@distilgpt2            | 0.9667   |