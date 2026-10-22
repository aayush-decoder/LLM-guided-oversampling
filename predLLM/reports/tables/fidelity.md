# Quality/diversity (Fig. 5)

Fidelity (discriminator, discriminator_lr, inverse_kl, density, coverage) is measured against the HELD-OUT test rows; privacy (dcr_mean, exact_copy_rate) against the training rows. Discriminators: lower is better.

| col                               |   discriminator |   discriminator_lr |   inverse_kl |   density |   coverage |   dcr_mean |   exact_copy_rate |
|:----------------------------------|----------------:|-------------------:|-------------:|----------:|-----------:|-----------:|------------------:|
| v1:great@distilgpt2               |          0.3087 |             0.0947 |       0.7954 |    0.2694 |     0.6667 |     0.1452 |            0      |
| v1:pred_llm@distilgpt2            |          0.1399 |             0.1193 |       0.8197 |    0.7292 |     0.8556 |     0.1005 |            0.0416 |
| v1:pred_llm@qwen2.5-0.5b          |          0.0412 |             0.1481 |       0.7808 |    1.0167 |     0.9333 |     0.0538 |            0.4944 |
| v1:pred_llm@qwen3-0.6b            |          0.0741 |             0      |       0.837  |    1.0069 |     0.9333 |     0.0552 |            0.4361 |
| v1:taptap@distilgpt2              |          0.4156 |             0.107  |       0.752  |    0.6403 |     0.7889 |     0.1198 |            0.0055 |
| v2:pred_llm_v2@distilgpt2         |          0.2593 |             0.2017 |       0.7938 |    0.5903 |     0.8444 |     0.1024 |            0      |
| v2:pred_llm_v2@qwen2.5-0.5b       |          0.1029 |             0.0206 |       0.7617 |    0.818  |     0.8555 |     0.0813 |            0.2667 |
| v2:pred_llm_v2@qwen2.5-1.5b       |          0.1029 |             0.0782 |       0.8109 |    1.0278 |     0.9222 |     0.064  |            0.4694 |
| v2:pred_llm_v2@qwen3-0.6b         |          0.2222 |             0.0206 |       0.8084 |    0.9556 |     0.8667 |     0.0649 |            0.4583 |
| v2:pred_llm_v2@qwen3-1.7b         |          0.0206 |             0.107  |       0.8098 |    1.0417 |     0.9222 |     0.042  |            0.6806 |
| v2:v2_encoded@distilgpt2          |          0.1193 |             0.0453 |       0.8003 |    0.7153 |     0.8889 |     0.0964 |            0      |
| v2:v2_plain_loss@distilgpt2       |          0.2181 |             0.037  |       0.8049 |    0.593  |     0.8889 |     0.1031 |            0      |
| v3:pred_llm_v3@distilgpt2         |          0.2583 |             0.237  |       0.7727 |    0.5205 |     0.785  |     0.0978 |            0.0293 |
| v3:v3_cfg_negative@distilgpt2     |          0.1193 |             0.0947 |       0.7814 |    0.4889 |     0.8333 |     0.1099 |            0      |
| v3:v3_no_curation@distilgpt2      |          0.1482 |             0.1852 |       0.7839 |    0.6167 |     0.8555 |     0.0989 |            0.0083 |
| v3:v3_no_negative@distilgpt2      |          0.1193 |             0.0947 |       0.7814 |    0.4889 |     0.8333 |     0.1099 |            0      |
| v3:v3_no_outlier_range@distilgpt2 |          0.1029 |             0      |       0.789  |    0.45   |     0.8444 |     0.1094 |            0      |
| v3:v3_plain@distilgpt2            |          0.2305 |             0.1729 |       0.7869 |    0.5917 |     0.8778 |     0.1042 |            0      |