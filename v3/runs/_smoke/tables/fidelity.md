# Quality/diversity (Fig. 5)

Fidelity (discriminator, discriminator_lr, inverse_kl, density, coverage) is measured against the HELD-OUT test rows; privacy (dcr_mean, exact_copy_rate) against the training rows. Discriminators: lower is better.

| col                              |   discriminator |   discriminator_lr |   inverse_kl |   density |   coverage |   dcr_mean |   exact_copy_rate |
|:---------------------------------|----------------:|-------------------:|-------------:|----------:|-----------:|-----------:|------------------:|
| v3:original@tiny-gpt2            |          0      |             0      |       0.9174 |    0.7333 |     0.9    |     0      |                 1 |
| v3:pred_llm_v3@tiny-gpt2         |          1      |             0.8765 |       0.6154 |    0.0042 |     0.0333 |     0.4303 |                 0 |
| v3:v3_no_negative@tiny-gpt2      |          1      |             0.8765 |       0.6154 |    0.0042 |     0.0333 |     0.4303 |                 0 |
| v3:v3_no_outlier_range@tiny-gpt2 |          0.9753 |             0.7037 |       0.6468 |    0.0042 |     0.0333 |     0.4538 |                 0 |