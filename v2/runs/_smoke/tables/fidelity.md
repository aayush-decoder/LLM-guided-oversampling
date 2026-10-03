# Quality/diversity (Fig. 5)

Fidelity (discriminator, discriminator_lr, inverse_kl, density, coverage) is measured against the HELD-OUT test rows; privacy (dcr_mean, exact_copy_rate) against the training rows. Discriminators: lower is better.

| col                        |   discriminator |   discriminator_lr |   inverse_kl |   density |   coverage |   dcr_mean |   exact_copy_rate |
|:---------------------------|----------------:|-------------------:|-------------:|----------:|-----------:|-----------:|------------------:|
| v2:original@tiny-gpt2      |               0 |             0      |       0.9174 |    0.7333 |        0.9 |     0      |                 1 |
| v2:pred_llm_v2@tiny-gpt2   |               1 |             0.8519 |       0.6626 |    0      |        0   |     0.4297 |                 0 |
| v2:v2_encoded@tiny-gpt2    |               1 |             0.9506 |       0.5339 |    0      |        0   |     0.5986 |                 0 |
| v2:v2_plain_loss@tiny-gpt2 |               1 |             0.8519 |       0.6626 |    0      |        0   |     0.4297 |                 0 |