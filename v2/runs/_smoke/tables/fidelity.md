# Quality/diversity (Fig. 5)

Fidelity (discriminator, discriminator_lr, inverse_kl, density, coverage) is measured against the HELD-OUT test rows; privacy (dcr_mean, exact_copy_rate) against the training rows. Discriminators: lower is better.

| col                        |   discriminator |   discriminator_lr |   inverse_kl |   density |   coverage |   dcr_mean |   exact_copy_rate |
|:---------------------------|----------------:|-------------------:|-------------:|----------:|-----------:|-----------:|------------------:|
| v2:original@tiny-gpt2      |          0      |             0      |       0.9174 |    0.7333 |     0.9    |     0      |                 1 |
| v2:pred_llm_v2@tiny-gpt2   |          0.9753 |             0.8272 |       0.6668 |    0.0125 |     0.0667 |     0.3371 |                 0 |
| v2:v2_encoded@tiny-gpt2    |          1      |             1      |       0.5877 |    0      |     0      |     0.7029 |                 0 |
| v2:v2_plain_loss@tiny-gpt2 |          0.9753 |             0.8272 |       0.6668 |    0.0125 |     0.0667 |     0.3371 |                 0 |