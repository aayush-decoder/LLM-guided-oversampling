# Quality/diversity (Fig. 5)

Fidelity (discriminator, discriminator_lr, inverse_kl, density, coverage) is measured against the HELD-OUT test rows; privacy (dcr_mean, exact_copy_rate) against the training rows. Discriminators: lower is better.

| col                    |   discriminator |   discriminator_lr |   inverse_kl |   density |   coverage |   dcr_mean |   exact_copy_rate |
|:-----------------------|----------------:|-------------------:|-------------:|----------:|-----------:|-----------:|------------------:|
| v1:ctgan@distilgpt2    |          1      |             1      |       0.7149 |    0      |     0      |     0.368  |                 0 |
| v1:great@distilgpt2    |          0.6543 |             0.1111 |       0.6838 |    0.0875 |     0.5333 |     0.1464 |                 0 |
| v1:original@distilgpt2 |          0      |             0      |       0.9174 |    0.7333 |     0.9    |     0      |                 1 |
| v1:pred_llm@distilgpt2 |          0.4321 |             0.284  |       0.8178 |    0.2208 |     0.5333 |     0.1208 |                 0 |
| v1:taptap@distilgpt2   |          0.679  |             0.2099 |       0.6473 |    0.1417 |     0.6    |     0.1318 |                 0 |