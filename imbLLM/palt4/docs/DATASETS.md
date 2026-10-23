# Datasets

`tilt data list` shows all 41 registry entries and their groups.

## The thesis benchmark: `natural` (naturally imbalanced OpenML tables)

- Run with q = 0.8. The generator sees 80% of the real minority; the other 20% is the non-member / fidelity reference.
- The target is OpenML's **default target** (renamed `target` on fetch). The minority is the rarest *training* class, so no column names or labels are guessed.
- IR (imbalance ratio) and N are from the OpenML API [V, 28 Sep 2026].

| key | OpenML id | N | IR | features | notes |
|---|---|---|---|---|---|
| `kc1_nat` | 1067 | 2,109 | 5.5 | integer code metrics | NASA defects |
| `churn_nat` | 40701 | 5,000 | 6.1 | mixed | `phone_number` (an id) dropped |
| `pc4` | 1049 | 1,458 | 7.2 | integer | NASA defects |
| `bank_full` | 1461 | 45,211 | 7.5 | mixed, V1–V16 | **V12 (call duration) dropped: known only after the call, so it leaks the label** |
| `pc1` | 1068 | 1,109 | 13.4 | integer | NASA defects |
| `sick_nat` | 38 | 3,772 | 15.3 | mixed | TBG, TBG_measured dropped (all missing) |
| `wilt` | 40983 | 4,839 | 17.5 | numeric | – |
| `car_vgood` | 40975 | 1,728 | 25.6 | 6 **ordinal** | minority = vgood; ordinal orders declared |
| `mammography` | 310 | 11,183 | 42 | anonymised attr1–6 | – |
| `satellite` | 40900 | 5,100 | 67 | numeric | – |

- `natural_fast` = pc1, kc1_nat, wilt: small, and good for laptop development.
- Also registered but outside the group: `ozone` (1487, IR 14.8). Its 72 features make rows exceed 256 tokens, so use it with SIR only.

**Considered and left out:**

| dataset | why left out |
|---|---|
| creditcard (1597) | 284k rows, PCA features hostile to an LLM prior; subsample if added |
| Amazon_employee_access (4135) | very high-cardinality categoricals |
| coil2000 (298) | 85 columns: long rows |
| compas-two-years | IR ≈ 1.1, and already one-hot encoded |

**Other sources for more tables:**
- CLIMB (NeurIPS 2025 D&B, 2505.17451): 73 naturally imbalanced OpenML tables, loaded via `imbalanced-ensemble`.
- KEEL: 145 tables.
- imbalanced-learn `fetch_datasets`: 27 tables. Avoid it for LLMs: the data are already one-hot encoded and anonymised.

## Comparability sets (from palt3)

- `paper`: the 10 ImbLLM tables, run under the ImbLLM protocol (q = 0.2) with `--config replication`.
- `predllm`: the 20 Pred-LLM tables.

Details, verified shapes and known issues are in palt3's `docs/DATASETS.md`:
- **cardio** leaks its label; use `cardio_noleak`.
- adult has 735 train/test duplicate rows.

## Column kinds

- **Discrete:** integer-valued columns are detected from the training rows, and v2 / TILT snap generated values onto their observed support.
- **Ordinal:** declared orders (car, car_vgood, gem_price) are interpolated on rank.

Both are logged in `splits.json`. Details: palt3's `docs/what_corrected.md` §H, vendored here.

## Fetching

- `tilt data fetch --datasets natural` downloads through `sklearn.fetch_openml` into `data/raw/`, and only when you run it.
- `tilt data fetch --datasets all --from ../palt3/data/raw` imports the 21 tables palt3 already has.
- `tilt data verify --datasets natural` checks the shapes. A mismatch in N after NaN-dropping is a logged finding, not an error.
