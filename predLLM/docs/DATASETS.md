# Datasets: download guide

All 27 datasets (the paper's 20 plus the extended 10; adult, bank and fuel are in both) are listed in
`configs/datasets.yaml`. Each is stored once as `data/raw/<name>.csv` plus `<name>.meta.json`
(target name and column types).

```bash
python -m tabgen data download --dataset all      # everything that can be fetched automatically
python -m tabgen data list                        # shows which datasets are available
python -m tabgen audit data --dataset all         # checks sizes against the table below
```

## Automatic (no action needed)

| name | rows × features | classes | source |
|---|---|---|---|
| iris, breast | 150×4, 569×30 | 3, 2 | scikit-learn (bundled) |
| california | 20640×8 | regression | scikit-learn (`fetch_california_housing`) |
| blood, steel, diabetes, australian, balance, qsar, phoneme, waveform, churn, kc1, kc2, cardio | see Table I | | OpenML ids 1464, 1504, 37, 40981, 11, 1494, 1489, 60, 40701, 1067, 1063, 1466 |
| cardio_noleak | 2126 | 25 | OpenML 1466 without V26–V35 (see the warning below) |
| german (credit-g), car, sick | 1000×20, 1728×6, 3772→cleaned | 2, 4, 2 | OpenML ids 31, 40975, 38 |
| adult, bank, compas, abalone, fuel | 30162×11, 4521×14, 4010×8, 4177×8, 639×8 | | CSVs from the authors' repo (github.com/nphdang/Pred-LLM/data), same preprocessing as their `read_data.py` |
| insurance | 1338×6 | regression | public GitHub mirror of Kaggle `mirichoi0218/insurance` |
| housing | 20640→20433×9 | regression | public GitHub mirror (ageron/handson-ml2) of Kaggle `camnugent/california-housing-prices` |

## Kaggle-only: `gem_price`, `credit_card`

These two have no stable public mirror. Use either option.

**Option A: Kaggle API (automatic)**
1. `pip install kaggle`
2. On kaggle.com open *Settings → API → Create New Token*. This downloads `kaggle.json`.
3. Put it at `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/macOS).
4. `python -m tabgen data download --dataset gem_price,credit_card`

**Option B: manual**

| dataset | page | file to download | put it at |
|---|---|---|---|
| gem_price | kaggle.com/datasets/colearninglounge/gemstone-price-prediction | `cubic_zirconia.csv` | `data/raw/cubic_zirconia.csv` |
| credit_card | kaggle.com/datasets/sakshigoyal7/credit-card-customers | `BankChurners.csv` | `data/raw/BankChurners.csv` |

Then run `python -m tabgen data download --dataset gem_price,credit_card`, which converts the files
into the standard `data/raw/<name>.csv` + `.meta.json`.

The same manual route also works for `insurance.csv` / `housing.csv` if the mirrors are ever down.

## Cleaning applied (`tabgen.data.preprocess.clean_table`)

* Selects the features in the registry. For explicit lists (authors' CSVs), numeric columns come
  first and categorical columns second, as in `read_data.py`.
* `?` and empty strings become missing, and rows with any missing value are dropped
  (adult 32561→30162, gem_price 26967→26270, housing 20640→20433).
* Dropped columns: credit_card `CLIENTNUM` and the two `Naive_Bayes_Classifier_*` leakage columns;
  sick `TBG` (all missing) and `TBG_measured` (constant).
* Commas inside category strings are replaced with `;`, because `,` is the serialization separator.

## Encoding applied (paper protocol, `TablePreprocessor`)

Categories become integer codes (sorted order), and every feature is MinMax-scaled to [0, 1] and
rounded to 2 decimals. Class labels become 0..K−1 and regression targets stay in raw units. The
scaler is fitted on the full table (`fit_scaler_on: full`, as in the authors' code; set `train` to
fit on the training split only). The split is a stratified 80/20 with `random_state = seed`.

## Notes on differences from the extended table

* `car`: the table says "6 features, 2 numerical, 6 categorical". All 6 features are categorical.
* `german`: OpenML types give 7 numerical and 13 categorical (table: 6 and 14). This only affects
  how the encoder treats one ordinal column.
* `sick`: dropping every row with a missing value leaves fewer rows than the table's 3,560.
  `audit data` prints the exact count. Set `dropna: false` in the registry and impute if you need
  the larger version.
* `fuel`, `insurance`, `gem_price`, `housing` are regression by default (MSE). The table's K=2 is
  available with `--binarize median` (y > median → 1). Those runs are written to
  `runs/<name>_bin/`.


## Warning: cardio leaks its label

In OpenML 1466 (`cardio`), columns **V26–V35 are an exact one-hot encoding of the 10-class
target**: these are the CTG spreadsheet's `A … SUSP` class indicators. For example, V26 = 1 for
exactly the 384 rows of class 1. A classifier trained on these columns reads the label off them,
which is why the Pred-LLM paper reports 1.0000 accuracy for both Original and Pred-LLM on cardio.
The OpenML-CC18 benchmark excluded this dataset as "too easy" without naming the cause.

* `cardio` is kept as published, so the paper's table can be compared.
* `cardio_noleak` (M = 25) drops V26–V35. Draw conclusions from it.
* `python -m tabgen audit data --dataset cardio` flags the columns: its leakage check now also
  tests each feature as a one-vs-rest score for every class.
