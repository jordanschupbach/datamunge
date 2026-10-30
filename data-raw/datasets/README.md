# Raw data for the built-in datasets

These CSV files are the source of truth for the datasets compiled into the
library (`datamunge::datasets::wine()`, `titanic()`, ...). `just generate-datasets`
turns each one into `src/datamunge/datasets/embedded/<name>.cpp`, which embeds
the CSV text; the library parses it with `datasets::parse_csv()` on each call.

To add a built-in dataset: drop `<name>.csv` here, add it to `DATASETS` in
`scripts/generate_datasets.py`, run `just generate-datasets`, then declare and
document the accessor in `include/datamunge/datasets/datasets.hpp`.

| File | Origin | Notes |
|------|--------|-------|
| `wine.csv` | UCI Wine (Forina et al.), via `sklearn.datasets.load_wine()` | `od280/od315_of_diluted_wines` renamed `od280_od315`; class as `cultivar` = `class_0..2` |
| `breast_cancer.csv` | UCI Breast Cancer Wisconsin (Diagnostic), via `sklearn.datasets.load_breast_cancer()` | spaces in names become `_`; `diagnosis` = `malignant`/`benign` |
| `diabetes.csv` | Efron et al. (2004) diabetes data, via `sklearn.datasets.load_diabetes(scaled=False)` | raw (unscaled) features; target as `progression` |
| `digits.csv` | UCI Optical Recognition of Handwritten Digits (8x8 test set), via `sklearn.datasets.load_digits()` | pixels 0..16 as `pixel_0..63`, row-major |
| `faithful.csv` | R `datasets::faithful` | |
| `misra1a.csv` | NIST StRD, Misra (1978), [source data](https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/Misra1a.dat) | 14 observed adsorption rows, `pressure` (x), `volume` (y); original units unspecified; no missing values |
| `mtcars.csv` | R `datasets::mtcars` | row names moved to a `model` column |
| `airquality.csv` | R `datasets::airquality` | empty cells are R's `NA` |
| `air_passengers.csv` | R `datasets::AirPassengers` (Box & Jenkins series G) | `ts` index written as `year`, `month` |
| `titanic.csv` | Kaggle "Titanic" training set (891 passengers), from `datasciencedojo/datasets` | empty cells are missing values |
| `volcano.csv` | R `datasets::volcano` (Maunga Whau, 10 m grid) | matrix written long: `x` = (row-1)*10, `y` = (column-1)*10, `elevation` |
| `eight_schools.csv` | Rubin (1981); Gelman et al., *Bayesian Data Analysis*, Table 5.2 | |
| `orthodont.csv` | Potthoff and Roy (1964), via R `nlme::Orthodont` | columns lower-cased; rows sorted by `subject`, then `age` |

Downloaded datasets (`diamonds`, `california_housing`, MNIST, Fashion-MNIST,
CIFAR-10/100) are not stored here; their pinned URLs and MD5 checksums live in
`src/datamunge/datasets/download.cpp`.
