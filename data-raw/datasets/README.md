# Raw data for the built-in datasets

These CSV files are the source of truth for the datasets compiled into the
library (`datamunge::datasets::wine()`, `titanic()`, ...). `just generate-datasets`
turns each one into `src/datamunge/datasets/embedded/<name>.cpp`, which embeds
the CSV text; the library parses it with `datasets::parse_csv()` on each call.

To register another small teaching table, add its CSV and a row to
`classic_catalog.tsv`, then run `just generate-datasets`. The generator checks
the size budget and CSV shape and emits both the embedded payload and the
registry entry. The legacy named C++ accessors are listed separately in
`scripts/generate_datasets.py`. List the generated source in
`lib_sources.txt` for builds that consume the native source manifest.

## Expansion plan

- Bundle complete, small teaching tables. Prefer canonical sources with clear
  provenance, preserve observations and missing values, and keep each added CSV at
  or under 64 KiB and all added CSVs at or under 256 KiB. Existing embedded tables
  keep their current storage. Additions are listed in `classic_catalog.tsv`.
- Cover experimental design and repeated measures (PlantGrowth, InsectSprays,
  warpbreaks, CO2 uptake, ChickWeight, sleep, Orange, Loblolly, npk, DNase),
  regression and calibration (cars, trees, pressure, anscombe, longley, stackloss,
  swiss, attitude, freeny, rock), and binary, ordinal and count outcomes (esoph,
  infert, HairEyeColor, UCBAdmissions, aggregated Titanic, crab satellites,
  cod parasites).
- Include pharmacokinetics and nonlinear kinetics (Theoph, Indometh, Puromycin,
  BOD, Formaldehyde), spatial and multivariate examples (USArrests, quakes,
  state socioeconomic indicators), and historical series (Nile, atmospheric CO2,
  LakeHuron, JohnsonJohnson, sunspots, lynx, Aussie population and more).
- Keep named row identifiers when they identify observations. Export
  contingency arrays as level columns plus `count`, preserve time-series
  observation times, and preserve factor labels as strings.
- Use the catalog as the shared inventory and dispatch table. Add the C++ call
  `dm::data("iris")` while retaining existing typed loaders and binding entry
  points. The same registry serves downloaded tables, using the existing
  verified local cache. MNIST, Fashion-MNIST and CIFAR remain batch streams,
  and graphs keep their typed loader.
- Verify every embedded table's catalog dimensions and check representative
  original values, missing values, offline lookup, and binary size.

### UCI Machine Learning Repository

`uci_catalog.tsv` is a pinned audit of UCI's public index. The native registry
includes official single-CSV tables with an explicit CC BY 4.0 or CC0 notice,
publisher attribution, and verified dimensions.
Only metadata enters the library. Unmodified CSVs download into the verified
cache when requested, as `uci_<id>/data.csv`. The in-library catalog exposes
each source's license, DOI, creators and checksum.

Refresh from a new workspace-local or temporary cache with
`python3 scripts/audit_uci_datasets.py CACHE --output NEW_CATALOG.tsv`. Inspect
`CACHE/decisions.json` before adopting the replacement manifest, then run
`just generate-datasets`. Archive-only and unverified sources remain
outside the single-table registry. The audit streams CSVs to disk and hashes
them in blocks. Existing audit caches automatically revisit the former 16 MiB
size exclusions.

The 2026-11-20 snapshot reviewed 690 entries and registered 142, including 12
CSVs over 16 MiB. The audit
found 479 without a canonical single CSV, 35 without the expected DOI or
creator record, 18 with row-count discrepancies, 11
with incompatible CSV structure, and five without an explicit supported
license. The accepted snapshots are all CC BY 4.0.

### Classic source snapshots

Base R tables come from the `datasets` package shipped with R 4.5.2.
`classic_catalog.tsv` records the original R object, table representation and
string-typed columns. See the
[R datasets reference](https://search.r-project.org/R/refmans/datasets/html/00Index.html)
for the data authors and original references. The two count-regression tables
come from the [countreg source package](https://zeileis.codeberg.page/countreg/),
which distributes its package under GPL-2 or GPL-3. Their measured observations
come from Hemmingsen et al. (2005) and Brockmann (1996), respectively. Only
the tabulated observations are exported here. For the exact inputs, use
[`CodParasites.rda`](https://raw.githubusercontent.com/r-forge/countreg/1811461f9881c7154d5a2baf74e3274cdd36ffbc/pkg/data/CodParasites.rda) and
[`CrabSatellites.rda`](https://raw.githubusercontent.com/r-forge/countreg/1811461f9881c7154d5a2baf74e3274cdd36ffbc/pkg/data/CrabSatellites.rda).
Run `Rscript --vanilla scripts/import_classic_datasets.R COUNTREG_DATA_DIR`
to refresh the snapshots, then `just generate-datasets`.

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
