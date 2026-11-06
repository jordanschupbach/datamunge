# Dataset studies

These reports start from a complete dataset and examine the whole modelling
workflow: provenance, exploratory analysis, an explicit fitting/validation/test
policy, model selection, convergence, held-out performance and failure analysis.
Numerical results come from executed native datamunge programs.

| Dataset | Report | Reproduction |
| --- | --- | --- |
| CIFAR-10 | [Compact ResNet-18, calibration and interpretability](cifar10/cifar10.mepml) | `cifar10/run.sh`; add `--train` to repeat tuning and training |

Each study keeps editable diagrams, rendered figures and compact numerical
artifacts beside its report. Raw datasets remain in the canonical shared cache;
transient binaries and rendering logs remain in ignored `build/` directories.
