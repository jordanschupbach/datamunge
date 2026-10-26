# TensorFlow CPU comparison quick start

The single [performance plan and report](../PERFORMANCE.md) contains the full
case catalog, measurement/accuracy contract, optimization results, limitations,
and every current/historical TensorFlow/native comparison table.

```sh
python3.13 -m venv build/tensorflow-venv
build/tensorflow-venv/bin/python -m pip install -r benchmarks/tensorflow-requirements.txt
just bench-tensorflow --validate-only --output benchmarks/results/tensorflow/local-validation.json
just bench-tensorflow --paired-native --label local
```

The optional harness reads `tensorflow_cases.def`, requires verified Release core
and driver builds, and keeps CPU eager/graph timings separate. Output allocation
is timed; setup/tracing/host copies are excluded. `--case REGEX` selects a subset;
use the direct Python command in the report for shell-safe regex quoting.
`--validate-only` checks outputs without collecting calibrated timings.

JSON/TSV and provenance remain under `results/tensorflow/`. Successful timing runs
there automatically refresh the full tables in the single report. To refresh
manually, run `python3 scripts/performance_report.py`. Frozen external cores require
`--core-provenance RESULT.json`; unverified overrides are rejected.
