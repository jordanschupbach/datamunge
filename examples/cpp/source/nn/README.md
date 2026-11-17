# Neural-network C++ reports

All416 standalone NN examples have verified executable replacements under
[`docs/mepml/nn`](../../../../docs/mepml/nn/).
Report discovery follows the files in that collection.

Run a report by its original name with `just run-nn NAME`, or use
`just docs-mepml-run REPORT.mepml`. The report contains its complete editable
C++ program, computed results and figures. Training budgets are set in that
program. The obsolete standalone CMake build recipes are retired.

Reusable implementations live in datamunge. Binding compatibility headers live
in `prebindings/nn/include/common`. Native Python and R examples remain in their
own language directories.
