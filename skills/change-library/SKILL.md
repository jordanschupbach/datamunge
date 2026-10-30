---
name: change-library
description: Implement or change this repository's C/C++ library and generated language bindings using its canonical core, native packaging, and documented symbol contracts.
---

# Change the library and its bindings

Read [AGENTS.md](../../AGENTS.md) before choosing a source file or workflow. The
current project is named in the root `CMakeLists.txt`; old template names in
instructions do not identify a separate core library. Preserve renameability.

Find the canonical definition under `include/` and `src/datamunge/` before
editing a language facade. Extend behavior in the core first and keep glue thin.
The normal entry points are `just prebuild-<lang>`, `just build-<lang>`, and
`just test-<lang>`; inspect the actual recipe before selecting a shell. Use the
language's native test runner and package layout. `.#cpp-tools` provides build
tools without first building an installed copy of the library; `.#cpp` also
provides an installed core and suits installed-library workflows.

Distinguish handwritten wrappers from generated output by inspecting their
headers and regeneration scripts. SWIG interfaces are under `prebindings/` and
language-specific `swig/` directories. Python/R core regeneration uses
`scripts/prebuild_bindings.py`; NN APIs and examples have separate generators.
Prefer the existing `just` recipe over inferring that one generator covers all
interfaces. Keep deterministic outputs and update generated files only through
their source generator.

Apply [maintain-docs](../maintain-docs/SKILL.md) to changed or added symbols and
update their contracts in the same patch. For a new binding, follow the Binding
Contract in AGENTS.md, including loading/function/object-lifetime tests, a dev
shell, package metadata, and truthful implementation status. A missing-doc audit
is useful evidence but does not replace behavior tests.

Choose checks proportional to changed behavior. Comments alone need Doxygen and
representative symbol-help checks; build logic needs its actual configure/build
path; binding changes need regeneration and affected native tests. Report
remaining coverage and environment limitations precisely. Keep unrelated working
tree changes intact.
