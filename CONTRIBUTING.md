# Contributing to the Project

Thank you for your interest in contributing to this project! We welcome
contributions from everyone. Here are some guidelines to help you get started:

## Getting Started

The dev cycle is triven by the command runner `just` (basically just make). Look at the justfile
to see the available commands. You can run `just` to see a list of all tasks.

A nixos environment is used to lock down the dev environment. You can use `nix
develop` to enter or alternatively there is a .envrc file to automatically
enable the environment when you enter the directory using direnv.

to run the tests, you simply run:

```bash
just test
```

## Code Style

Use Doxygen comments on maintained C/C++ definitions, including private helpers,
enum values, fields, constants, and static state. Describe behavior, constraints,
ownership, and errors where relevant; put comments at declarations so symbol
help can display them. Preserve generated documentation through its headers or
generators rather than editing generated wrappers.

`just docs` uses the `cpp` Nix shell so executable reports get matching installed
headers and library; Nix builds that library if necessary. The `cpp-tools` shell
and `just docs-audit` remain independent of an installed core. `just docs-audit` reports
missing documentation, and `just docs-audit --check --path include/path.hpp`
checks a finished batch. See the [documentation completion plan](docs/documentation-plan.md)
for the checklist, scope, and verification requirements.
Completed source batches are listed in `docs/documented-paths.txt` and enforced
by CI with `python3 scripts/check_documentation.py --check --completed` after
fresh XML generation.
`just docs-hover` verifies representative descriptions through clangd at their
definitions, and `just test-docs` tests the auditor against real Doxygen output.

The repository's [documentation maintenance skill](skills/maintain-docs/SKILL.md)
and [library development skill](skills/change-library/SKILL.md) describe the
repeatable workflow for future changes. Keep contracts current in the same patch
as the symbols they describe.
