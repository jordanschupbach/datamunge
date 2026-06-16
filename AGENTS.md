# Agent Notes (Template Repo)

This repository is a _template_ for multi-language libraries.

The single source of truth is the C/C++ core library. Bindings for other
languages are generated (mostly automatically) via SWIG, and each language
should feel like a "real" native library in that ecosystem: standard layout,
standard packaging, standard test runner, and standard developer workflow.

If you are working on this repo, optimize for:

- repeatable local dev (`just …`, nix dev shells),
- standard packaging per language (PyPI/CRAN/npm/NuGet/Maven/etc),
- keeping the C/C++ API stable and well-tested (bindings follow),
- minimal per-language glue (prefer SWIG + thin wrappers),
- template-friendliness (easy to rename/fork without bespoke steps).

## Repo Principles

- C/C++ is canonical: add features/fixes in the core first, then regenerate and
  update bindings.
- Each language has its own toolchain and tests: avoid "one weird script"
  workflows; prefer that `just test-<lang>` runs the ecosystem's normal tests.
- Generated code is OK when necessary, but keep it deterministic and produced
  by explicit `just prebuild-<lang>` / `just build-<lang>` tasks.
- Keep renameability: this repo is designed to be cloned and renamed via
  `./rename_octra <new_project_name>`. Avoid hard-coding `octra` in new places
  unless it is expected to be renamed by that script.

## Where Things Live (Conventions)

- Core implementation: `source/`, `include/`, `src/` (CMake drives the build).
- SWIG interface files: `prebindings/<binding>/src/*.i`
- Generated wrapper sources: typically `src/*_wrap.cpp` (varies by language).
- Language packages (examples; not exhaustive):
  - Python packaging: `pyproject.toml`, `setup.py`, `Manifest.in`
  - R packaging: `DESCRIPTION`, `NAMESPACE`, `R/`
  - JavaScript packaging: `package.json`, `binding.gyp`, `index.js`
  - Java packaging: `joctra/`, native side in `joctra-octra/`
  - .NET packaging: `octradotnet/`, tests in `octradotnet.tests/`
- Cross-language binding tests: `bindings_tests/<lang>/…`
- Task runner: `justfile` (the entrypoint for most workflows)
- Nix dev shells/tooling: `flake.nix` and `nix/`

## The "Binding Contract"

When adding or modifying a binding:

- Provide `just prebuild-<lang>` that runs SWIG with the right flags and writes
  generated artifacts to the expected locations.
- Provide `just build-<lang>` that builds the binding/package using idiomatic
  tools for that ecosystem.
- Provide `just test-<lang>` that runs _real_ tests for that language.
- Provide `just run-<lang>` and/or `just repl-<lang>` when it makes sense.
- Add at least one minimal test in `bindings_tests/<lang>/` that exercises:
  - loading/importing the module/package,
  - calling 1-2 functions,
  - creating/disposing at least one non-trivial object (if applicable).

Also update (as applicable):

- `README.md` bindings table (Implemented ✅/✖),
- `flake.nix` to add a dev shell for the language (`.#<lang>`),
- `.gitignore` for language build artifacts,
- CI configuration (if present in this template) to include the new binding.

## Adding a New Language (Checklist)

1. Decide the ecosystem-level artifact and name:
   - package name (PyPI/npm/CRAN/etc),
   - module/namespace naming conventions,
   - how the native library is loaded (shared library name, search path).

2. Create `prebindings/<binding>/src/<binding>.i`:
   - keep it small; include the common headers you want to expose,
   - use SWIG typemaps/directives sparingly and document any non-obvious ones,
   - avoid binding STL types unless you must; prefer C-friendly APIs.

3. Wire up the build:
   - add the SWIG generation step to `just prebuild-<lang>`,
   - add the language build step to `just build-<lang>`,
   - ensure it works in `nix develop .#<lang>` (add a shell if missing).

4. Package it:
   - add the minimal packaging metadata expected by that ecosystem,
   - ensure artifacts include/ship the compiled extension correctly.

5. Tests:
   - add binding tests under `bindings_tests/<lang>/`,
   - ensure `just test-<lang>` runs in a clean environment (venv, local build,
     etc), not relying on global user state.

6. Document:
   - add a small `examples/<lang>/` example (optional but preferred),
   - update `README.md` and any per-language README/docs.

## Common Pitfalls

- Breaking the "template rename" flow by adding hard-coded `octra` references
  that `rename_octra` does not update (prefer central constants/config).
- Divergent language APIs: keep bindings conceptually consistent across
  languages unless a language strongly prefers a different shape.
- Relying on non-portable loader behavior:
  - on Linux/macOS/Windows, shared library naming and search paths differ;
    prefer explicit load directives or packaging-time configuration.
- Shipping generated files without a clear regeneration command: always make it
  possible to reproduce generated output from repo sources using `just`.

## Preferred Workflow Commands

- Core build/tests:
  - `just build` / `just test` / `just examples`
- Language workflows (examples; see `justfile` for the full set):
  - `just prebuild-<lang>`
  - `just build-<lang>`
  - `just test-<lang>`
  - `just run-<lang>` / `just repl-<lang>`

## Scope Note: "As Many Languages As Possible"

This template aims to support many languages, but quality matters:

- Only mark a language as "Implemented ✅" when it has:
  - a working build in a dev shell,
  - a package skeleton aligned with that ecosystem,
  - at least one automated test.
