---
name: maintain-docs
description: Maintain Doxygen comments (full tags, compiled examples, original-source references), binding help, and documentation coverage when changing this repository's symbols or auditing its documentation.
---

# Maintain repository documentation

Read [the completion plan](../../docs/documentation-plan.md) for the current
standard, finished batches, and remaining coverage. Keep documentation changes
within the user's requested code changes or audit scope; an ordinary fix does
not require completing the repository's existing documentation backlog.

## Document the source of truth

C/C++ headers and implementations are canonical. Use adjacent `/// @brief`
comments and `///<` field/enum descriptions so Doxygen and language servers can
attach prose to the definition. Document private helpers, aliases, static state,
and file purposes too. Explain constraints, units/shapes, defaults, mutation,
ownership/lifetimes, errors, and concurrency when they matter. Preserve accurate
existing prose and read implementations before promising behavior.

Put public contracts on declarations. Check overloads separately. A group heading
does not document its members. Review function-local static variables and named
lambdas directly: Doxygen XML does not enumerate those definitions. Namespace
descriptions can live at one canonical declaration, with documented implementation
namespaces where appropriate.

When switching from a trailing `///<` field description to a leading `///`
description for the next field, separate the blocks with a blank line. Otherwise
Doxygen can merge the next description into the previous field's comment.

For generated APIs, edit the header, SWIG interface, or generator and regenerate
with the established `just prebuild-<lang>` command. Do not hand-edit generated
proxies or SWIG runtime code. Read the relevant regeneration script to identify
which outputs actually carry documentation; Python/Java/C# support is not proof
that every target propagates comments. Keep native docstrings/help in handwritten
facades when the language's editor tooling requires them.

## Required content for every documented symbol

Documentation is not finished until a reader can use the symbol from its hover
alone. Every new or changed public symbol (function, method, constructor, class,
struct, alias, enum) gets the full set, in this order:

```cpp
/// @brief One-sentence summary (name the R/dplyr/pandas/ggplot analogue if there is one).
///
/// Details: semantics, units/shapes, defaults, null/NaN/empty handling, ownership and
/// lifetime, mutation, complexity when it matters, thread-safety when relevant.
///
/// @tparam T  ...                       (each template parameter)
/// @param name  ...                     (every parameter, in order, with its default)
/// @return  ...                         (unless void or a constructor)
/// @throws std::invalid_argument  ...   (each exception the implementation can throw)
///
/// @par Example
/// @code
/// const auto x = dm::runif(0, 1, 5);
/// std::print("{}\n", dm::length(x));   // 5
/// @endcode
///
/// @par References
/// - Original source, e.g. A. Author (1977). "Title". Journal 12(3), 45-67. doi:10.xxxx/yyyy
/// - Canonical web reference, e.g. https://www.rdocumentation.org/packages/base/topics/Round
///
/// @see related_function(), RelatedClass
```

- **Examples are mandatory** for every user-facing function, method, and class.
  Write them as a user would (`dm::` spelling, `#include <dm/dm.hpp>`), keep them
  short, and **compile and run every one**: `// ...` output comments must match the
  real output. Overloads may point to the primary overload's example; trivial
  accessors (`size()`, `begin()`) and private members need no example but still
  need `@brief`, `@param`, and `@return`.
- **References are expected whenever behavior comes from outside the codebase**:
  an algorithm, statistical method, numerical scheme, file format, or a function
  that mirrors another library (R, dplyr, ggplot2, pandas, NumPy). **Prefer the
  original source**: the paper or book that introduced the method (with DOI or a
  stable URL), or the official specification or upstream documentation, over
  blog posts or secondary summaries. Add secondary sources only alongside the
  original, for example an open-access copy. Say what was followed when the
  implementation deviates (a variant, a correction, a different default). Never
  invent a citation: check that the title, authors, year, and link are real, and
  leave the reference out rather than guess.
- Do not paste a reference into simple code that has no external origin, such
  as accessors and plain containers.
- Use exactly these tags (`@par Example`, `@code`, `@par References`, `@see`) so
  editor hover can lay them out as sections. Keep a blank `///` line between a
  group marker (`/// @{`) and the first member's comment, and between a trailing
  `///<` and the next leading block.
- Members inherited or forwarded (`Same as X but ...`) may be brief, but they
  still need their own `@param` and `@return`.

## Check the affected area

Use `just docs-audit` to generate fresh core XML and print missing definitions.
Use repeatable `--path` arguments and `--check` for a finished batch, for example:

```sh
just docs-audit --check --path include/datamunge/vector.hpp --path src/datamunge/vector.cpp
```

The full machine-readable report is `build/debug/docs/coverage.json`. The script
`scripts/check_documentation.py --help` lists reporting options. It validates
inventory freshness and file completeness before checking descriptions. If Nix
is unavailable, configure `docs/` with CMake, build `GenerateDocsXml`, and run
that script with the same arguments. `just docs` builds the website (from Doxygen's
XML, `scripts/website/`) in the `cpp` shell so executable reports have matching
installed headers/library; Nix may build that core package first. Audit-only
workflows keep using `cpp-tools` without an installed core. Doxygen's warnings
are in `build/debug/docs/audit/warnings.log`.

`EXTRACT_ALL` is intentional: turning it off hides missing definitions from
coverage. Doxygen's own undocumented-symbol warnings are disabled by that option,
so a successful Doxygen process alone does not establish documentation coverage.
Presence checks do not establish semantic accuracy, parameter completeness, or
hover integration. Inspect generated descriptions and language-server help for
representative changed symbols; report mep UI verification separately if no
display is available.

`just docs-hover` checks representative definition help through clangd and saves
the replies in `build/debug/docs/hover.json`. Its temporary compilation database
uses checkout headers and the selected compiler, without local `.clangd` overrides.
Extend its targets when a new documentation syntax or symbol kind needs a check.
`just test-docs` runs real Doxygen fixtures for the coverage auditor, plus pytest
checks for report toolchain failures, preparation paths and website image export.

Record completed files and evidence in the plan. Preserve
`docs/documentation-baseline.json` as the initial reference; do not overwrite it
to conceal regressions. Check off only work that was verified. New language
areas need an explicit inventory before their coverage can be claimed. Keep this
skill current when commands, scope, or regeneration behavior changes.

Add a verified batch to `docs/documented-paths.txt`; `just docs-audit --check
--completed` enforces all listed paths. Linux CI runs this gate and the auditor's
integration tests. Keep incomplete areas out of that manifest until they pass.
