# Build source for the core library and its language bindings.
#
# Every package used to take the whole repository as `src`, so editing *any* file (a report, an
# .mepml notebook, a generated .svg, the justfile, ...) changed every package's hash and forced the
# core plus every binding to rebuild on the next `nix develop` / direnv reload.
#
# This file is the single place that says which files can affect a build. Anything listed in
# `notBuildInputs` is excluded, so editing it never triggers a rebuild. It is a denylist rather
# than an allowlist so that new build files (a new binding dir, a new top-level packaging file)
# are picked up without editing this file; when you add a new docs/examples-style directory,
# add it here.
{ lib }:
let
  fs = lib.fileset;
  root = ../.;
  under = rel: fs.maybeMissing (root + "/${rel}");

  notBuildInputs = fs.unions [
    # build products / local state
    (under "build")
    (under "dist")
    (under "result")
    (under "node_modules")
    (under ".direnv")
    (under ".cache")
    (under "tmp")

    # docs, examples, tests, reports: consumed by humans and test runners, never by package builds
    (under "docs")
    (under "examples")
    (under "tests")
    (under "bindings_tests")
    (under "benchmarks")
    (under "assets")
    (under "task")
    (under "playground")
    (under ".github")

    # nix + dev workflow files: changes here reach builds through derivation attributes, not src
    (under "nix")
    (under "flake.nix")
    (under "flake.lock")
    (under ".envrc")
    (under ".envrc.local")
    (under "justfile")
    (under "init.el")
    (under "list-of-algorithms.txt")
    (under "current_prompt.md")
    (under "TODO.org")
    (under "AGENTS.md")
    (under "CONTRIBUTING.md")
    (under "PERFORMANCE.md")

    # Measurement/editor-audit tooling is never executed by core/binding package builds.
    # Updating benchmark metadata or hover expectations must not rebuild the library.
    (under "scripts/benchmark_fruit_render.py")
    (under "scripts/benchmark_fruit_render_gpu.py")
    (under "scripts/benchmark_fruit_wrist.py")
    (under "scripts/check_documentation_hover.py")

    # report / notebook / plot output anywhere in the tree
    (fs.fileFilter (
      f:
      lib.any f.hasExt [
        "org"
        "mepml"
        "svg"
        "pdf"
        "html"
        "tex"
        "shp"
        "dbf"
      ]
    ) root)
  ];
in
fs.toSource {
  inherit root;
  fileset = fs.difference root notBuildInputs;
}
