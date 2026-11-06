# Report shells

`nix develop .#report-<lang>` gives a shell with the language runtime and the **nix-built**
datamunge binding for that language, so a report block can run with no `just build-<lang>`
step. mepml blocks use them through their header, e.g.

    ```{python, nix-shell="git+file:../..#report-python", ...}

Nix rebuilds a binding only when its build inputs change (see `../source.nix`); editing reports
never does. The ordinary dev shells (`.#python`, ...) deliberately leave the nix bindings out
and use the working-tree builds instead; see the comment above `localBindingsHook` in flake.nix.

Each file here takes the shared argument set `reportShellArgs` from flake.nix and returns a
`mkShell`.
