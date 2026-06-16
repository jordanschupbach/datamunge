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
