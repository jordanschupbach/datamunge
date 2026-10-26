"""Changing immutable core paths must relink cached Python extensions."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def test_core_prefix_change_forces_link_without_recompiling_objects(tmp_path):
    compile_forces, link_forces = [], []

    class Compiler:
        compiler_type = "other"
        force = False

        def compile(self, *args, **kwargs):
            compile_forces.append(self.force)

        def link_shared_object(self, *args, **kwargs):
            link_forces.append(self.force)

    class Parent:
        def build_extensions(self):
            for extension in self.extensions:
                self.build_extension(extension)

        def build_extension(self, extension):
            self.compiler.compile([])
            self.compiler.link_shared_object([])

    # Load the actual command class without invoking setup() or generating SWIG.
    path = Path(__file__).resolve().parents[2] / "setup.py"
    tree = ast.parse(path.read_text())
    definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BuildExt"
    )
    namespace = dict(build_ext=Parent, Path=Path, json=json, sys=sys, system_core=True)
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    command = namespace["BuildExt"]()
    command.force = False
    command.parallel = 1
    command.build_temp = str(tmp_path)
    command.compiler = Compiler()
    extension = SimpleNamespace(
        name="pydatamunge._nn_native",
        extra_compile_args=[],
        extra_link_args=["-L/core/first"],
        libraries=[],
        library_dirs=[],
        extra_objects=[],
    )
    command.extensions = [extension]
    command.build_extensions()
    extension.extra_link_args = ["-L/core/first"]
    command.build_extensions()
    extension.extra_link_args = ["-L/core/second"]
    command.build_extensions()
    assert compile_forces == [False, False, False]
    assert link_forces == [True, False, True]
    assert not command.force and not command.compiler.force
