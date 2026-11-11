"""Self-contained native extension, or an explicit pkg-config system-core build."""

import os
import json
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def pkg_config(*args):
    return subprocess.check_output(
        ["pkg-config", *args, "datamunge"], text=True
    ).strip()


class BuildExt(build_ext):
    def initialize_options(self):
        super().initialize_options()
        # Concurrent compiler processes: MAX_JOBS, else CMake's/Nix's job settings, else all CPUs.
        self.parallel = int(
            os.environ.get("MAX_JOBS")
            or os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
            or os.environ.get("NIX_BUILD_CORES")
            or os.cpu_count()
            or 4
        )

    def build_extensions(self):
        for extension in self.extensions:
            if self.compiler.compiler_type == "msvc":
                extension.extra_compile_args += ["/O2", "/std:c++latest", "/EHsc"]
                if any("/datamunge/cad/" in str(s).replace("\\", "/") for s in extension.sources):
                    extension.extra_compile_args += ["/fp:strict"]
            else:
                extension.extra_compile_args += ["-O3", "-std=c++23"]
                # Python's default flags include -g, which makes each extension ~500 MB and
                # slows linking and wheel packing. DATAMUNGE_DEBUG_INFO=1 keeps debug info.
                if os.environ.get("DATAMUNGE_DEBUG_INFO") != "1":
                    extension.extra_compile_args += ["-g0"]
                if sys.platform != "darwin":
                    extension.extra_compile_args += ["-pthread"]
                    extension.extra_link_args += ["-pthread"]
        compiler = self.compiler
        # Objects are reused by timestamp alone, so recompile everything when the flags change.
        flags_state = Path(self.build_temp) / "compile-flags.json"
        flags_signature = json.dumps(
            # Sets: build_extensions appends its flags again whenever it runs.
            {e.name: [sorted(set(e.extra_compile_args)), sorted(map(str, e.define_macros)), sorted(set(e.include_dirs))]
             for e in self.extensions},
            sort_keys=True,
        )
        # A fresh build directory has nothing to recompile, so only a recorded change forces.
        previous_compiler_force = compiler.force
        if flags_state.exists() and flags_state.read_text() != flags_signature:
            compiler.force = True
        # Build extensions in dependency order, parallelizing individual compiler
        # invocations instead. setuptools' parallel option otherwise only runs
        # different extension modules concurrently.
        jobs = max(1, int(self.parallel or 1))
        previous_parallel = self.parallel
        self.parallel = None
        self._nn_core_objects = []
        for extension in self.extensions[1:]:
            if extension.name in ("pydatamunge._nn_examples", "pydatamunge._nn_native") and not system_core:
                core_sources = [s for s in sources if s not in wrapper_sources]
                self._nn_core_objects = compiler.object_filenames(
                    core_sources, output_dir=self.build_temp
                )
                extension.extra_objects = (
                    list(extension.extra_objects or []) + self._nn_core_objects
                )
        original_compile = compiler.compile
        original_link = compiler.link_shared_object

        def link_shared_object(*args, **kwargs):
            # build_ext.force does not reach UnixCCompiler._need_link. Force
            # only the link phase, retaining the normal object-file cache.
            previous_force = compiler.force
            if self.force:
                compiler.force = True
            try:
                return original_link(*args, **kwargs)
            finally:
                compiler.force = previous_force

        compiler.link_shared_object = link_shared_object
        if compiler.compiler_type == "unix":
            original_single = compiler._compile

            def compile_sources(*args, **kwargs):
                pending = []
                dependencies = kwargs.get("depends") or []
                newest_header = max(
                    (Path(p).stat().st_mtime_ns for p in dependencies), default=0
                )
                with ThreadPoolExecutor(max_workers=jobs) as pool:

                    def submit(obj, source, *compile_args, **compile_kwargs):
                        # Current setuptools recompiles every source whenever
                        # one extension needs updating. Keep unchanged objects.
                        target = Path(obj)
                        newest_input = max(
                            Path(source).stat().st_mtime_ns, newest_header
                        )
                        if (
                            not compiler.force
                            and target.exists()
                            and target.stat().st_mtime_ns >= newest_input
                        ):
                            return
                        pending.append(
                            pool.submit(
                                compile_single,
                                obj,
                                source,
                                *compile_args,
                                **compile_kwargs,
                            )
                        )

                    def compile_single(obj, source, ext, cc_args, extra_postargs, pp_opts):
                        # Robust CAD predicates require separately rounded
                        # operations in source-built wheels, including ARM.
                        if "/datamunge/cad/" in source.replace("\\", "/"):
                            extra_postargs = list(extra_postargs or []) + ["-ffp-contract=off"]
                        return original_single(obj, source, ext, cc_args, extra_postargs, pp_opts)

                    compiler._compile = submit
                    try:
                        objects = original_compile(*args, **kwargs)
                        for future in as_completed(pending):
                            future.result()
                        return objects
                    except BaseException:
                        for future in pending:
                            future.cancel()
                        raise
                    finally:
                        compiler._compile = original_single

            compiler.compile = compile_sources
        try:
            super().build_extensions()
            flags_state.parent.mkdir(parents=True, exist_ok=True)
            flags_state.write_text(flags_signature)
        finally:
            compiler.force = previous_compiler_force
            compiler.compile = original_compile
            compiler.link_shared_object = original_link
            self.parallel = previous_parallel

    def build_extension(self, extension):
        # An existing core .so may survive after its temporary objects were
        # cleaned. Recreate those objects before linking a self-contained demo.
        previous_force = self.force
        # A new immutable system-library prefix can have an older (normalized)
        # timestamp than this extension. Track link inputs as well as mtimes.
        link_state = Path(self.build_temp) / (extension.name + ".link.json")
        link_signature = json.dumps({
            "version": 1,
            "arguments": extension.extra_link_args or [],
            "libraries": extension.libraries or [],
            "library_dirs": extension.library_dirs or [],
            "objects": extension.extra_objects or [],
        }, sort_keys=True)
        if not link_state.exists() or link_state.read_text() != link_signature:
            self.force = True
        if extension is self.extensions[0] and any(
            not Path(obj).exists() for obj in getattr(self, "_nn_core_objects", [])
        ):
            self.force = True
        if extension.name in ("pydatamunge._nn_examples", "pydatamunge._nn_native") and extension.extra_objects:
            binary = Path(self.get_ext_fullpath(extension.name))
            if not binary.exists() or any(
                Path(obj).stat().st_mtime_ns > binary.stat().st_mtime_ns
                for obj in extension.extra_objects
            ):
                # Force the extension's link check, not compiler.force: core
                # object changes should not recompile unrelated demo sources.
                self.force = True
        try:
            super().build_extension(extension)
            link_state.parent.mkdir(parents=True, exist_ok=True)
            link_state.write_text(link_signature)
        finally:
            self.force = previous_force


system_core = os.environ.get("DATAMUNGE_USE_SYSTEM_LIB") == "1"
# Rosetta's generated extension, split into parts (scripts/prebuild_bindings.py).
wrapper_sources = sorted(p.as_posix() for p in Path("src").glob("datamunge_python_wrap*.cpp"))
sources = list(wrapper_sources)
compile_args, link_args, macros = [], [], []
if system_core:
    compile_args = shlex.split(pkg_config("--cflags"))
    link_args = shlex.split(pkg_config("--libs"))
    if sys.platform != "win32":
        link_args += ["-Wl,-rpath," + pkg_config("--variable=libdir")]
else:
    # Match CMake's canonical source tree; never silently omit a missing manifest entry.
    sources += sorted(p.as_posix() for p in Path("src/datamunge").rglob("*.cpp"))
    if sys.platform == "darwin":
        macros += [
            ("DATAMUNGE_HAVE_CBLAS", "1"),
            ("DATAMUNGE_HAVE_LAPACK", "1"),
            ("ACCELERATE_NEW_LAPACK", "1"),
        ]
        link_args += ["-framework", "Accelerate"]

extensions = [
    Extension(
        "pydatamunge._datamunge",
        sources=sources,
        include_dirs=["include"],
        language="c++",
        define_macros=macros,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        depends=sorted(
            p.as_posix()
            for root in (Path("include"), Path("src/datamunge"), Path("src/pydatamunge/rosetta"))
            for p in root.rglob("*")
            if p.suffix in (".hpp", ".h")
        ),
    )
]

# The functional NN API is part of the normal package, with no experiment code.
subprocess.run([sys.executable, "scripts/prebuild_nn_api.py", "python"], check=True)
extensions.append(Extension(
    "pydatamunge._nn_native",
    # Rosetta's generated extension, split into parts (scripts/prebuild_nn_api.py).
    sources=sorted(p.as_posix() for p in Path("src").glob("nn_python_wrap*.cpp")),
    # examples/nn: header-only example components bound by src/pydatamunge/rosetta/nn_bindings.hpp.
    include_dirs=["include", "examples/nn", "examples/bindings"], language="c++", define_macros=macros,
    extra_compile_args=list(compile_args), extra_link_args=list(link_args),
    depends=extensions[0].depends
    + sorted(p.as_posix() for p in Path("examples/nn/common").rglob("*.hpp")),
))

# Example support is opt-in: ordinary wheels keep the small core-only surface.
if os.environ.get("DATAMUNGE_BUILD_NN_EXAMPLES") == "1":
    subprocess.run([sys.executable, "scripts/prebuild_nn_examples.py"], check=True)
    generated = Path("build/nn-python/generated")
    extensions.append(
        Extension(
            "pydatamunge._nn_examples",
            sources=sorted(p.as_posix() for p in generated.glob("*.cpp")),
            include_dirs=[
                "include",
                "examples/bindings",
                "examples/nn",
                "examples/cpp/source/nn",
            ],
            language="c++",
            define_macros=macros,
            extra_compile_args=list(compile_args),
            extra_link_args=list(link_args),
            depends=extensions[0].depends
            + sorted(p.as_posix() for p in Path("examples/nn/common").rglob("*.hpp"))
            + ["examples/bindings/nn_examples.hpp"],
        )
    )

setup(
    name="pydatamunge",
    version="0.0.1",
    description="Python bindings to the datamunge C++ library",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Jordan Schupbach",
    license="Unlicense",
    license_files=["LICENSE", "third_party_licenses/mep-cad-LICENSE"],
    python_requires=">=3.9",
    packages=["pydatamunge"],
    package_dir={"": "src"},
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExt},
)
