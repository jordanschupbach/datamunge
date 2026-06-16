import logging
import os
import subprocess

from setuptools import Extension, setup

# TODO:
# def find_cpp_sources(directory):
#


def find_cpp_sources(directory):
    sources = []
    try:
        with open("lib_sources.txt", "r") as f:
            for line in f:
                # Strip whitespace and check if line is not a comment
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith("#"):
                    # Transform the line to the correct path
                    transformed_source = f"{directory}/{stripped_line}.cpp"
                    # Check if the file exists before appending
                    if os.path.exists(transformed_source):
                        sources.append(transformed_source)
    except FileNotFoundError:
        logging.error("lib_sources.txt file not found.")

    return sources


sources = find_cpp_sources("src") + ["src/octra_python_wrap.cpp"]


def get_pkgconfig_include_dirs(package):
    result = subprocess.run(
        ["env", "pkg-config", "--cflags", package],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        include_dirs = [
            arg[2:] for arg in result.stdout.split() if arg.startswith("-I")
        ]
        logging.debug("Include dirs: %s", include_dirs)  # Log the include directories
        return include_dirs
    else:
        logging.error("pkg-config failed: %s", result.stderr)
        return []


# sources = find_cpp_sources('source/octra')
# sources = [ 'source/octra/octra.cpp', 'src/octra_python_wrap.cpp' ]
include_dirs = ["include/"]  # + get_pkgconfig_include_dirs('libxml-2.0')
ext_modules = [
    Extension(
        "pyoctra._octra",
        sources=sources,
        include_dirs=include_dirs,
        libraries=[],  # 'xml2'
        extra_compile_args=["-O3", "-std=c++23"],
    ),
]

setup(
    name="pyoctra",
    version="0.0.1",
    author="Jordan Schupbach",
    author_email="jordan.schupbach@montana.edu",
    description="A Python interface to the octra C/C++ library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jordanschupbach/octra",
    # packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="Unlicense",
    packages=["pyoctra"],
    package_dir={"": "src/"},
    ext_modules=ext_modules,
)


# from setuptools import setup, Extension
# import glob
# import os
#
#
# import os
# import subprocess
# from setuptools import setup, Extension
#
# def get_nix_include_path():
#     # Example command to get include paths using pkg-config
#     try:
#         include_path = subprocess.check_output(['pkg-config', '--cflags-only-I', 'your-library'])
#         # Strip out the include path
#         return include_path.decode().strip().split()[0].replace('-I', '')
#     except subprocess.CalledProcessError:
#         return []
#
# include_dirs = ["include/"]
# if os.environ.get('NIX_ENV'):
#     include_dirs.extend(get_nix_include_path())
#
# ext_modules = [
#     Extension(
#         "pyoctra._octra",
#         sources=['source/octra/octra.cpp', 'src/octra_python_wrap.cpp'],
#         include_dirs=include_dirs,
#         libraries=['xml2'],
#         extra_compile_args=["-O3", "-std=c++23"]
#     )
# ]

# setup(
#     name="pyoctra",
#     version="0.0.1",
#     packages=["pyoctra"],
#     ext_modules=ext_modules,
#     # Other setuptools parameters
# )

# setup(
#     name="pyoctra",
#     version="0.0.1",
#     ext_modules=ext_modules,
#     package_dir={"": "src"},
# )


# logging.basicConfig(filename='build.log', level=logging.DEBUG)
#
#
# # subprocess.run(['env', 'pkg-config', '--list-all'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# # subprocess.run(['env', 'pkg-config', '--cflags', 'libxml-2.0'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#
#
#
# def find_cpp_sources(directory):
#     return glob.glob(os.path.join(directory, '*.cpp'))
#
# # NOTE: need to be able to handle nix local environment properly
# def get_pkgconfig_include_dirs(package):
#     result = subprocess.run(['env', 'pkg-config', '--cflags', package], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     if result.returncode == 0:
#         include_dirs = [arg[2:] for arg in result.stdout.split() if arg.startswith('-I')]
#         logging.debug("Include dirs: %s", include_dirs)  # Log the include directories
#         return include_dirs
#     else:
#         logging.error("pkg-config failed: %s", result.stderr)
#         return []
#
#
# sources = find_cpp_sources('source/octra')
# include_dirs = ["include/"] + get_pkgconfig_include_dirs('libxml-2.0')
#
# print("Sources:", sources)
# print("Include dirs:", include_dirs)
# ext_modules = [
#     Extension(
#         'pyoctra._octra',
#         sources=sources,
#         include_dirs=include_dirs,
#         libraries=['xml2'],
#         extra_compile_args=["-O3", "-std=c++23"]
#     ),
# ]

# setup(
#     name="pyoctra",
#     version="0.0.1",
#     ext_modules=ext_modules,
#     package_dir={"": "src"},
#     # other metadata...
# )
