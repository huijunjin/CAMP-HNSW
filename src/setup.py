from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "ours_backend",
        ["ours_backend.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
]

setup(
    name="ours_backend",
    version="0.1.0",
    ext_modules=ext_modules,
)
