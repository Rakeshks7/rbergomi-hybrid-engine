import sys
from setuptools import setup, Extension
import pybind11

cpp_args = ['-std=c++11']
if sys.platform == 'darwin':
    cpp_args = ['-std=c++11', '-mmacosx-version-min=10.7', '-stdlib=libc++']

ext_modules = [
    Extension(
        'rbergomi_cpp',
        ['src/cpp/rbergomi_engine.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=cpp_args,
    ),
]

setup(
    name='rbergomi_cpp',
    version='0.0.1',
    author='Quant Dev',
    description='C++ Backend for rBergomi',
    ext_modules=ext_modules,
)