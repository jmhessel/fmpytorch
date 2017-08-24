from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("second_order_fast_inner.pyx"),
    include_dirs=[numpy.get_include()]
)
