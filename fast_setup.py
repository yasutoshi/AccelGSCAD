from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("fast_group_scad", sources=["fast_group_scad.pyx"], include_dirs=['.', get_include()])
setup(name="fast_group_scad", ext_modules=cythonize([ext],gdb_debug=True))

