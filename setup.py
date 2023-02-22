from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("group_scad", sources=["group_scad.pyx"], include_dirs=['.', get_include()])
setup(name="group_scad", ext_modules=cythonize([ext],gdb_debug=True))

