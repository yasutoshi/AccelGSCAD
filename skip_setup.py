from distutils.core import setup, Extension
from Cython.Build import cythonize
from numpy import get_include

ext = Extension("skip_group_scad", sources=["skip_group_scad.pyx"], include_dirs=['.', get_include()])
setup(name="skip_group_scad", ext_modules=cythonize([ext],gdb_debug=True))

