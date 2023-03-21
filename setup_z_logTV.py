from setuptools import setup
from Cython.Build import cythonize

setup(
    name='z_logTV app',
    ext_modules=cythonize("z_logTV.pyx"),
    zip_safe=False,
)