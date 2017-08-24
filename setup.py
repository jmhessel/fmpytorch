import numpy
from setuptools import setup, find_packages, Extension

extensions = [
    Extension('fmpytorch.second_order.second_order_fast_inner',
              ["fmpytorch/second_order/second_order_fast_inner.c"],
              include_dirs=[numpy.get_include()]),
]


setup(name="fmpytorch",
      version="0.1",
      description="A fast factorization machine in pytorch",
      author="Jack Hessel",
      author_email="jmhessel@gmail.com",
      license="MIT",
      packages=find_packages(),
      ext_modules=extensions,
      install_requires=['numpy'],
      zip_safe=False,
      classifiers=[
          'Intended Audience :: Science/Research',
          'Intended Audience :: Developers',
          'Programming Language :: C',
          'Programming Language :: Python',
          'Topic :: Software Development',
          'Topic :: Scientific/Engineering',
          'Operating System :: POSIX',
          'Operating System :: Unix',
          'Operating System :: MacOS']
)
