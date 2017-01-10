from __future__ import print_function
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='molecules',
      version='0.0.1',
      description='design molecules with deep learning',
      author='Mehdi Cherti',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='mehdicherti@gmail.com',
      )
