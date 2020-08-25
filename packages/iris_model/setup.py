import io
import os
from pathlib import Path


from setuptools import find_packages, setup

# Package meta-data.
NAME = 'iris_model'
DESCRIPTION = 'Train and deploy iris model'
URL = 'https://github.com/dinabandhu50/IRIS_PROECT2.git'
EMAIL = 'beheradinabandhu50@gmail.com'
AUTHOR = 'Dinabandhu Behera'
REQUIRES_PYTHON = '>=3.7.0'


# What packages are required for this module to be execured?
def list_reqs(fname='requirement.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()
