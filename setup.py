import os
from setuptools import find_packages, setup

os.chdir(os.path.dirname(os.path.abspath(__file__)))

setup(
    name='drl_nav',
    packages=find_packages(),
    version='0.0.1',
    description='Implementation of a simple deep Q-network agent.',
    author='Pierre Massey',
    license='',
    url="https://github.com/PierreMsy/DRL_navigation.git",
    include_package_data=True
)