"""
databhaiml
~~~~~~~~~~

This module provides utilities for loading models and 
other functionalities for the DataBhai Application.
"""

from setuptools import setup, find_packages

MODULE_DESCRIPTION = '''
This package is used to load models and provide utilities for DataBhai Application
'''
setup(
    name="databhaiml",
    version='0.1.9.9',
    packages=find_packages(include=["databhaiml", "databhaiml.*"]),
    url='',
    license='',
    author='Shrijeeth',
    description=MODULE_DESCRIPTION,
)
