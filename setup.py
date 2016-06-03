#!/usr/bin/python
from setuptools import setup

with open('README.md') as file:
    long_description = file.read()

setup(
    name='pymlx',
    packages=['pymlx'],
    version='0.0.9',
    description='Yet another machine learning framework',
    long_description=long_description,
    author='Kenneth Tran',
    author_email='one@kentran.net',
    url='https://github.com/zer0n/MLx',
    keywords="machine learning",
    license="BSD3",
    install_requires=[
        'numpy>=1.9.2',
        'scipy>=0.14.0',
        'pandas>=0.15.0',
        'scikit_learn>=0.16',
        'xgboost',
        'matplotlib>=1.4.3',
        'dill>=0.2.2',
        'ipywidgets'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Topic :: Database",
        "Topic :: Database :: Database Engines/Servers",
        "Operating System :: OS Independent"
    ]
)
