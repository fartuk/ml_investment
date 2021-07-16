import setuptools
from setuptools import find_packages
import os

here = os.path.abspath(os.path.dirname(__file__))

NAME = 'ml_investment'
DESCRIPTION = 'Machine learning tools for investment'
URL = 'https://github.com/fartuk/ml_investment'
EMAIL = 'fao3864@gmail.com'
AUTHOR = 'Artur Fattakhov'
PYTHON_REQUIRES = '>=3.6.0'
VERSION = "0.0.10"

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    REQUIRED = f.read().split('\n')
    
with open("README.rst", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()
    
setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    url=URL,
    project_urls={
        "Bug Tracker": "{}/issues".format(URL),
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=('tests', 'train', 'images', 'examples')),
    python_requires=PYTHON_REQUIRES,
    install_requires=REQUIRED,
)
