"""
Sets up RynLib, the package.
Building this out to provide more consistency / versioning / all that shit
"""

import setuptools
from .__init__ import VERSION_NUMBER # this should be replaced by versioneer in the future

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RynLib",
    version=VERSION_NUMBER,
    author="Mark Boyer, Ryan DiRisio, Jacob Finney",
    author_email="b3m2a1@uw.edu",
    description="An internal package for doing DMC on HPCs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/McCoyGroup/RynLib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)