from setuptools import find_packages, setup

setup(
    name="radimagenet-models",
    packages=find_packages(exclude=[]),
    version="0.1.1",
    install_requires=[
        "gdown",
    ],
)
