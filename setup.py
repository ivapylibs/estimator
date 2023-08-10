from setuptools import setup, find_packages

setup(
    name="estimator",
    version="1.0.0",
    description="Classes implementing estimation, filtering, and smoothing algorithms.",
    author="IVALab",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "dataclasses",
        "matplotlib",
        "scipy"
    ],
)
