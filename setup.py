# setup.py

from setuptools import setup, find_packages

setup(
    name="insurance_risk_modeling",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
