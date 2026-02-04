"""
setup.py
--------
Setup configuration for the OverNaN package.
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="overnan",
    version="0.2.0",
    author="Amanda Barnard",
    author_email="amanda.s.barnard@anu.edu.au",
    description="Oversampling for imbalanced learning with missing values (SMOTE, ADASYN, ROSE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amaxiom/OverNaN",
    py_modules=["overnan"],  # Single-file module
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
        "examples": [
            "xgboost>=1.4.0",
            "matplotlib>=3.3.0",
        ],
    },
)

