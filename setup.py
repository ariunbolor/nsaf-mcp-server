"""
Setup script for the Neuro-Symbolic Autonomy Framework (NSAF) prototype.
"""

from setuptools import setup, find_packages

setup(
    name="nsaf",
    version="0.1.0",
    description="Neuro-Symbolic Autonomy Framework (NSAF) Prototype",
    author="AI Research Team",
    author_email="research@example.com",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
        "pytest>=7.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
