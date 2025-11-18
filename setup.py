"""
Setup script for CAM-FD package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
else:
    requirements = []

setup(
    name="cam-fd",
    version="0.1.0",
    author="Brandone Fonya",
    author_email="bfonya@andrew.cmu.edu",
    description="CAM-FD: Robust Medical Image Classification using Class Activation Map Guided Feature Disentanglement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fonyabrandone/cam-fd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "ipython>=8.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "camfd-train=scripts.train:main",
            "camfd-evaluate=scripts.evaluate:main",
        ],
    },
)