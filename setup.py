import os
from setuptools import setup, find_packages

# Загрузка длинного описания из README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Загрузка зависимостей из requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="openmocap",
    version="0.1.0",
    author="OpenMoCap Team",
    author_email="info@openmocap.org",
    description="Открытая система захвата движения без маркеров",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorganization/openmocap",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "openmocap=openmocap.core.cli:main",
        ],
    },
)