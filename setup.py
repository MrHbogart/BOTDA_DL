"""
Setup configuration for BOTDA Deep Learning package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="botda-dl",
    version="1.0.0",
    author="BOTDA Thesis Project",
    author_email="",
    description="Unified BOTDA deep learning and signal analysis framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "pdnn": [
            "pandas",
            "h5py",
            "tensorflow-probability",
            "ipython",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
