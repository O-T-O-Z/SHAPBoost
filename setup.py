import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

with open("requirements.txt") as f:
    requirements = list(f.read().splitlines())

setup(
    name="shapboost",
    version="1.0.1",
    description="A Python package for the SHAPBoost feature selection algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/O-T-O-Z/SHAPBoost",
    author="Ömer Tarik Özyilmaz, Tamas Szili-Török",
    author_email="o.t.ozyilmaz@umcg.nl, t.szili-torok@umcg.nl",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="Feature Selection, Boosting, SHAPBoost",
    package_dir={"shapboost": "shapboost"},
    packages=find_packages(include=["shapboost", "shapboost.*"]),
    python_requires=">=3.10, <4",
    install_requires=requirements,
)
