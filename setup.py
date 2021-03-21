import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent

with open("README.md", "r") as fh:
    read_me_description = fh.read()

with open("requirements.txt") as reqs:
    requirements = reqs.read().split("\n")

setup(
    name="mcnn",
    packages=['mcnn'],
    include_package_data=True,
    version="1.0.1",
    license="new BSD",
    description="MakeCNN is how we make sure truly everybody can benefit from ML.",
    author="Anish Lakkapragada",
    author_email="anish.lakkapragada@gmail.com",
    url="https://github.com/anish-lakkapragada/MakeCNN",
    keywords=["Machine Learning", "Deep Learning", "CNN"],
    install_requires=requirements,
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    python_requires=">=3",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "Programming Language :: Python :: 3",
    ],
)
