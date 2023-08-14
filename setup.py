from setuptools import setup, find_packages

# Get version number
def getVersionNumber():
    with open("hadamuxer/VERSION", "r") as vfile:
        version = vfile.read().strip()
    return version


__version__ = getVersionNumber()

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/hekstra-lab/hadamuxer/issues",
    "Source Code": "https://github.com/hekstra-lab/hadamuxer",
}


LONG_DESCRIPTION = """
"""

setup(
    name="hadamuxer",
    version=__version__,
    author="Kevin Dalton, Maggie Klureza",
    license="MIT",
    include_package_data=True,
    packages=find_packages(),
    long_description=LONG_DESCRIPTION,
    description="Demux Hadamard TR-X data.",
    project_urls=PROJECT_URLS,
    python_requires=">=3.8,<3.12",
    url="https://github.com/hekstra-lab/hadamuxer",
    install_requires=[
        "torch",
        "fabio",
        "tqdm",
        "matplotlib",
        "seaborn",
    ],
    scripts=[
    ],
    entry_points={
        "console_scripts": [
            "hadamuxer.demux=hadamuxer.multiplex:main",
        ]
    },
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "pytest-cov", "pytest-xdist>=3"],
)
