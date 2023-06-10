from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

packages = find_packages()
print(f"Found packages {packages}")

setup(
    name="nlpmaps",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DataScienceProjectDow/nlpmaps",
    author="Andrew Simon, Meenal Rawlani, Shijie Zhang",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Industry/Research",
        "Topic :: NLP word embedding",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="NLP, word embedding",
    packages=packages,
    python_requires=">=3.6, <4",
)