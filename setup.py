import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parallelm",
    version="1.0.0",
    author="Artemis Panagopoulou",
    author_email="artemisp@seas.upenn.edu",
    description="Templates for parallel and efficient gpu training of language models powered by transformers and PyTorch lightning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artemisp/paralleLM",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)