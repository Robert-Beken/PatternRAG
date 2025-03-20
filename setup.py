from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="patternrag",
    version="1.0.0",
    author="Robert Beken",
    author_email="rbeken@gmail.com",
    description="A pattern-finding Retrieval-Augmented Generation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Robert-Beken/PatternRAG",
    project_urls={
        "Bug Tracker": "https://github.com/Robert-Beken/PatternRAG/issues",
        "Documentation": "https://github.com/Robert-Beken/PatternRAG/tree/main/docs",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Indexing",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "patternrag-ingest=patternrag.ingest:main",
            "patternrag-service=patternrag.service:main",
        ],
    },
    include_package_data=True,
)