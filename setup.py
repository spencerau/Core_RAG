from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="core-rag",
    version="0.1.0",
    author="Spencer Au",
    author_email="spencerau96@gmail.com",
    description="A generic, domain-agnostic RAG pipeline with Qdrant vector search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spencerau/Core_RAG",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: AGPL3 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
    install_requires=requirements,
)
