from setuptools import setup, find_packages

setup(
    name="llm_fact_acc",
    version="0.1.0",
    packages=find_packages(),
    install_requires= [
        "torch",
        "nltk",
        "transformers",
        "datasets",
    ],
    description="A Python library for LLM factual accuracy metrics.",
    author="Shweta Agrawal",
    url="https://github.com/ashweta1/llm_fact_acc",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)