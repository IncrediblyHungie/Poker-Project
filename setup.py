from setuptools import setup, find_packages

setup(
    name="pluribus_poker_bot",
    version="0.1.0",
    description="Pluribus-quality poker bot using Monte Carlo CFR",
    author="Poker Bot Developer",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pypokerengine==1.0.1",
        "numpy==1.24.0",
        "scipy==1.10.0",
        "scikit-learn==1.3.0",
        "joblib==1.3.0",
        "pyyaml==6.0",
        "tqdm==4.65.0"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)