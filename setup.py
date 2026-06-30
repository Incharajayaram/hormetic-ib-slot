from setuptools import setup, find_packages

setup(
    name="hormetic-ib-slot",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "opencv-python>=4.7.0",
        "Pillow>=9.0.0",
        "scikit-learn>=1.2.0",
    ],
)
