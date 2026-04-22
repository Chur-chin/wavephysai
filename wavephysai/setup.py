from setuptools import setup, find_packages

setup(
    name="wavephysai",
    version="0.1.0",
    author="Chur Chin",
    author_email="wavephysai@dongeuimc.kr",
    description="Physical Wave Neuromorphic Computing for Full-Body Humanoid Control",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "numba>=0.57",
        "torch>=2.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
