from setuptools import setup, find_packages

setup(
    name="coral-call",
    version="2.0",
    description="A dual context-aware basecaller for nanopore direct RNA sequencing",
    author="ShaoHui Xie",
    author_email="2060271006@email.szu.edu.cn",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.4",
        "matplotlib==3.10.0",
        "seaborn==0.13.2",
        "pandas==2.2.3",
        "scipy==1.15.1",
        "ont-fast5-api==4.1.3",
        "biopython==1.85",
        "parasail==1.3.4",
        "tqdm>=4.64.1",
        "pysam==0.22.1",
        "h5py==3.12.1",
        "tensorboard==2.18.0",
        "tensorboard-data-server==0.7.2",
        "tensorboardX==2.6.2.2",
        "pod5==0.3.23",
        "lib-pod5==0.3.23",
        "requests==2.32.3",
        "packaging==24.2",
        "ninja==1.11.1.3",
        "psutil==6.1.1",
        "einops==0.8.0",
        "tabulate==0.9.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "coral = coral.main:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    include_package_data=True,
)
