from setuptools import setup, find_packages
import os

install_requires = [
    "gymnasium==0.28.1",
    "pybullet==3.2.5",
    "pynput==1.7.6",
    "pygame==2.3.0",
    "scipy==1.9.1",
    "swig==4.1.1",
    "tensorboard==2.12.2",
    "tensorboard-data-server==0.7.0",
    "tensorboard-plugin-wit==1.8.1",
    "tensorboardX==2.6",
    "tqdm==4.65.0",
    "typing_extensions==4.5.0",
    "vcstool==0.3.0",
    "openpyxl==3.1.2",
    "et-xmlfile==1.1.0",
    "scikit-learn==1.3.0",
    "optuna==3.2.0",
    "graphviz==0.20.1",
    "prompt_toolkit==3.0.39",
    "wcwidth==0.2.6",
    "pytest==7.4.0",
    "stable_baselines3==2.0.0a9",
]

if os.name == "nt":  # For Windows
    print("Detected Windows OS")
    install_requires.extend(
        [
            "torch==2.0.1+cu118",
            "torchaudio==2.0.2+cu118",
            "torchvision==0.15.2+cu118",
        ]
    )
elif os.name == "posix":  # POSIX compliant (e.g., UNIX, LINUX, MAC OS X)
    if os.uname().sysname == "Darwin":  # For MacOS
        install_requires.extend(
            [
                "torch==2.0.1",
                "torchaudio==2.0.2",
                "torchvision==0.15.2",
            ]
        )
    else:
        install_requires.extend(
            [
                "torch==2.0.1+cu118",
                "torchaudio==2.0.2+cu118",
                "torchvision==0.15.2+cu118",
            ]
        )
    # elif for other POSIX compliant systems if needed

setup(
    name="loyalwingmen",
    version="1.0.0",
    author="Davi",
    author_email="your@email.address",
    description="loyalwingmen simulation based on pybullet engine and gymnasium environment",
    long_description=open("README.markdown").read(),
    long_description_content_type="text/markdown",
    keywords="example, setuptools",
    license="BSD 3-Clause License",
    python_requires="==3.8.17",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    zip_safe=True,
    include_package_data=True,
    install_requires=install_requires,
    package_data={
        "example": ["data/schema.json", "*.txt"],
        "*": ["README.markdown"],
    },
)
