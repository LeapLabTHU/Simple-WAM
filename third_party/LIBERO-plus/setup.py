# read the contents of your README file
from os import path

from setuptools import find_packages, setup

setup(
    name="libero",
    packages=[package for package in find_packages() if package.startswith("libero")],
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3",
    description="LIBERO-plus: In-Depth Robustness Analysis For Vision-Language-Action Models",
    author="Anonymous",
    author_email="Anonymous",
    version="0.1.0",
    long_description="LIBERO-plus",
    long_description_content_type="text/markdown",
    entry_points={
        "console_scripts": [
            "lifelong.main=libero.lifelong.main:main",
            "lifelong.eval=libero.lifelong.evaluate:main",
            "libero.config_copy=scripts.config_copy:main",
            "libero.create_template=scripts.create_template:main",
        ]
    },
)
