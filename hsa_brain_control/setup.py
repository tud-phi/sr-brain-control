from glob import glob
import os
from setuptools import find_packages, setup

package_name = "hsa_brain_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (os.path.join("share", package_name), glob("launch/*.py")),
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Maximilian Stolzle",
    author_email="maximilian@stoelzle.ch",
    maintainer="Sonal Baberwal, Maximilian Stolzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="Brain control of (planar) HSA robots.",
    license="MIT",
    tests_require=["numpy", "pytest"],
    entry_points={
        "console_scripts": [
            "planar_hsa_bending_brain_control_node = hsa_brain_control.planar_hsa_bending_brain_control_node:main",
            "planar_hsa_cartesian_brain_control_node = hsa_brain_control.planar_hsa_cartesian_brain_control_node:main"
        ],
    },
)
