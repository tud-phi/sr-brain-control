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
        (os.path.join("share", package_name, "assets"), glob("assets/*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Sonal Baberwal, Maximilian Stolzle",
    author_email="sonal.baberwal2@mail.dcu.ie, maximilian@stoelzle.ch",
    maintainer="Sonal Baberwal, Maximilian Stolzle",
    maintainer_email="sonal.baberwal2@mail.dcu.ie, maximilian@stoelzle.ch",
    description="Brain control of (planar) HSA robots.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "keyboard_to_joy_node = hsa_brain_control.keyboard_to_joy_node:main",
            "planar_hsa_bending_brain_control_node = hsa_brain_control.planar_hsa_bending_brain_control_node:main",
            "planar_hsa_cartesian_brain_control_node = hsa_brain_control.planar_hsa_cartesian_brain_control_node:main"
        ],
    },
)
