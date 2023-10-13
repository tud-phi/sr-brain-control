from glob import glob
import os
from setuptools import find_packages, setup

package_name = "joylike_operation"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maximilian Stölzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="TeleOperation for robots: Receive data from OpenViBE sent over a TCP socket and publish it as Joy ROS2 messages or read keyboard inputs.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"keyboard_to_joy_node = {package_name}.keyboard_to_joy_node:main",
            f"openvibe_stimulation_to_joy_node = {package_name}.openvibe_stimulation_to_joy_node:main",
        ],
    },
)
