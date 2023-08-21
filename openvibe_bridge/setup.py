from setuptools import find_packages, setup

package_name = "openvibe_bridge"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Maximilian Stölzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="Receive data from OpenViBE sent over a TCP socket and publish it as ROS2 messages.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "stimulation_receiver_node = openvibe_bridge.stimulation_receiver_node:main"
        ],
    },
)
