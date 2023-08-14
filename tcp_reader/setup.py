from setuptools import find_packages, setup

package_name = "tcp_reader"

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
    description="Read data from a TCP socket and publish it as ROS2 messages.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["socket_reader_node = tcp_reader.socket_reader_node:main"],
    },
)
