from setuptools import find_packages, setup

package_name = "hsa_trajectory_planner"

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
    maintainer="Maximilian Stolzle",
    maintainer_email="maximilian@stoelzle.ch",
    description="Plan setpoint trajectories for the HSA end-effector.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "planar_bending_trajectory_node = hsa_trajectory_planner.planar_bending_trajectory_node:main"
        ],
    },
)
