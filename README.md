# Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control

This repository contains the code for plotting the experimental data of the paper **Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control**.

## Citation

This repository is part of the publication **Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control** presented at the 
_7th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2024)_.

Please use the following citation if you use our method in your (scientific) work:

```bibtex
@inproceedings{stolzle2024guiding,
  title={Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control},
  author={St{\"o}lzle, Maximilian and Baberwal, Sonal and Rus, Daniela and Coyle, Shirley and Della Santina, Cosimo},
  booktitle={2024 IEEE 7th International Conference on Soft Robotics (RoboSoft)},
  year={2024},
  organization={IEEE}
}
```

## Installation

All necessary dependencies can be installed using `pip`:

```bash
pip install -r requirements.txt
```

If you encounter any problems with the JAX installation, please refer to the [JAX installation guide](https://github.com/google/jax#installation).

## Usage

You can plot the experimental data using the `plot_experiment.py` script:

```bash
python scripts/experiment_analysis/plot_experiment.py
```

## See also

You might also be interested in the following repositories:
 - The [`jax-soft-robot-modelling`](https://github.com/tud-phi/jax-soft-robot-modelling) repository contains a JAX implementation 
 of various soft robot models, which can be, for example, used to simulate the robot's forward dynamics.
 - The [`hsa-planar-control`](https://github.com/tud-phi/hsa-planar-control) repository contains JAX and ROS2 implementations
 of model-based control algorithms for planar HSA robots.
 - The [`jax-spcs-kinematics`](https://github.com/tud-phi/jax-spcs-kinematics) repository contains an implementation
 of the Selective Piecewise Constant Strain (SPCS) kinematics in JAX. We have shown in our paper that this kinematic 
model is suitable for representing the shape of HSA rods.
 - The [`HSA-PyElastica`](https://github.com/tud-phi/HSA-PyElastica) repository contains a plugin for PyElastica
for the simulation of HSA robots.
