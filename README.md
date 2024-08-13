# Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control

This repository contains the code for plotting the experimental data of the paper **Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control**. Furthermore, it contains the OpenVibe `.xml` files used for data processing of the EEG signals, training the classifiers, and inference of the EEG pipeline online.

[![Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control - Video](https://img.youtube.com/vi/wZTOxBPZmPc/0.jpg)](https://www.youtube.com/watch?v=wZTOxBPZmPc)

**Abstract:** Integrating Brain-Machine Interfaces into non-clinical applications like robot motion control remains difficult - despite remarkable advancements in clinical settings. Specifically, EEG-based motor imagery systems are still error-prone, posing safety risks when rigid robots operate near humans. This work presents an alternative pathway towards safe and effective operation by combining wearable EEG with physically embodied safety in soft robots. We introduce and test a pipeline that allows a user to move a soft robot's end effector in real time via brain waves that are measured by as few as three EEG channels. A robust motor imagery algorithm interprets the user's intentions to move the position of a virtual attractor to which the end effector is attracted, thanks to a new Cartesian impedance controller. We specifically focus here on planar soft robot-based architected metamaterials, which require the development of a novel control architecture to deal with the peculiar nonlinearities - e.g., non-affinity in control. We preliminarily but quantitatively evaluate the approach on the task of setpoint regulation. We observe that the user reaches the proximity of the setpoint in 66% of steps and that for successful steps, the average response time is 21.5s. We also demonstrate the execution of simple real-world tasks involving interaction with the environment, which would be extremely hard to perform if it were not for the robot's softness.

![Video of activity of daily living controlled with motor imagery](assets/20231031_203004_backview_4x.gif)

## Citation

This repository is part of the publication **Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control** which received the Best Paper Award (1st place) at the 
_7th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2024)_. The paper is available online on [IEEEXplore](10.1109/RoboSoft60065.2024.10522005) or on [arXiv](https://arxiv.org/abs/2401.13441).

Please use the following citation if you use our method in your (scientific) work:

```bibtex
@inproceedings{stolzle2024guiding,
  title={Guiding Soft Robots with Motor-Imagery Brain Signals and Impedance Control},
  author={St{\"o}lzle, Maximilian and Baberwal, Sonal Santosh and Rus, Daniela and Coyle, Shirley and Della Santina, Cosimo},
  booktitle={2024 IEEE 7th International Conference on Soft Robotics (RoboSoft)},
  pages={276--283},
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

The `openvibe` folder contains OpenVibe `.xml` files use for data processing of the EEG signals, training the classifiers, and inference of the EEG pipeline online.

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
