import dill
from functools import partial
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import matplotlib

matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from hsa_planar_control.analysis.utils import trim_time_series_data

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Romand"],
    }
)

EXPERIMENT_ID = "20231031_185546"  # experiment id

DURATION = 543.0  # duration of the experiment [s]
if EXPERIMENT_ID == "20231031_185546":
    # setpoint regulation with brain controller
    CONTROLLER_TYPE = "brain"  # "computational", "brain", or "keyboard"
elif EXPERIMENT_ID == "20231031_181745":
    # setpoint regulation with keyboard controller
    CONTROLLER_TYPE = "keyboard"  # "computational", "brain", or "keyboard"
elif EXPERIMENT_ID == "20231030_181558":
    # setpoint regulation with computational controller
    CONTROLLER_TYPE = "computational"  # "computational", "brain", or "keyboard"
else:
    CONTROLLER_TYPE = "brain"

def main():
    experiment_folder = Path("data") / "experiments" / EXPERIMENT_ID
    with open(
        str(experiment_folder / ("rosbag2_" + EXPERIMENT_ID + "_0.dill")), "rb"
    ) as f:
        data_ts = dill.load(f)

    # absolute start time of the experiment
    start_time = data_ts["ts_chiee_des"][0]
    # trim the dictionary with the time series data
    data_ts = trim_time_series_data(data_ts, start_time, DURATION)
    ci_ts = data_ts["controller_info_ts"]

    print("Available time series data:\n", data_ts.keys())
    print("Available controller info", ci_ts.keys())

    figsize = (4.5, 3.0)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linewidth_dashed = 2.7
    linewidth_dotted = 2.7
    linewidth_solid = 2.0
    dots = (1.2, 0.8)
    dashes = (2.5, 1.2)

    plt.figure(figsize=figsize, num="End-effector position")
    ax = plt.gca()
    # plot the desired end-effector position
    ax.step(
        data_ts["ts_chiee_des"],
        data_ts["chiee_des_ts"][:, 0] * 1e3,
        where="post",
        color=colors[0],
        linestyle="--",
        dashes=dashes,
        linewidth=linewidth_dashed,
        label=r"$x^\mathrm{d}$",
    )
    ax.step(
        data_ts["ts_chiee_des"],
        data_ts["chiee_des_ts"][:, 1] * 1e3,
        where="post",
        color=colors[1],
        linestyle="--",
        dashes=dashes,
        linewidth=linewidth_dashed,
        label=r"$y^\mathrm{d}$",
    )
    if CONTROLLER_TYPE != "computational":
        # plot the attractor position
        ax.plot(
            ci_ts["ts"],
            ci_ts["chiee_des"][:, 0] * 1e3,
            color=colors[0],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dots,
            label=r"$x^\mathrm{at}$",
        )
        ax.plot(
            ci_ts["ts"],
            ci_ts["chiee_des"][:, 1] * 1e3,
            color=colors[1],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dots,
            label=r"$y^\mathrm{at}$",
        )
    ax.plot(
        ci_ts["ts"],
        ci_ts["chiee"][:, 0] * 1e3,
        color=colors[0],
        linewidth=linewidth_solid,
        label=r"$x$",
    )
    ax.plot(
        ci_ts["ts"],
        ci_ts["chiee"][:, 1] * 1e3,
        color=colors[1],
        linewidth=linewidth_solid,
        label=r"$y$",
    )
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"End-effector position $x$ [mm]")
    if CONTROLLER_TYPE == "computational":
        plt.legend(ncols=2)
    else:
        plt.legend(ncols=3)
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_pee.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_pee.eps"))
    plt.show()

    plt.figure(figsize=figsize, num="End-effector position x-axis")
    ax = plt.gca()
    # plot the desired end-effector position
    ax.step(
        data_ts["ts_chiee_des"],
        data_ts["chiee_des_ts"][:, 0] * 1e3,
        where="post",
        color=colors[0],
        linestyle="--",
        dashes=dashes,
        linewidth=linewidth_dashed,
        label=r"$x^\mathrm{d}$",
    )
    if CONTROLLER_TYPE != "computational":
        # plot the attractor position
        ax.plot(
            ci_ts["ts"],
            ci_ts["chiee_des"][:, 0] * 1e3,
            color=colors[1],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dots,
            label=r"$x^\mathrm{at}$",
        )
    ax.plot(
        ci_ts["ts"],
        ci_ts["chiee"][:, 0] * 1e3,
        color=colors[2],
        linewidth=linewidth_solid,
        label=r"$x$",
    )
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"End-effector x-coordinate $x_1$ [mm]")
    plt.grid(True)
    plt.box(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_pee_x.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_pee_x.eps"))
    plt.show()

    plt.figure(figsize=figsize, num="End-effector position y-axis")
    ax = plt.gca()
    # plot the desired end-effector position
    ax.step(
        data_ts["ts_chiee_des"],
        data_ts["chiee_des_ts"][:, 1] * 1e3,
        where="post",
        color=colors[0],
        linestyle="--",
        dashes=dashes,
        linewidth=linewidth_dashed,
        label=r"$x^\mathrm{d}$",
    )
    if CONTROLLER_TYPE != "computational":
        # plot the attractor position
        ax.plot(
            ci_ts["ts"],
            ci_ts["chiee_des"][:, 1] * 1e3,
            color=colors[1],
            linestyle=":",
            linewidth=linewidth_dotted,
            dashes=dots,
            label=r"$x^\mathrm{at}$",
        )
    ax.plot(
        ci_ts["ts"],
        ci_ts["chiee"][:, 1] * 1e3,
        color=colors[2],
        linewidth=linewidth_solid,
        label=r"$x$",
    )
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"End-effector y-coordinate $x_2$ [mm]")
    plt.grid(True)
    plt.box(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_pee_y.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_pee_y.eps"))
    plt.show()

    plt.figure(figsize=figsize, num="Control input")
    ax = plt.gca()
    ax.plot(
        ci_ts["ts"],
        ci_ts["phi_des_sat"][:, 0],
        color=colors[0],
        linewidth=linewidth_solid,
        label=r"$\phi_1$",
    )
    ax.plot(
        ci_ts["ts"],
        ci_ts["phi_des_sat"][:, 1],
        color=colors[1],
        linewidth=linewidth_solid,
        label=r"$\phi_2$",
    )
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel(r"Control input $\phi$ [rad]")
    plt.legend()
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_phi.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_phi.eps"))
    plt.show()

    fig = plt.figure(figsize=figsize, num="Configuration")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.plot(
        ci_ts["ts"],
        ci_ts["q"][:, 0],
        color=colors[0],
        linewidth=linewidth_solid,
        label=r"$\kappa_\mathrm{be}$",
    )
    ax2.plot(
        ci_ts["ts"],
        ci_ts["q"][:, 1],
        color=colors[1],
        linewidth=linewidth_solid,
        label=r"$\sigma_\mathrm{sh}$",
    )
    ax2.plot(
        ci_ts["ts"],
        ci_ts["q"][:, 2],
        color=colors[2],
        linewidth=linewidth_solid,
        label=r"$\sigma_\mathrm{ax}$",
    )
    ax1.set_xlabel(r"Time $t$ [s]")
    ax1.set_ylabel(r"Bending strain $\kappa_\mathrm{be}$ [rad/m]")
    ax2.set_ylabel(r"Linear strains $\sigma$ [-]")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.grid(True)
    plt.box(True)
    plt.tight_layout()
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_q.pdf"))
    plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_q.eps"))
    plt.show()

    if CONTROLLER_TYPE != "computational":
        fig = plt.figure(figsize=figsize, num="Joy signal")
        ax = plt.gca()
        ax.plot(
            data_ts["ts_joy_signal"],
            data_ts["joy_signal_ts"][:, 0],
            color=colors[0],
            linewidth=linewidth_solid,
            label=r"$u_x$",
        )
        ax.plot(
            data_ts["ts_joy_signal"],
            data_ts["joy_signal_ts"][:, 1],
            color=colors[1],
            linewidth=linewidth_solid,
            label=r"$u_y$",
        )
        ax.set_xlabel(r"Time $t$ [s]")
        ax.set_ylabel(r"Joy signal $u$ [-]")
        ax.legend(loc="upper left")
        plt.grid(True)
        plt.box(True)
        plt.tight_layout()
        plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_joy_signal.pdf"))
        plt.savefig(str(experiment_folder / f"{EXPERIMENT_ID}_joy_signal.eps"))
        plt.show()


if __name__ == "__main__":
    main()
