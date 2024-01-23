import dill
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)  # double precision
jax_config.update("jax_platform_name", "cpu")  # use CPU
import jax.numpy as jnp
import matplotlib

matplotlib.use("Qt5Cairo")
from matplotlib import animation
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from hsa_planar_control.analysis.utils import trim_time_series_data

# latex text
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


EXPERIMENT_ID = "20231031_185546"  # experiment id

TRAJECTORY_TYPE = (
    "setpoint_regulation"  # trajectory type. Can be "setpoint_regulation" or "adl"
)
SPEEDUP = 16  # speedup factor for the animation
STEP_SKIP = 28  # step skip for the animation
REL_START_TIME = 0.0  # relative start time of the experiment [s]
PLOT_TYPE = "end_effector_position"  # plot type. Can be "control_input" or "end_effector_position"
if EXPERIMENT_ID == "20231031_185546":
    # setpoint regulation with brain controller
    CONTROLLER_TYPE = "brain"  # "computational", "brain", or "keyboard"
elif EXPERIMENT_ID == "20231031_181745":
    # setpoint regulation with keyboard controller
    CONTROLLER_TYPE = "keyboard"  # "computational", "brain", or "keyboard"
elif EXPERIMENT_ID == "20231030_181558":
    # setpoint regulation with computational controller
    CONTROLLER_TYPE = "computational"  # "computational", "brain", or "keyboard"
elif EXPERIMENT_ID == "20231031_203004":
    # hairspray interaction with brain controller
    CONTROLLER_TYPE = "brain"  # "computational", "brain", or "keyboard"
    REL_START_TIME = 379.0
    DURATION = 111.0
    TRAJECTORY_TYPE = "adl"  # trajectory type
else:
    CONTROLLER_TYPE = "brain"

if TRAJECTORY_TYPE == "setpoint_regulation":
    DURATION = 543.0  # duration of the experiment [s]

figsize = (5.0, 2.5)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
linewidth_dashed = 2.7
linewidth_dotted = 2.7
linewidth_solid = 2.0
dots = (1.2, 0.8)
dashes = (2.5, 1.2)


def main():
    experiment_folder = Path("data") / "experiments" / EXPERIMENT_ID
    with open(
        str(experiment_folder / ("rosbag2_" + EXPERIMENT_ID + "_0.dill")), "rb"
    ) as f:
        data_ts = dill.load(f)

    # absolute start time of the experiment
    if TRAJECTORY_TYPE == "setpoint_regulation":
        start_time = data_ts["ts_chiee_des"][0] + REL_START_TIME
    else:
        start_time = data_ts["controller_info_ts"]["ts"][0] + REL_START_TIME
    # trim the dictionary with the time series data
    data_ts = trim_time_series_data(data_ts, start_time, DURATION)
    ci_ts = data_ts["controller_info_ts"]

    print("Available time series data:\n", data_ts.keys())
    print("Available controller info", ci_ts.keys())

    t_ts = ci_ts["ts"]
    # frame rate
    frame_rate = SPEEDUP / STEP_SKIP * (1 / (t_ts[1:] - t_ts[:-1]).mean().item())
    print("Frame rate:", frame_rate)
    pbar = tqdm(total=t_ts.shape[0])

    def animate_end_effector_position():
        fig = plt.figure(figsize=figsize, num="End-effector position", dpi=200)
        ax = plt.gca()
        # plot the desired end-effector position
        pee_des_lines = []
        if "ts_chiee_des" in data_ts.keys() and "chiee_des_ts" in data_ts.keys():
            (line,) = ax.step(
                [],
                [],
                where="post",
                color=colors[0],
                linestyle="--",
                dashes=dashes,
                linewidth=linewidth_dashed,
                label=r"$x^\mathrm{d}$",
            )
            pee_des_lines.append(line)
            (line,) = ax.step(
                [],
                [],
                where="post",
                color=colors[1],
                linestyle="--",
                dashes=dashes,
                linewidth=linewidth_dashed,
                label=r"$y^\mathrm{d}$",
            )
            pee_des_lines.append(line)
        pee_at_lines = []
        if CONTROLLER_TYPE != "computational":
            # plot the attractor position
            (line,) = ax.plot(
                [],
                [],
                color=colors[0],
                linestyle=":",
                linewidth=linewidth_dotted,
                dashes=dots,
                label=r"$x^\mathrm{at}$",
            )
            pee_at_lines.append(line)
            (line,) = ax.plot(
                [],
                [],
                color=colors[1],
                linestyle=":",
                linewidth=linewidth_dotted,
                dashes=dots,
                label=r"$y^\mathrm{at}$",
            )
            pee_at_lines.append(line)
        pee_lines = []
        (line,) = ax.plot(
            [],
            [],
            color=colors[0],
            linewidth=linewidth_solid,
            label=r"$x$",
        )
        pee_lines.append(line)
        (line,) = ax.plot(
            [],
            [],
            color=colors[1],
            linewidth=linewidth_solid,
            label=r"$y$",
        )
        pee_lines.append(line)
        ax.set_xlabel(r"Time $t$ [s]")
        plt.ylabel(r"End-effector position [mm]")
        ax.set_xlim(t_ts[0], t_ts[-1])
        ax.legend()
        ax.grid(True)
        plt.box(True)
        plt.tight_layout()

        ymin = jnp.min(jnp.concatenate([ci_ts["chiee"][..., :2], ci_ts["chiee_des"][..., :2]], axis=0))
        ymax = jnp.max(jnp.concatenate([ci_ts["chiee"][..., :2], ci_ts["chiee_des"][..., :2]], axis=0))
        ax.set_ylim(
            ymin * 1e3 - 5,
            ymax * 1e3 + 5,
        )


        def animate(time_idx):
            for _i, _line in enumerate(pee_lines):
                _line.set_data(
                    ci_ts["ts"][:time_idx],
                    ci_ts["chiee"][:time_idx, _i] * 1e3,
                )
            for _i, _line in enumerate(pee_at_lines):
                _line.set_data(
                    ci_ts["ts"][:time_idx],
                    ci_ts["chiee_des"][:time_idx, _i] * 1e3,
                )

            # plot the reference trajectory
            # define the time selector
            chiee_des_selector = data_ts["ts_chiee_des"][:time_idx] <= t_ts[time_idx]
            if jnp.sum(chiee_des_selector) > 0:
                # currently active reference
                chiee_des_current = data_ts["chiee_des_ts"][chiee_des_selector][-1]
                # add the currently active reference
                ts_chiee_des = jnp.concatenate(
                    [
                        data_ts["ts_chiee_des"][chiee_des_selector],
                        jnp.expand_dims(t_ts[time_idx], axis=0),
                    ],
                    axis=0,
                )
                chiee_des_ts = jnp.concatenate(
                    [
                        data_ts["chiee_des_ts"][chiee_des_selector],
                        jnp.expand_dims(chiee_des_current, axis=0),
                    ],
                    axis=0,
                )
                for _i, _line in enumerate(pee_des_lines):
                    _line.set_data(
                        ts_chiee_des,
                        chiee_des_ts[..., _i] * 1e3,
                    )

            lines = pee_lines + pee_des_lines + pee_at_lines
            pbar.update(STEP_SKIP)
            return lines

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=jnp.arange(t_ts.shape[0], step=STEP_SKIP),
            interval=1000 / frame_rate,
            blit=True,
        )

        movie_writer = animation.FFMpegWriter(fps=frame_rate)
        ani.save(
            str(experiment_folder / f"{EXPERIMENT_ID}_pee_{SPEEDUP:.0f}x.mp4"),
            writer=movie_writer,
        )

        plt.show()
        pbar.close()

    def animate_control_input_trajectory():
        fig = plt.figure(figsize=figsize, num="Control input", dpi=200)
        ax = plt.gca()
        phi_sat_lines = []
        for i in range(2):
            (line,) = ax.plot(
                [],
                [],
                color=colors[i],
                linewidth=linewidth_solid,
                label=r"$\phi_" + str(i + 1) + "$",
            )
            phi_sat_lines.append(line)
        ax.set_xlabel(r"Time $t$ [s]")
        ax.set_ylabel(r"Control input $\phi$ [rad]")
        ax.set_xlim(t_ts[0], t_ts[-1])
        ax.set_ylim(
            jnp.min(ci_ts["phi_des_sat"]) - jnp.pi / 16,
            jnp.max(ci_ts["phi_des_sat"]) + jnp.pi / 16,
        )
        ax.legend()
        ax.grid(True)
        plt.box(True)
        plt.tight_layout()

        def animate(time_idx):
            for _i, _line in enumerate(phi_sat_lines):
                _line.set_data(
                    ci_ts["ts"][:time_idx],
                    ci_ts["phi_des_sat"][:time_idx, _i],
                )

            lines = phi_sat_lines
            pbar.update(STEP_SKIP)
            return lines

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=jnp.arange(t_ts.shape[0], step=STEP_SKIP),
            interval=1000 / frame_rate,
            blit=True,
        )

        movie_writer = animation.FFMpegWriter(fps=frame_rate)
        ani.save(
            str(experiment_folder / f"{EXPERIMENT_ID}_phi_{SPEEDUP:.0f}x.mp4"),
            writer=movie_writer,
        )

        plt.show()
        pbar.close()

    if PLOT_TYPE == "end_effector_position":
        animate_end_effector_position()
    elif PLOT_TYPE == "control_input":
        animate_control_input_trajectory()
    else:
        raise ValueError("Unknown plot type.")


if __name__ == "__main__":
    main()
