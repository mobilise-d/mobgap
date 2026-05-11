r"""
.. _wtd_megaritis_signal:

WTD Megaritis Signal
====================

This example shows how to use the Wtd_Megaritis_signal algorithm for wear-time
detection using gyroscope rotational patterns and accelerometer variability.

We start by defining some helpers for plotting and loading the data.
"""

# Plotting Helper
# ---------------
# We define a helper function to plot wear-time detection results.
import matplotlib.pyplot as plt


def plot_wtd_outputs(data, **kwargs):
    """Plot IMU data with wear-time periods overlaid."""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot accelerometer signal
    ax.plot(data["acc_is"].to_numpy(), label="acc_is", alpha=0.7)

    color_cycle = iter(plt.rcParams["axes.prop_cycle"])

    y_max = 1.1
    plot_props = [
        {
            "data": v,
            "label": k,
            "alpha": 0.2,
            "ymax": (y_max := y_max - 0.1),
            "color": next(color_cycle)["color"],
        }
        for k, v in kwargs.items()
    ]

    for props in plot_props:
        for wt_period in props.pop("data").itertuples(index=False):
            ax.axvspan(
                wt_period.start, wt_period.end, label=props.pop("label", None), **props
            )

    ax.set_xlabel("Sample")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.legend()
    return fig, ax


# %%
# Loading example data
# --------------------
# .. note :: More info about data loading can be found in the data loading example.
#
# We load example data from the lab dataset.
from mobgap.data import LabExampleDataset
from mobgap.utils.conversions import to_body_frame

example_data = LabExampleDataset(
    reference_system="INDIP", reference_para_level="wb"
)

# %%
# Performance on a lab trial
# --------------------------
# Below we apply the algorithm to a lab trial containing activities of daily living.
# Note: This is controlled lab data where the sensor was worn for the entire recording,
# so we expect the algorithm to detect close to 100% wear-time.
from mobgap.weartime import Wtd_Megaritis_signal

single_test = example_data.get_subset(
    cohort="MS", participant_id="001", test="Test11", trial="Trial1"
)

imu_data = to_body_frame(single_test.data_ss)
sampling_rate_hz = single_test.sampling_rate_hz

# Apply algorithm
wtd_output = Wtd_Megaritis_signal().detect(
    imu_data, sampling_rate_hz=sampling_rate_hz
)

print("Detected Wear-Time Periods:\n")
print(wtd_output.weartime_list_)
print(f"\nTotal wear-time samples: {wtd_output.total_weartime_samples_}")
print(f"Performance: {wtd_output.perf_}")

# %%
# Plotting the results
# --------------------
# We can visualize the detected wear-time periods overlaid on the IMU signal.

fig, ax = plot_wtd_outputs(
    imu_data,
    detected=wtd_output.weartime_list_,
)
plt.tight_layout()
plt.show()