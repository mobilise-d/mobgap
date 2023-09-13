import json

import pandas as pd

from gaitlink import PACKAGE_ROOT
from gaitlink.consts import GRAV
from gaitlink.data import LabExampleDataset
from gaitlink.gsd._gsd_iluz import GsdIluz

test_trial = LabExampleDataset(reference_system="INDIP").get_subset(
    cohort="MS", participant_id="001", test="Test5", trial="Trial2"
)

out = GsdIluz().detect(test_trial.data["LowerBack"], sampling_rate_hz=test_trial.sampling_rate_hz)
gsd_list = out.gsd_list_

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(test_trial.data["LowerBack"]["acc_x"].to_numpy(), label="acc_x")

print_label = True
for wb in test_trial.reference_parameters_["wb"]:
    ax.axvspan(
        wb["Start"] * 100, wb["End"] * 100, color="green", alpha=0.2, ymax=1, label="INDIP" if print_label else None
    )
    print_label = False

with (PACKAGE_ROOT.parent / "example_data/original_results/gsd_iluz/lab/MS/001/GSDA_Output.json").open() as f:
    orig_algo_results = json.load(f)["GSDA_Output"]["TimeMeasure1"]["Test5"]["Trial2"]["SU"]["LowerBack"]["GSD"]

if not isinstance(orig_algo_results, list):
    orig_algo_results = [orig_algo_results]
orig_algo_results = pd.DataFrame.from_records(orig_algo_results)

print_label = True
for gsd in orig_algo_results.itertuples(index=False):
    ax.axvspan(
        gsd.GaitSequence_Start * 100,
        gsd.GaitSequence_End * 100,
        color="blue",
        alpha=0.2,
        ymax=0.9,
        label="matlab" if print_label else None,
    )
    print_label = False
print(orig_algo_results)

print_label = True
for gsd in gsd_list.itertuples(index=False):
    ax.axvspan(gsd.start, gsd.end, color="red", alpha=0.2, ymax=0.8, label="python" if print_label else None)
    print_label = False


ax.legend()
fig.show()

print(gsd_list)
