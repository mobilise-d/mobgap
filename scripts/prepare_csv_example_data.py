import mobgap
from mobgap.data import LabExampleDataset
from mobgap.consts import GRAV_MS2

BASE_PATH = mobgap.PACKAGE_ROOT.parent / "example_data"/ "data_csv"

for d in LabExampleDataset():
    # We store the data in a csv file
    sub_path = BASE_PATH / d.group_label.cohort / d.group_label.participant_id
    sub_path.mkdir(parents=True, exist_ok=True)
    file_name = sub_path / f"{d.group_label.time_measure}_{d.group_label.test}_{d.group_label.trial}.csv"
    data = d.data_ss.reset_index(drop=True).rename_axis("samples")
    # We change the units of the acc data, just so that we can show the back conversion in the example
    data[["acc_x", "acc_y", "acc_z"]] = data[["acc_x", "acc_y", "acc_z"]] / GRAV_MS2
    data.to_csv(file_name)
