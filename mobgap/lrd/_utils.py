from gaitlink.pipeline import GsIterator


def extract_ref_data(datapoint):
    """
    Extracts reference data from a given datapoint.

    This function extracts the IMU data from the "LowerBack" of the datapoint,
    and the reference walking bouts and initial contacts from the reference parameters of the datapoint.
    It then iterates over the length of the reference walking bouts, and for each bout, it appends the corresponding IMU data, initial contacts, and labels to their respective lists.

    Parameters
    ----------
    datapoint : (object)
        The datapoint from which to extract the reference data.

    Returns
    -------
    tuple: A tuple containing three lists:
           - data_list: A list of pandas DataFrames, each containing the IMU data for a walking bout.
           - ic_list: A list of pandas DataFrames, each containing the initial contacts for a walking bout, zero-indexed relative to the start of the walking bout.
           - label_list: A list of pandas DataFrames, each containing the labels for a walking bout.
    """
    imu_data = datapoint.data["LowerBack"]
    ref = datapoint.reference_parameters_relative_to_wb_

    gs_iterator = GsIterator()

    data_list, ic_list, label_list = zip(
        *[
            (
                data_per_wb.reset_index(drop=True),
                ref.ic_list.loc[ref.ic_list.index.get_level_values("wb_id") == wb.wb_id, ["ic"]].reset_index(drop=True),
                ref.ic_list.loc[ref.ic_list.index.get_level_values("wb_id") == wb.wb_id, ["lr_label"]].reset_index(
                    drop=True
                ),
            )
            for (wb, data_per_wb), _ in gs_iterator.iterate(imu_data, ref.wb_list)
        ]
    )

    return list(data_list), list(ic_list), list(label_list)
