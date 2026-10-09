import numpy as np
import pandas as pd


def _to_stride_list_per_foot(ic_lr_list: pd.DataFrame) -> pd.DataFrame:
    return (
        ic_lr_list[["ic", "lr_label"]]
        .rename(columns={"ic": "start"})
        .assign(end=lambda df_: df_["start"].shift(-1))
        .dropna()
        .astype({"start": "int64", "end": "int64"})
    )


def _unify_stride_list(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype({"start": "int64", "end": "int64", "lr_label": pd.CategoricalDtype(categories=["left", "right"])})[
        ["start", "end", "lr_label"]
    ]
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.rename("s_id", level=-1)
    else:
        df.index.name = "s_id"
    return df


def strides_list_from_ic_lr_list(ic_lr_list: pd.DataFrame) -> pd.DataFrame:
    """Convert an initial contact list with left-right labels to a list of strides.

    Each stride is defined from one initial contact to the next initial contact of the same foot.
    This means no correction is applied and some strides might be relatively long, if ICs are not detected correctly
    or there are breaks in the walking pattern.

    Parameters
    ----------
    ic_lr_list
        A DataFrame with the columns "ic" and "lr_label".

    Returns
    -------
    stride_list
        A DataFrame with the columns "start", "end", and "lr_label".
    """
    if ic_lr_list.empty:
        return pd.DataFrame(columns=["start", "end", "lr_label"], index=ic_lr_list.index).pipe(_unify_stride_list)

    labels = ic_lr_list["lr_label"]
    if (
        type(ic_lr_list) is pd.DataFrame
        and not ic_lr_list.attrs
        and ic_lr_list["ic"].dtype == np.dtype("int64")
        and isinstance(labels.dtype, pd.CategoricalDtype)
        and not labels.cat.ordered
        and list(labels.cat.categories) == ["left", "right"]
        and not labels.isna().any()
    ):
        # The pipeline's standardized contacts need only adjacent positions per foot.
        # Keep both sorts and the category group order to preserve ties exactly.
        ordered = ic_lr_list.sort_values("ic")
        label_codes = ordered["lr_label"].cat.codes.to_numpy()
        foot_positions = [np.flatnonzero(label_codes == code) for code in (0, 1)]
        start_positions = np.concatenate([positions[:-1] for positions in foot_positions])
        end_positions = np.concatenate([positions[1:] for positions in foot_positions])
        contacts = ordered["ic"].to_numpy()
        stride_labels = (
            ordered["lr_label"].array.take(start_positions).astype(pd.CategoricalDtype(categories=["left", "right"]))
        )
        stride_list = pd.DataFrame(
            {
                "start": contacts[start_positions],
                "end": contacts[end_positions],
                "lr_label": stride_labels,
            },
            index=ordered.index.take(start_positions),
        ).sort_values("start")
        stride_list.columns.name = ic_lr_list.columns.name
        if isinstance(stride_list.index, pd.MultiIndex):
            stride_list.index = stride_list.index.rename("s_id", level=-1)
        else:
            stride_list.index.name = "s_id"
        return stride_list

    # TODO: Warn if strides are fully contained in other strides. This indicates missing ICs.
    return (
        ic_lr_list.sort_values("ic")
        .groupby("lr_label", as_index=False, group_keys=False, observed=True)[["ic", "lr_label"]]
        .apply(_to_stride_list_per_foot)
        .sort_values("start")
        .pipe(_unify_stride_list)
    )
