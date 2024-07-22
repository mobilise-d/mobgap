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

    # TODO: Warn if strides are fully contained in other strides. This indicates missing ICs.
    return (
        ic_lr_list.sort_values("ic")
        .groupby("lr_label", as_index=False, group_keys=False, observed=True)[["ic", "lr_label"]]
        .apply(_to_stride_list_per_foot)
        .sort_values("start")
        .pipe(_unify_stride_list)
    )
