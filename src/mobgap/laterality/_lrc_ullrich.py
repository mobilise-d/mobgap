import warnings
from collections.abc import Iterable
from functools import cache
from importlib.resources import files
from itertools import chain, repeat
from types import MappingProxyType
from typing import Any, Final, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning, NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted
from tpcp import cf
from tpcp.misc import classproperty, set_defaults
from typing_extensions import Self, TypedDict, Unpack

from mobgap.data_transform import ButterworthFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.laterality.base import BaseLRClassifier, _unify_ic_lr_list_df, base_lrc_docfiller
from mobgap.utils._sklearn_protocol_types import SklearnClassifier, SklearnScaler
from mobgap.utils.dtypes import assert_is_sensor_data


@cache
def _load_model_files(
    file_name: str, expect_warning: bool = False
) -> Union[SklearnClassifier, SklearnScaler, Pipeline]:
    file_path = files("mobgap") / "laterality" / "_ullrich_pretrained_models" / file_name
    if not expect_warning:
        # In case this line throws a `InconsistentVersionWarning` for one of the models, create an issue in Mobgap, so
        # that we can update our models to the newest version of sklearn.
        # You can still use the models in the meantime (they will likely work just fine).
        # In case you are a developer and you are here, because someone created an issue, checkout the section in the
        # developer guide (docs/guides/developer_guide.rst) on how to update the models.
        return joblib.load(file_path)

    with file_path.open("rb") as file, warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        return joblib.load(file)


@base_lrc_docfiller
class LrcUllrich(BaseLRClassifier):
    """Machine-Learning based algorithm for laterality classification of initial contacts.

    This algorithm uses the band-pass filtered vertical ("gyr_is") and anterior-posterior ("gyr_pa") angular velocity.
    For both axis a set of features consisting of the value, the first and second derivative are extracted at the time
    points of the ICs ([1]_).
    This results in a 6-dimensional feature vector for each IC.
    This feature set is normalized using the provided scaler and then classified using the provided model.

    We provide a set of pre-trained models that are based on the MS-Project ([2]_) dataset.
    They all use a Min-Max Scaler in combination with a linear SVC classifier.
    The parameters of the SVC depend on the cohort and were tuned as explained in the paper ([1]_).
    See more on the models in the Notes section.

    Parameters
    ----------
    smoothing_filter
        The bandpass filter used to smooth the data before feature extraction.
    clf_pipe
        A sklearn pipeline used to perform the laterality classification based on the extracted features.
        All pretrained pipelines consist of a ``MinMaxScaler`` and a linear ``SVC`` classifier.

    Attributes
    ----------
    %(ic_lr_list_)s
    feature_matrix_
        The 6-dimensional feature vector, containing the vertical and anterior-posterior filtered angular
        velocities and their first (gradient) and second (curvature) derivatives.
        This might be helpful for debugging or further analysis.
        This feature matrix will be passed into the provided pipeline.

    Other Parameters
    ----------------
    %(other_parameters)s


    Notes
    -----
    Models: All models are trained based on the MsProject dataset. All models specified with the suffix "_old" were
    trained using the original implementation of this algorithm using an old version of the sklearn library.
    We don't recommend using them, unless you want to reproduce old results.

    .. [1] Ullrich M, Küderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left
           and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021,
           available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653
    .. [2] Angelini L, Hodgkinson W, Smith C, Dodd JM, Sharrack B, Mazzà C, Paling D. Wearable sensors can reliably
           quantify gait alterations associated with disability in people with progressive multiple sclerosis in a
           clinical setting. J Neurol. 2020 Oct;267(10):2897-2909. doi: 10.1007/s00415-020-09928-8. Epub 2020 May 28.
           PMID: 32468119; PMCID: PMC7501113.

    """

    smoothing_filter: BaseFilter
    clf_pipe: Pipeline

    feature_matrix_: pd.DataFrame

    _feature_matrix_cols: Final = [
        "filtered__gyr_is",
        "gradient__gyr_is",
        "curvature__gyr_is",
        "filtered__gyr_pa",
        "gradient__gyr_pa",
        "curvature__gyr_pa",
    ]

    class PredefinedParameters:
        """Predefined parameters for the LrdUllrich class."""

        _BW_FILTER = ButterworthFilter(order=4, cutoff_freq_hz=(0.5, 2), filter_type="bandpass")

        class _ModelConfig(TypedDict):
            smoothing_filter: BaseFilter
            clf_pipe: Pipeline

        @classmethod
        def _load_old_model_config(cls, model_name: str) -> _ModelConfig:
            model = _load_model_files(f"old/{model_name}_model.gz", expect_warning=True)
            scaler = _load_model_files(f"old/{model_name}_scaler.gz", expect_warning=True)

            # Note: this clip property was added here to ensure compatibility with sklearn version 0.23.1, which was
            #  used for training the models and storing the corresponding min-max scalers.
            #  Since version 0.24 onwards, this needs to be specified.
            scaler.clip = False

            return MappingProxyType(
                {
                    "smoothing_filter": cls._BW_FILTER,
                    # Note, that we use names for the pipeline steps, that are allow us to identify, that these are the
                    # old pre-trained models.
                    "clf_pipe": Pipeline([("scaler_old", scaler), ("clf_old", model)]),
                }
            )

        @classmethod
        def _load_model_config(cls, model_name: str) -> _ModelConfig:
            # We reinitialze the pipeline to avoid issues with changes of the pipeline class between sklearn versions.
            pipe = Pipeline(_load_model_files(f"{model_name}_model.gz", expect_warning=False).steps)

            return MappingProxyType(
                {
                    "smoothing_filter": cls._BW_FILTER,
                    "clf_pipe": pipe,
                }
            )

        @classproperty
        def msproject_all_old(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_old_model_config("msproject_all")

        @classproperty
        def msproject_ms_old(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_old_model_config("msproject_ms")

        @classproperty
        def msproject_all(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_model_config("msproject_all")

        @classproperty
        def untrained_svc(cls) -> _ModelConfig:  # noqa: N805
            return {
                "smoothing_filter": cls._BW_FILTER,
                "clf_pipe": Pipeline([("scaler", MinMaxScaler()), ("clf", SVC(kernel="linear"))]),
            }

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.msproject_all.items()})
    def __init__(
        self,
        smoothing_filter: BaseFilter,
        clf_pipe: Pipeline,
    ) -> None:
        self.smoothing_filter = smoothing_filter
        self.clf_pipe = clf_pipe

    @base_lrc_docfiller
    def predict(
        self,
        data: pd.DataFrame,
        ic_list: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """
        %(predict_short)s.

        Parameters
        ----------
        %(predict_para)s
        kwargs
            Additional kwargs that are passed to the ``self.clf_pipe.predict`` method.

        %(predict_return)s
        """
        self.data = data
        self.ic_list = ic_list
        self.sampling_rate_hz = sampling_rate_hz

        assert_is_sensor_data(data, frame="body")

        if data.empty or ic_list.empty:
            self.ic_lr_list_ = (
                pd.DataFrame(columns=["ic", "lr_label"], index=ic_list.index).dropna().pipe(_unify_ic_lr_list_df)
            )
            self.feature_matrix_ = pd.DataFrame(columns=self._feature_matrix_cols, index=ic_list.index).dropna()
            return self

        # create a copy of ic_list, otherwise, they will get modified when adding the predicted labels
        # We also remove the "lr_label" column, if it exists, to avoid conflicts
        ic_list = ic_list.copy().drop(columns="lr_label", errors="ignore")

        try:
            check_is_fitted(self.clf_pipe)
        except NotFittedError as e:
            raise RuntimeError("Model is not fitted. Call self_optimize before calling detect.") from e

        feature_matrix = self.extract_features(self.data, ic_list, self.sampling_rate_hz)
        self.feature_matrix_ = feature_matrix

        # The old models were trained without feature names, however, when we retrain the models, we do it with
        # feature names.
        # Once, we retrained all models, we can remove this distinction and assume we are working with dataframes
        # all the time.
        # Hence, we need to separate here:
        if (scaler := self.clf_pipe.named_steps.get("scaler_old")) is not None and getattr(
            scaler, "feature_names_in_", None
        ) is None:
            feature_matrix = feature_matrix.to_numpy()

        # We use sklearn metadata routing to determine the valid kwargs for the predict method.
        # This way, we avoid issues when invalid kwargs are passed to the predict method.
        # TODO: Test this with proper metdata routing active
        potential_meta_data_routed = chain(
            *(
                v.values()
                for v in self.clf_pipe.get_metadata_routing().route_params(caller="predict", params=kwargs).values()
            )
        )
        valid_predict_kwargs = {k: v for d in potential_meta_data_routed for k, v in d.items()}

        ic_list["lr_label"] = self.clf_pipe.predict(feature_matrix, **valid_predict_kwargs)
        ic_list = ic_list.replace({"lr_label": {0: "left", 1: "right"}})

        self.ic_lr_list_ = _unify_ic_lr_list_df(ic_list)

        return self

    @base_lrc_docfiller
    def self_optimize(
        self,
        data_sequences: Iterable[pd.DataFrame],
        ic_list_per_sequence: Iterable[pd.DataFrame],
        ref_ic_lr_list_per_sequence: Iterable[pd.DataFrame],
        *,
        sampling_rate_hz: Union[float, Iterable[float]],
        **kwargs: Unpack[dict[str, Any]],
    ) -> Self:
        """Retrain the classifier pipeline using the provided data.

        .. note:: We only support a full re-fit of the model and the scaler.
           Therefore, you have to pass an untrained instance to the algorithm instance before calling ``self_optimize.``

        Parameters
        ----------
        %(self_optimize_paras)s
        kwargs
            Additional keyword arguments that are passed to the ``self.clf_pipe.fit``

        %(self_optimize_return)s
        """
        try:
            check_is_fitted(self.clf_pipe)
        except NotFittedError:
            pass
        else:
            raise RuntimeError(
                "Pipeline is already fitted. "
                "Initialize the algorithm with a untrained classifier object. "
                "If you want to use the same classifier and scaler as for the pre-trained MS-Project pipelines, you"
                "can use the `LrcUllrich(**LrcUllrich.PredefinedParameters.untrained_svc)` class to load them."
            )

        if isinstance(sampling_rate_hz, float):
            sampling_rate_hz = repeat(sampling_rate_hz)

        features = [
            self.extract_features(dp, ic, sr)
            for dp, ic, sr in zip(data_sequences, ic_list_per_sequence, sampling_rate_hz)
        ]
        all_features = pd.concat(features, axis=0, ignore_index=True) if len(features) > 1 else features[0]
        # Concatenate the labels if there is more than one GS
        label_list = [ic_lr_list["lr_label"] for ic_lr_list in ref_ic_lr_list_per_sequence]
        all_labels = pd.concat(label_list, axis=0, ignore_index=True) if len(label_list) > 1 else label_list[0]

        # We use sklearn metadata routing to determine the valid kwargs for the predict method.
        # This way, we avoid issues when invalid kwargs are passed to the predict method.
        # TODO: Test this with proper metdata routing active
        potential_meta_data_routed = chain(
            *(
                v.values()
                for v in self.clf_pipe.get_metadata_routing().route_params(caller="fit", params=kwargs).values()
            )
        )
        valid_predict_kwargs = {k: v for d in potential_meta_data_routed for k, v in d.items()}

        self.clf_pipe.fit(all_features, all_labels, **valid_predict_kwargs)

        return self

    def extract_features(
        self,
        data: pd.DataFrame,
        ics: pd.DataFrame,
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
        """Extract features from the provided gait data and initial contact list.

        Here, the feature set ic composed of the first (gradient) and second derivatives (curvature) of the filtered
        signals at the time points of the  ICs.
        Consequently, for a dataset containing a total of  ICs, this results in a feature matrix.
        To ensure uniformity, the feature set is normalized.

        .. note:: Usually you don't want to call this method directly. Instead, use the `detect` method,
            which calls this method internally.

        Parameters
        ----------
        data
            The gait data.
        ics
            The initial contact list.
        sampling_rate_hz
            The sampling rate in Hz.

        Returns
        -------
        feature_df
            The DataFrame containing the extracted features.
        """
        assert_is_sensor_data(data, frame="body")

        gyr = data[["gyr_is", "gyr_pa"]]
        gyr_filtered = self.smoothing_filter.clone().filter(gyr, sampling_rate_hz=sampling_rate_hz).filtered_data_
        # We use numpy gradient instead of diff, as it preserves the shape of the input and hence, can handle ICs that
        # are close to the beginning or end of the data.
        gyr_gradient = np.gradient(gyr_filtered, axis=0)
        curvature = np.gradient(gyr_gradient, axis=0)
        gyr_gradient = pd.DataFrame(gyr_gradient, columns=["gyr_is", "gyr_pa"], copy=False, index=gyr.index)
        curvature = pd.DataFrame(curvature, columns=["gyr_is", "gyr_pa"], copy=False, index=gyr.index)

        signal_paras = pd.concat({"filtered": gyr_filtered, "gradient": gyr_gradient, "curvature": curvature}, axis=1)
        signal_paras.columns = ["__".join(c) for c in signal_paras.columns]

        # Extract features corresponding to the adjusted ics values
        feature_df = signal_paras.iloc[ics["ic"].to_numpy()]
        feature_df.index = ics.index

        # Reorder the columns, to be consistent with the original implementation of [1].
        # Note: this order is necessary for the scaler to be applied correctly.
        feature_df = feature_df[self._feature_matrix_cols]

        return feature_df
