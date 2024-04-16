from collections.abc import Iterable
from functools import cache
from importlib.resources import files
from itertools import cycle
from typing import Any, Final, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from tpcp import cf
from tpcp.misc import classproperty, set_defaults
from typing_extensions import Self, TypedDict, Unpack

from mobgap.data_transform import ButterworthFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.lrd.base import BaseLRDetector, base_lrd_docfiller
from mobgap.utils._sklearn_protocol_types import SklearnClassifier, SklearnScaler


@cache
def _load_model_files(file_name: str) -> Union[SklearnClassifier, SklearnScaler]:
    file_path = files("mobgap") / "lrd" / "_ullrich_pretrained_models" / file_name
    with file_path.open("rb") as file:
        return joblib.load(file)


# TODO: Instead of having a separate scaler and model, we should use a sklearn pipeline.


class LrdUllrich(BaseLRDetector):
    """Machine-Learning based algorithm for laterality detection of initial contacts.

    This algorithm uses the band-pass filtered vertical ("gyr_x") and anterior-posterior ("gyr_z") angular velocity.
    For both axis a set of features consisting of the value, the first and second derivative are extracted at the time
    points of the ICs ([1]_).
    This results in a 6-dimensional feature vector for each IC.
    This feature set is normalized using the provided scaler and then classified using the provided model.

    We provde a set of pre-trained models that are based on the MS-Project (TODO: Ref paper) dataset.
    They all use a Min-Max Scaler in combination with a linear SVC classifier.
    The parameters of the SVC depend on the cohort and were tuned as explained in the paper ([1]_).


    Parameters
    ----------
    smoothing_filter
        The bandpass filter used to smooth the data before feature extraction.
    model
        The machine learning model used for step detection.
        This is expected to be a ML classifier instance.
    scaler
        The scikit-learn scaler used for scaling/normalizing the feature matrix.

    Attributes
    ----------
    %(ic_lr_list_)s
    feature_matrix_
        The 6-dimensional scaled feature vector, containing the vertical and anterior-posterior filtered angular
        velocities and their first (gradient) and second (curvature) derivatives.
        This might be helpful for debugging or further analysis.

    Other Parameters
    ----------------
    %(other_parameters)s


    Notes
    -----
    Instead of using diff, this implementation uses numpy.gradient to calculate the first and second derivative of the
    filtered signals.
    Compared to diff, gradient can estimate reliable derivatives for values at the edges of the data, which is
    important when the ICs are close to the beginning or end of the data.

    .. [1] Ullrich M, Küderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left
           and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021,
           available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653
    """

    model: SklearnClassifier
    scaler: SklearnScaler
    smoothing_filter: BaseFilter

    feature_matrix_: pd.DataFrame

    _feature_matrix_cols: Final = [
        "filtered__gyr_x",
        "gradient__gyr_x",
        "curvature__gyr_x",
        "filtered__gyr_z",
        "gradient__gyr_z",
        "curvature__gyr_z",
    ]

    class PredefinedParameters:
        """Predefined parameters for the LrdUllrich class."""

        _BW_FILTER = ButterworthFilter(order=4, cutoff_freq_hz=(0.5, 2), filter_type="bandpass")

        class _ModelConfig(TypedDict):
            smoothing_filter: BaseFilter
            model: SklearnClassifier
            scaler: SklearnScaler

        @classmethod
        def _load_model_config(cls, model_name: str) -> _ModelConfig:
            if model_name not in ["msproject_all", "msproject_hc", "msproject_ms"]:
                raise ValueError("Invalid model name.")

            model = _load_model_files(f"{model_name}_model.gz")
            scaler = _load_model_files(f"{model_name}_scaler.gz")

            # Note: this clip property was added here to ensure compatibility with sklearn version 0.23.1, which was
            #  used for training the models and storing the corresponding min-max scalers.
            #  Since version 0.24 onwards, this needs to be specified.
            # TODO: Might be a good idea to resave both the models and scalers to the current sklearn version.
            scaler.clip = False

            return {
                "smoothing_filter": cls._BW_FILTER,
                "model": model,
                "scaler": scaler,
            }

        @classproperty
        def msproject_all(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_model_config("msproject_all")

        @classproperty
        def msproject_hc(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_model_config("msproject_hc")

        @classproperty
        def msproject_ms(cls) -> _ModelConfig:  # noqa: N805
            return cls._load_model_config("msproject_ms")

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.msproject_all.items()})
    def __init__(
        self,
        smoothing_filter: BaseFilter,
        model: SklearnClassifier,
        scaler: SklearnScaler,
    ) -> None:
        self.smoothing_filter = smoothing_filter
        self.model = model
        self.scaler = scaler

    def _check_model(self, model: SklearnClassifier) -> None:
        if not isinstance(model, ClassifierMixin) or not isinstance(model, BaseEstimator) or not is_classifier(model):
            raise TypeError(
                f"Unknown model type {type(model).__name__}."
                "The model must inherit from ClassifierDtype and BaseEstimator. "
                "Any valid scikit-learn classifier can be used."
            )

    @base_lrd_docfiller
    def detect(
        self,
        data: pd.DataFrame,
        ic_list: pd.DataFrame,
        *,
        sampling_rate_hz: float,
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """
        %(detect_short)s.

        Parameters
        ----------
        %(detect_para)s

        %(detect_return)s
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("'data' must be a pandas DataFrame")

        if not isinstance(ic_list, pd.DataFrame):
            raise TypeError("'ic_list' must be a pandas DataFrame")

        self.data = data
        self.ic_list = ic_list
        self.sampling_rate_hz = sampling_rate_hz

        if data.empty or ic_list.empty:
            self.ic_lr_list_ = pd.DataFrame(columns=["ic", "lr_label"])
            self.feature_matrix_ = pd.DataFrame(columns=self.feature_matrix_.columns)
            return self

        self._check_model(self.model)

        # create a copy of ic_list, otherwise, they will get modified when adding the predicted labels
        # We also remove the "lr_label" column, if it exists, to avoid conflicts
        ic_list = ic_list.copy().drop(columns="lr_label", errors="ignore")

        try:
            check_is_fitted(self.model)
        except NotFittedError as e:
            raise RuntimeError("Model is not fitted. Call self_optimize before calling detect.") from e

        feature_matrix = self.extract_features(self.data, ic_list, self.sampling_rate_hz)

        # The old models were trained wihtout feature names, however, when we retrain the models, we do it with
        # feature names.
        # Once, we retrained all models, we can remove this destinction and assume we are working with dataframes
        # all the time.
        # Hence, we need to separate here:
        _cols = feature_matrix.columns
        _index = feature_matrix.index
        if getattr(self.scaler, "feature_names_in_", None) is None:
            feature_matrix = feature_matrix.to_numpy()

        feature_matrix_scaled = self.scaler.transform(feature_matrix)

        ic_list["lr_label"] = self.model.predict(feature_matrix_scaled)
        ic_list = ic_list.replace({"lr_label": {0: "left", 1: "right"}})

        self.ic_lr_list_ = ic_list

        if not isinstance(feature_matrix_scaled, pd.DataFrame):
            self.feature_matrix_ = pd.DataFrame(feature_matrix_scaled, columns=_cols, index=_index)

        return self

    @base_lrd_docfiller
    def self_optimize(
        self,
        data_sequences: Iterable[pd.DataFrame],
        ic_list_per_sequence: Iterable[pd.DataFrame],
        ref_ic_lr_list_per_sequence: Iterable[pd.DataFrame],
        *,
        sampling_rate_hz: Union[float, Iterable[float]],
        **_: Unpack[dict[str, Any]],
    ) -> Self:
        """Retrain the internal model and scaler using the provided data.

        .. note:: We only support a full re-fit of the model and the scaler.
           Therefore, you have to pass an untrained instance to the algorithm instance before calling ``self_optimize.``

        Parameters
        ----------
        %(self_optimize_paras)s

        %(self_optimize_return)s
        """
        self._check_model(self.model)

        try:
            check_is_fitted(self.model)
        except NotFittedError:
            pass
        else:
            raise RuntimeError("Model is already fitted. Initialize the algorithm with a untrained classifier object.")

        try:
            check_is_fitted(self.scaler)
        except NotFittedError:
            pass
        else:
            raise RuntimeError("Scaler is already fitted. Initialize the algorithm with a untrained scaler object.")

        if isinstance(sampling_rate_hz, float):
            sampling_rate_hz = cycle([sampling_rate_hz])

        features = [
            self.extract_features(dp, ic, sr)
            for dp, ic, sr in zip(data_sequences, ic_list_per_sequence, sampling_rate_hz)
        ]
        all_features = pd.concat(features, axis=0, ignore_index=True) if len(features) > 1 else features[0]
        all_features = self.scaler.fit_transform(all_features)

        # Concatenate the labels if there is more than one GS
        label_list = [ic_lr_list["lr_label"] for ic_lr_list in ref_ic_lr_list_per_sequence]
        all_labels = pd.concat(label_list, axis=0, ignore_index=True) if len(label_list) > 1 else label_list[0]

        self.model.fit(all_features, all_labels)

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
        gyr = data[["gyr_x", "gyr_z"]]
        gyr_filtered = self.smoothing_filter.clone().filter(gyr, sampling_rate_hz=sampling_rate_hz).filtered_data_
        # We use numpy gradient instead of diff, as it preserves the shape of the input and hence, can handle ICs that
        # are close to the beginning or end of the data.
        gyr_gradient = np.gradient(gyr_filtered, axis=0)
        curvature = np.gradient(gyr_gradient, axis=0)
        gyr_gradient = pd.DataFrame(gyr_gradient, columns=["gyr_x", "gyr_z"], copy=False, index=gyr.index)
        curvature = pd.DataFrame(curvature, columns=["gyr_x", "gyr_z"], copy=False, index=gyr.index)

        signal_paras = pd.concat({"filtered": gyr_filtered, "gradient": gyr_gradient, "curvature": curvature}, axis=1)
        signal_paras.columns = ["__".join(c) for c in signal_paras.columns]

        # Extract features corresponding to the adjusted ics values
        feature_df = signal_paras.iloc[ics["ic"].to_numpy()]
        feature_df.index = ics.index

        # Reorder the columns, to be consistent with the original implementation of [1].
        # Note: this order is necessary for the scaler to be applied correctly.
        feature_df = feature_df[self._feature_matrix_cols]

        return feature_df
