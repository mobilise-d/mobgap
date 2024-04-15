from functools import cache
from importlib.resources import files
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from tpcp import cf
from tpcp.misc import classproperty, set_defaults
from typing_extensions import Self, Unpack

from mobgap.data_transform import ButterworthFilter
from mobgap.data_transform.base import BaseFilter
from mobgap.lrd.base import BaseLRDetector, base_lrd_docfiller


@cache
def _load_model_files(file_name: str):
    file_path = files("mobgap") / "lrd" / "_ullrich_pretrained_models" / file_name
    with file_path.open("rb") as file:
        return joblib.load(file)


class LrdUllrich(BaseLRDetector):
    """
    Machine-Learning based algorithm for laterality detection of initial contacts.

    This algorithm uses the band-pass filtered vertical ("gyr_x") and anterior-posterior ("gyr_z") angular velocity signal in combination with their first and second derivatives at the time points of the ICs, as described in [1]. This results in a 6-dimensional feature set.

    To ensure uniformity, the feature set is min-max normalised.

    Parameters
    ----------
    model
        The machine learning model used for step detection. This is the base class for all classifiers used in scikit-learn.
    scaler
        The scikit-learn scaler used for the min-max normalisation.
    smoothing_filter
        The bandpass filter used to smooth the data.

    Attributes
    ----------
    %(ic_lr_list_)s

    feature_matrix_
        The 6-dimensional feature set, containing the vertical and anterior-posterior filtered angular velocities and their first and second derivatives.
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

    .. [1] Ullrich M, Küderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021, available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653
    """

    model: Optional[ClassifierMixin]
    scaler: Optional[MinMaxScaler]
    smoothing_filter: BaseFilter

    feature_matrix_: pd.DataFrame

    _feature_matrix_cols =  [
                "filtered_gyr_x",
                "gradient_gyr_x",
                "curvature_gyr_x",
                "filtered_gyr_z",
                "gradient_gyr_z",
                "curvature_gyr_z",
            ]

    class PredefinedParameters:
        """Predefined parameters for the LrdUllrich class."""

        _BW_FILTER = ButterworthFilter(order=4, cutoff_freq_hz=(0.5, 2), filter_type="bandpass")

        @classmethod
        def _load_from_file(cls, model_name):
            if model_name not in ["msproject_all", "msproject_hc", "msproject_ms"]:
                raise ValueError("Invalid model name.")

            model = _load_model_files(f"{model_name}_model.gz")
            scaler = _load_model_files(f"{model_name}_scaler.gz")

            # Note: this clip property was added here to ensure compatibility with sklearn version 0.23.1, which was
            #  used for training the models and storing the corresponding min-max scalers.
            #  Since version 0.24 onwards, this needs to be specified.
            # TODO: Might be a good idea to resave both the models and scalers to the current sklearn version.
            scaler.clip = False

            return model, scaler

        @classproperty
        def msproject_all(cls):
            model, scaler = cls._load_from_file("msproject_all")
            return {
                "smoothing_filter": cls._BW_FILTER,
                "model": model,
                "scaler": scaler,
            }

        @classproperty
        def msproject_hc(cls):
            model, scaler = cls._load_from_file("msproject_hc")
            return {
                "smoothing_filter": cls._BW_FILTER,
                "model": model,
                "scaler": scaler,
            }

        @classproperty
        def msproject_ms(cls):
            model, scaler = cls._load_from_file("msproject_ms")
            return {
                "smoothing_filter": cls._BW_FILTER,
                "model": model,
                "scaler": scaler,
            }

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.msproject_all.items()})
    def __init__(
        self,
        smoothing_filter: BaseFilter,
        model: ClassifierMixin,
        scaler: MinMaxScaler,
    ) -> None:
        self.smoothing_filter = smoothing_filter
        self.model = model
        self.scaler = scaler

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
            self.feature_matrix_ = pd.DataFrame(
                columns=self.feature_matrix_.columns
            )
            return self

        if (
            not isinstance(self.model, ClassifierMixin)
            or not isinstance(self.model, BaseEstimator)
            and is_classifier(self.model)
        ):
            raise TypeError(
                f"Unknown model type {type(self.model).__name__}."
                "The model must inherit from ClassifierMixin and BaseEstimator. "
                "Any valid scikit-learn classifier can be used."
            )

        # create a copy of ic_list, otherwise, they will get modified when adding the predicted labels
        # We also remove the "lr_label" column, if it exists, to avoid conflicts
        ic_list = ic_list.copy().drop(columns="lr_label", errors="ignore")

        try:
            check_is_fitted(self.model)
        except NotFittedError as e:
            raise RuntimeError("Model is not fitted. Call self_optimize before calling detect.") from e

        feature_matrix = self.extract_features(self.data, ic_list, self.sampling_rate_hz)
        feature_matrix_scaled = self.scaler.transform(feature_matrix.to_numpy())

        # Setting the feature matrix for debugging purposes
        self.feature_matrix_ = pd.DataFrame(
            feature_matrix_scaled, columns=feature_matrix.columns, index=feature_matrix.index
        )
        del feature_matrix
        ic_list["lr_label"] = self.model.predict(feature_matrix_scaled)
        ic_list = ic_list.replace({"lr_label": {0: "left", 1: "right"}})

        self.ic_lr_list_ = ic_list
        return self

    def self_optimize(
        self,
        data_list: list[pd.DataFrame],
        ic_list: list[pd.DataFrame],
        label_list: list[pd.DataFrame],
        sampling_rate_hz: float,
    ) -> Self:
        """
        Model optimization method based on the provided gait data, initial contact list, and the reference label list.

        Parameters
        ----------
        data
            The gait data.
        ic_list
            The initial contact list.
        label_list
            The label list.
        sampling_rate_hz
            The sampling rate in Hz.

        Returns
        -------
        self
         The instance of the LrdUllrich class.
        """
        if not isinstance(data_list, list):
            raise TypeError("'data' must be a list of pandas DataFrames")

        if not isinstance(ic_list, list):
            raise TypeError("'ic_list' must be a list of pandas DataFrame")

        if not isinstance(label_list, list):
            raise TypeError("'label_list' must be a list of pandas DataFrame")

        features = [self.extract_features(data, ic, sampling_rate_hz) for data, ic in zip(data_list, ic_list)]
        all_features = pd.concat(features, axis=0, ignore_index=True) if len(features) > 1 else features[0]
        all_features = pd.DataFrame(
            self.scaler.fit_transform(all_features.values), columns=all_features.columns, index=all_features.index
        )

        # Concatenate the labels if there is more than one GS
        all_labels = pd.concat(label_list, axis=0, ignore_index=True) if len(label_list) > 1 else label_list[0]

        self.model.fit(all_features.to_numpy(), np.ravel(all_labels))

        return self

    def extract_features(
        self,
        data: pd.DataFrame,
        ics: pd.DataFrame,
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
        """
        Extracts features from the provided gait data and initial contact list.

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
        gyr_gradient = pd.DataFrame(gyr_gradient, columns=["gyr_x", "gyr_z"], copy=False)
        curvature = pd.DataFrame(curvature, columns=["gyr_x", "gyr_z"], copy=False)

        signal_paras = pd.concat({"filtered": gyr_filtered, "gradient": gyr_gradient, "curvature": curvature}, axis=1)
        signal_paras.columns = ["_".join(c) for c in signal_paras.columns]

        # Extract features corresponding to the adjusted ics values
        feature_df = signal_paras.loc[ics["ic"].to_numpy()]
        feature_df.index = ics.index

        # Reorder the columns, to be consistent with the original implementation of [1].
        # Note: this order is necessary for the scaler to be applied correctly.
        feature_df = feature_df[
           self._feature_matrix_cols
        ]

        return feature_df
