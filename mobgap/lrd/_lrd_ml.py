import os
import joblib
import numpy as np
import pandas as pd
from importlib.resources import files
from typing import Optional, Any
from typing_extensions import Self, Unpack


from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler

from mobgap.data_transform.base import BaseFilter
from mobgap.data_transform import ButterworthFilter
from mobgap.lrd.base import BaseLRDetector, base_lrd_docfiller

from tpcp import cf
from tpcp.misc import set_defaults
from tpcp.misc import classproperty


class LrdUllrich(BaseLRDetector):
    """
    Machine-Learning based algorithm for laterality detection of initial contacts.

    This algorithm uses the band-pass filtered vertical ("gyr_x") and anterior-posterior ("gyr_z") angular velocity signal in combination with their first and second derivatives at the time points of the ICs, as described in [1]. This results in a 6-dimensional feature set. 

    To ensure uniformity, the feature set is min-max normalised.

    Parameters
    ---------------
    model: Optional[ClassifierMixin]
        The machine learning model used for step detection. This is the base class for all classifiers used in scikit-learn.
    scaler: Optional[MinMaxScaler]
        The scikit-learn scaler used for the min-max nomalisation. 
    smoothing_filter:
        The bandpass filter used to smooth the data.
    
    Attributes
    ---------------
    %(ic_lr_list_)s

    feature_matrix_
        The 6-dimensional feature set, containing the vertical and anterior-posterior filtered angular velocities and their first and second derivatives.
        This might be helpful for debugging or further analysis.
        
    Other Parameters
    ----------------
    %(other_parameters)s
    

    Reference papers
    ----------------
    .. [1] Ullrich M, Küderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021, available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653
    """
    model: Optional[ClassifierMixin]
    scaler: Optional[MinMaxScaler]
    smoothing_filter: BaseFilter


    class PredefinedParameters:
        """
        A class used to manage predefined parameters for the LrdUllrich class.

        Attributes:
        ---------------
        _model_cache: dict
            A dictionary used to cache loaded models to avoid loading the same model multiple times.
        _scaler_cache: dict
            A dictionary used to cache loaded scalers to avoid loading the same scaler multiple times.
        """
        _model_cache = {}
        _scaler_cache = {}

        @classmethod
        def _load_from_file(cls, model_name):
            if (model_name in cls._model_cache) and (model_name in cls._scaler_cache):
                print("Loading cached model and scaler...")
                return cls._model_cache[model_name], cls._scaler_cache[model_name]

            if model_name in ["msproject_all", "msproject_hc", "msproject_ms"]:
                print(f"Loading {model_name} model and scaler from file...")

                model_path = files('gaitlink') / 'lrd' / 'pretrained_models' / f'{model_name}_model.gz'
                with model_path.open('rb') as file:
                    model = joblib.load(file)
                cls._model_cache[model_name] = model

                scaler_path = files('gaitlink') / 'lrd' / 'pretrained_models' / f'{model_name}_scaler.gz'
                with scaler_path.open('rb') as file:
                    scaler = joblib.load(file)

                # Note: this clip property was added here to ensure compatibility with sklearn version 0.23.1, which was used for training the models and storing the corresponding min-max scalers. Since version 0.24 onwards, this needs to be specified. Might be a good idea to resave both the models and scalers to the current sklearn version. 

                scaler.clip = False
                cls._scaler_cache[model_name] = scaler

                return model, scaler
            
            else:
                raise NotImplementedError("The specified pretrained configuration is not supported")
        
        @classproperty    
        def msproject_all(cls):
            model, scaler = cls._load_from_file("msproject_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_hc(cls):
            model, scaler = cls._load_from_file("msproject_hc")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_ms(cls):
            model, scaler = cls._load_from_file("msproject_ms")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}

    @set_defaults(**{k: cf(v) for k, v in PredefinedParameters.msproject_all.items()})
    def __init__(self,
                smoothing_filter: BaseFilter,
                model: ClassifierMixin,
                scaler: MinMaxScaler,
    ) -> None:
        super().__init__()
        self.smoothing_filter = smoothing_filter
        self.model = model
        self.scaler = scaler


    # Model checking
    @staticmethod
    def _check_and_init_model(model):
        """
        Checks if the provided model is of type ClassifierMixin and returns it. Raises a TypeError if the check fails.

        Parameters:
        ---------------
        model : ClassifierMixin)
            The scikit-learn model to check.

        Returns:
        ---------------
        model : ClassifierMixin
            The checked model.
        """
        if isinstance(model, ClassifierMixin):
            return model
        raise TypeError(f"Unknown model type {type(model).__name__}. The model must be of type {ClassifierMixin.__name__}")
    
    @base_lrd_docfiller
    def detect(self,
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
            self.ic_lr_list_ = pd.DataFrame(columns = ["ic", "lr_label"])
            self.feature_matrix_ = pd.DataFrame(columns = ["filtered_gyr_x", "gradient_gyr_x", "diff_2_gyr_x", "filtered_gyr_z", "gradient_gyr_z", "diff_2_gyr_z"])
            return self

        self.model = self._check_and_init_model(self.model)

        # create a copy of ic_list, otherwise, they will get modified when adding the predicted labels
        # We also remove the "lr_label" column, if it exists, to avoid conflicts
        ic_list = ic_list.copy().drop(columns="lr_label", errors="ignore")

        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise RuntimeError("Model is not fitted. Call self_optimize before calling detect.")

        feature_matrix = self.extract_features(self.data, ic_list, self.sampling_rate_hz)
        self.feature_matrix_ = pd.DataFrame(self.scaler.transform(feature_matrix.to_numpy()), columns = feature_matrix.columns, index = feature_matrix.index)
        prediction_per_gs = pd.DataFrame(self.model.predict(self.feature_matrix_.to_numpy()), columns = ["lr_label"])
        
        mapping = {0: "left", 1: "right"}
        prediction_per_gs = prediction_per_gs.replace(mapping)

        self.ic_lr_list_ = pd.DataFrame({"ic": self.ic_list.to_numpy().flatten(), "lr_label": prediction_per_gs["lr_label"]})
        return self

    def self_optimize(self,
        data_list: list[pd.DataFrame],
        ic_list: list[pd.DataFrame], 
        label_list: list[pd.DataFrame],
        sampling_rate_hz: float,
    ) -> Self:  
        """
        Model optimization method based on the provided gait data, initial contact list, and the reference label list.

        Parameters:
        ---------------
        data : list[pd.DataFrame]
            The gait data.
        ic_list : list[pd.Series]
            The initial contact list.
        label_list : list[pd.Series]
            The label list.
        sampling_rate_hz : float
            The sampling rate in Hz.

        Returns:
        ---------------
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
        all_features = pd.DataFrame(self.scaler.fit_transform(all_features.values), columns=all_features.columns, index=all_features.index)

        # Concatenate the labels if there is more than one GS
        all_labels = pd.concat(label_list, axis=0, ignore_index=True) if len(label_list) > 1 else label_list[0]

        self.model.fit(all_features.to_numpy(), np.ravel(all_labels))

        return self
        
    
    def extract_features(self,
        data: pd.DataFrame,
        ics: pd.DataFrame,
        sampling_rate_hz: float,
    ) -> pd.DataFrame:
        """
        Extracts features from the provided gait data and initial contact list.

        Here, the feature set ic composed of the first and second derivatives of the filtered signals at the time points of the  ICs. Consequently, for a dataset containing a total of  ICs, this results in a feature matrix. To ensure uniformity, the feature set is min-max normalized.

        Parameters:
        ---------------
        data (pd.DataFrame): The gait data.
        ics (pd.DataFrame): The initial contact list.
        sampling_rate_hz (float): The sampling rate in Hz.

        Note:
        ---------------
        The last initial contact (IC) is shifted by 3 samples to ensure that the second derivative can be calculated. This shift only occurs if the last IC is within 3 samples of the end of the data. Additionally, any ICs that are less than 2 are set to 2 to avoid extracting rows with NaN values due to the calculation of the derivatives.

        Returns:
        ---------------
        feature_df (pd.DataFrame): The DataFrame containing the extracted features.
        """
        ics = ics.copy()
        gyr = data[["gyr_x", "gyr_z"]]
        gyr_filtered = self.smoothing_filter.clone().filter(gyr, sampling_rate_hz = sampling_rate_hz).filtered_data_
        gyr_diff = gyr_filtered.diff()
        gyr_diff_2 = gyr_diff.diff()

        signal_paras = pd.concat({"filtered": gyr_filtered, "gradient": gyr_diff, "diff_2": gyr_diff_2}, axis=1)
        signal_paras.columns = ["_".join(c) for c in signal_paras.columns]

        # shift the last IC by 3 samples to make the second derivative work
        if (ics.iloc[-1]).any() >= len(signal_paras):
            ics.iloc[-1] -= 3
        ics[ics < 2] = 2

        # Extract features corresponding to the adjusted ics values
        feature_df = signal_paras.loc[ics['ic'].to_numpy().tolist()]

        # Reorder the columns, to be consistent with the original implementation of [1].
        # Note: this order is necessary for the scaler to be applied correctly.
        feature_df = feature_df[["filtered_gyr_x", "gradient_gyr_x", "diff_2_gyr_x", "filtered_gyr_z", "gradient_gyr_z", "diff_2_gyr_z"]]

        return feature_df.reset_index(drop = True)