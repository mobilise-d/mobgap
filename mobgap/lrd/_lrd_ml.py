import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
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
from tpcp.misc import classproperty


class LrdUllrich(BaseLRDetector):
    """
    Machine-Learning based algorithm for laterality detection of initial contacts.

    This algorithm uses the band-pass filtered vertical ("gyr_x") and anterior-posterior ("gyr_z") angular velocity signal in combination with their first and second derivatives at the time points of the ICs, as described in [1]. This results in a 6-dimensional feature set. 

    To ensure uniformity, the feature set is min-max normalised.

    Parameters:
    ---------------
    model: Optional[ClassifierMixin]
        The machine learning model used for step detection. This is the base class for all classifiers used in scikit-learn.
    scaler: Optional[MinMaxScaler]
        The scikit-learn scaler used for the min-max nomalisation. 
    smoothing_filter:
        The bandpass filter used to smooth the data.
    
    Attributes:
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
            # get path to pretrained models and the name of all available model models
            # base_dir = Path(os.getenv('MODEL_DIR', './pretrained_models'))
            base_dir = Path(__file__).parent / os.getenv('MODEL_DIR', 'pretrained_models')
            valid_model_names = [f.name.rpartition('_')[0] for f in base_dir.glob('*') if f.is_file() and not f.name.startswith('.')]
            # remove duplicates (as the same name exists for both the model and the scaler) and sort alphabetically
            valid_model_names = list(set(valid_model_names))
            valid_model_names.sort()
            
            if (model_name in cls._model_cache) and (model_name in cls._scaler_cache):
                print("Loading cached model and scaler...")
                return cls._model_cache[model_name], cls._scaler_cache[model_name]

            if model_name in valid_model_names:
                print(f"Loading {model_name} model and scaler from file...")
                model_path = base_dir / f'{model_name}_model.gz'
                model = joblib.load(model_path) 
                cls._model_cache[model_name] = model
                scaler_path = base_dir / f'{model_name}_scaler.gz'
                scaler = joblib.load(scaler_path)

                # Note: this clip property was added here to ensure compatibility with sklearn version 0.23.1, which was used for training the models and storing the corresponding min-max scalers. Since version 0.24 onwards, this needs to be specified. Might be a good idea to resave both the models and scalers to the current sklearn version. 

                scaler.clip = False
                cls._scaler_cache[model_name] = scaler

                return model, scaler
            
            else:
                raise NotImplementedError("The pretrained configuration specified is not supported")
        
        @classproperty    
        def icicle_all(cls):
            model, scaler = cls._load_from_file("icicle_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}

        @classproperty
        def icicle_all_all(cls):
            model, scaler = cls._load_from_file("icicle_all_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def icicle_hc(cls):
            model, scaler = cls._load_from_file("icicle_hc")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def icicle_hc_all(cls):
            model, scaler = cls._load_from_file("icicle_hc_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def icicle_pd(cls):
            model, scaler = cls._load_from_file("icicle_pd")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def icicle_pd_all(cls):
            model, scaler = cls._load_from_file("icicle_pd_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_all(cls):
            model, scaler = cls._load_from_file("msproject_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_all_all(cls):
            model, scaler = cls._load_from_file("msproject_all_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_hc(cls):
            model, scaler = cls._load_from_file("msproject_hc")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_hc_all(cls):
            model, scaler = cls._load_from_file("msproject_hc_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_ms(cls):
            model, scaler = cls._load_from_file("msproject_ms")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def msproject_ms_all(cls):
            model, scaler = cls._load_from_file("msproject_ms_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_all(cls):
            model, scaler = cls._load_from_file("uniss_unige_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_all_all(cls):
            model, scaler = cls._load_from_file("uniss_unige_all_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_ch(cls):
            model, scaler = cls._load_from_file("uniss_unige_ch")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_ch_all(cls):
            model, scaler = cls._load_from_file("uniss_unige_ch_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_hc(cls):
            model, scaler = cls._load_from_file("uniss_unige_hc")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_hc_all(cls):
            model, scaler = cls._load_from_file("uniss_unige_hc_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_pd(cls):
            model, scaler = cls._load_from_file("uniss_unige_pd")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_pd_all(cls):
            model, scaler = cls._load_from_file("uniss_unige_pd_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        @classproperty    
        def uniss_unige_st(cls):
            model, scaler = cls._load_from_file("uniss_unige_st")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}
        
        @classproperty    
        def uniss_unige_st_all(cls):
            model, scaler = cls._load_from_file("uniss_unige_st_all")
            return {"smoothing_filter" : ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type = "bandpass"),
                    "model": model, "scaler": scaler}


    def __init__(self,
                 model: ClassifierMixin = None,
                 scaler: MinMaxScaler = None,
                #  lower_band: float = 0.5,
                #  upper_band: float = 2):
                smoothing_filter: BaseFilter = cf(ButterworthFilter(order = 4, cutoff_freq_hz = (0.5, 2), filter_type="bandpass"))
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.smoothing_filter = smoothing_filter

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
        # print("Model checking...")
        if isinstance(model, ClassifierMixin):
            return model
        raise TypeError(f"Unknown model type {type(model)}. The model must be of type {ClassifierMixin}")
    
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
        
        # Check if data and ic_list are empty
        if data.empty:
            raise ValueError("'data' must contain at least one data point")
        if ic_list.empty:
            raise ValueError("'ic_list' must contain at least one initial contact point")


        self.data = data
        self.ic_list = ic_list
        self.sampling_rate_hz = sampling_rate_hz

        self.model = self._check_and_init_model(self.model)

        # create a copy of ic_list, otherwise, they will get modified when adding the predicted labels
        # We also remove the "lr_label" column, if it exists, to avoid conflicts
        ic_list = ic_list.copy().drop(columns="lr_label", errors="ignore")

        try:
            check_is_fitted(self.model)
        except NotFittedError:
            raise RuntimeError("Model is not fitted. Call self_optimize before calling detect.")

        # Extract the features and scale the feature matrix to ensure uniformity.
        feature_matrix = self.extract_features(self.data, ic_list, self.sampling_rate_hz)
        self.feature_matrix_ = pd.DataFrame(self.scaler.transform(feature_matrix.values), columns = feature_matrix.columns, index = feature_matrix.index)

        prediction_per_gs = pd.DataFrame(self.model.predict(self.feature_matrix_.to_numpy()), columns = ["lr_label"])
        
        # Map model's output 0s and 1s to 'left' and 'right'
        mapping = {0: "left", 1: "right"}
        prediction_per_gs = prediction_per_gs.replace(mapping)

        # Flatten ic_list and create DataFrame
        self.ic_lr_list_ = pd.DataFrame({"ic": self.ic_list.values.flatten(), "lr_label": prediction_per_gs["lr_label"]})

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
        ic_list : list[pd.DataFrame]
            The initial contact list.
        label_list : list[pd.DataFrame]
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
            raise TypeError("'ic_list' must be a list of pandas DataFrames")
        
        # preprocess data for all GSs.
        features = [self.extract_features(data, ic, sampling_rate_hz) for data, ic in zip(data_list, ic_list)]

        # Concatenate the features if there is more than one GS
        all_features = pd.concat(features, axis=0, ignore_index=True) if len(features) > 1 else features[0]

        # Initialize the scaler if it's not already set
        if self.scaler is None:
            self.scaler = MinMaxScaler()

        # Fit the scaler if it hasn't been fitted yet, otherwise just transform the data
        if not hasattr(self.scaler, 'scale_'):
            all_features = pd.DataFrame(self.scaler.fit_transform(all_features.values), columns=all_features.columns, index=all_features.index)
        else:
            all_features = pd.DataFrame(self.scaler.transform(all_features.values), columns=all_features.columns, index=all_features.index)

        # Concatenate the labels if there is more than one GS
        all_labels = pd.concat(label_list, axis=0, ignore_index=True) if len(label_list) > 1 else label_list[0]

        # Fit the model with the features and labels
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

        Returns:
        ---------------
        feature_df (pd.DataFrame): The DataFrame containing the extracted features.
        """
        # copy ics, otherwise it will be modified by ics -= 1 and then shifted.
        ics = ics.copy()
        
        # Apply Butterworth filtering and extract the first and second derivatives.        
        gyr = data[["gyr_x", "gyr_z"]].rename({"gyr_x": "v", "gyr_z": "ap"})
        gyr_filtered = self.smoothing_filter.clone().filter(gyr, sampling_rate_hz = sampling_rate_hz).filtered_data_
        gyr_diff = gyr_filtered.diff()
        gyr_diff_2 = gyr_diff.diff()

        # Combine filtered data and its derivatives into a single DataFrame
        signal_paras = pd.concat({"filtered": gyr_filtered, "gradient": gyr_diff, "diff_2": gyr_diff_2}, axis=1)
        # Squash the multi index
        signal_paras.columns = ["_".join(c) for c in signal_paras.columns]

        # shift the last IC by 3 samples to make the second derivative work    
        ics -= 1
        ics[ics < 2] = 2

        # Extract features corresponding to the adjusted ics values
        feature_df = signal_paras.loc[ics['ic'].values.tolist()]

        # Reorder the columns, to be consistent with the original implementation of [1].
        # Note: this order is necessary for the scaler to be applied correctly.
        feature_df = feature_df[["filtered_gyr_x", "gradient_gyr_x", "diff_2_gyr_x", "filtered_gyr_z", "gradient_gyr_z", "diff_2_gyr_z"]]

        return feature_df.reset_index(drop = True)