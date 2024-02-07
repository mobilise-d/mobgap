import numpy as np
import pandas as pd
from typing import Union

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from tpcp import Algorithm


from gaitlink.data_transform import ButterworthFilter
from gaitlink.lr_detection.base import BaseLRDetector, base_lr_detector_docfiller, PretrainedModel


from tpcp.misc import set_defaults
from tpcp.misc import classproperty
from typing import Optional
import os
from pathlib import Path
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class UllrichLRDetection(BaseLRDetector):
    lower_band: float
    upped_band: float
    model: Optional[ClassifierMixin]
    scaler: Optional[MinMaxScaler]
    """
    This class uses machine learning techniques to predict whether each pre-determined initial contact (IC) corresponds to a left or a right step.

    The methodology used here is based on the following reference paper:
    Reference Papers: Ullrich M, Kuderle A, Reggi L, Cereatti A, Eskofier BM, Kluge F. Machine learning-based distinction of left and right foot contacts in lower back inertial sensor gait data. Annu Int Conf IEEE Eng Med Biol Soc. 2021, available at: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9630653

    Attributes:
    ---------------
    lower_band: float
        The lower frequency band for the Butterworth filter used for processing the data.
    upper_band : float
        The upper frequency band for the Butterworth filter used for processing the data.
    model : Optional[ClassifierMixin]
        The machine learning model used for step detection. This is the base class for all classifiers used in scikit-learn.
    """


    class PredefinedParameters:
        """
        A class used to manage predefined parameters for the UllrichLRDetection class.

        Attributes:
        ---------------
        _cache: dict
            A dictionary used to cache loaded models to avoid loading the same model multiple times.
        """
        _model_cache = {}
        _scaler_cache = {}

        @classmethod
        def _load_from_file(cls, model_name):
            if (model_name in cls._model_cache) and (model_name in cls._scaler_cache):
                print("Loading cached model and scaler...")
                return cls._model_cache[model_name], cls._scaler_cache[model_name]

            if model_name in ['hc', 'pd', 'all']:
                print(f"Loading {model_name} model and scaler from file...")
                base_dir = Path(os.getenv('MODEL_DIR', './pretrained_models'))
                model_path = base_dir / f'uniss_unige_{model_name}_model.gz'
                model = joblib.load(model_path) 
                cls._model_cache[model_name] = model
                scaler_path = base_dir / f'uniss_unige_{model_name}_scaler.gz'
                scaler = joblib.load(scaler_path)

                # TODO: this clip property was added here to ensure compatibility with sklearn version 0.23.1, which was used for training the models and storing the corresponding min-max scalers. Since version 0.24 onwards, this needs to be specified. Might be a good idea to resave both the models and scalers to the current sklearn version. 

                scaler.clip = False
                cls._scaler_cache[model_name] = scaler

                return model, scaler
            
            else:
                raise NotImplementedError("The pretrained configuration specified is not supported")
        
        # TODO: add the remaining pre-trained models. The lower and upper bands will stay the same.
        # TODO: maybe also change the names, i.e. pd -> parkinson, ms -> multiple_sclerosis, etc... probably not, as this was, we ensure uniformity with the model and scaler naming convention...
            
        @classproperty    
        def hc(cls):
            model, scaler = cls._load_from_file("hc")
            return {"lower_band": 0.5, "upper_band" : 2, "model": model, "scaler": scaler}

        @classproperty
        def pd(cls):
            model, scaler = cls._load_from_file("pd")
            return {"lower_band": 0.5, "upper_band" : 2, "model": model, "scaler": scaler}
        
        @classproperty    
        def all(cls):
            model, scaler = cls._load_from_file("all")
            return {"lower_band": 0.5, "upper_band" : 2, "model": model, "scaler": scaler}

    def __init__(self,
                 model: ClassifierMixin = None,
                 scaler: MinMaxScaler = None,
                 lower_band: float = 0.5,
                 upper_band: float = 2):
        self.model = model
        self.scaler = scaler
        self.lower_band = lower_band
        self.upper_band = upper_band

    # Model checking
    @staticmethod
    def _check_and_init_model(model):
        """
        Checks if the provided model is of type ClassifierMixin and returns it. Raises a TypeError if the check fails.

        Parameters:
        ---------------
        model : ClassifierMixin)
            The scikit-learn-based machine learning model to check.

        Returns:
        ---------------
        model : ClassifierMixin
            The checked machine learning model.
        """
        print("Model checking...")
        if isinstance(model, ClassifierMixin):
            return model
        raise TypeError(f"Unknown model type {type(model)}. The model must be of type {ClassifierMixin}")
    
    @base_lr_detector_docfiller
    def detect(self, data: list[pd.DataFrame], ic_list: list[pd.DataFrame], sampling_rate_hz: float = 100):
        """
        Method used for detection of left and right steps in the provided gait data. Predictions are stored post-execution using the `ic_lr` attribute of the instance, i.e. self.ic_lr.

        Parameters:
        ---------------
        data : list[pd.DataFrame]
            The gait data.
        ic_list: list[pd.DataFrame]
            The initial contact list.
        sampling_rate_hz : float
            The sampling rate in Hz.

        Returns:
        ---------------
        self: 
            The instance of the class.

        Predictions can be retrieved using self.ic_lr
        """

        if not isinstance(data, list):
            raise TypeError("data must be a list")
            
        if not isinstance(ic_list, list):
            raise TypeError("ic_list must be a list")

        self.data = data
        self.ics = ic_list
        self.sampling_rate_hz = sampling_rate_hz

        self.model = self._check_and_init_model(self.model)

        # TODO: is this copy necessary? probably not...
        model = self.model 

        try:
            check_is_fitted(model)
        except NotFittedError:
            raise RuntimeError("Model is not fitted. Call self_optimize before calling detect.")

        self.processed_data = []
        self.ic_lr = []

        for gs in range(len(data)):
            self.processed_data.append(self.extract_features(self.data[gs], self.ics[gs], self.sampling_rate_hz))

            # store the predictions in a df, to be consistent with the label_list
            prediction_name = ["predicted_lr_label"]
            prediction_per_gs = pd.DataFrame(model.predict(self.processed_data[gs].to_numpy()), columns = prediction_name)
            
            # Note that petrained models output 0s and 1s. For consistency, these are mapped to 'Left' and 'Right'
            mapping = {0: "Left", 1: "Right"}
            prediction_per_gs = pd.DataFrame(prediction_per_gs).replace(mapping)

            self.ic_lr.append(prediction_per_gs)
        
        # ALEX: It might be more elegant to only return the value of the prediction and nothing else: i.e. do not store anything internally (data, ic_list, processed_data, predictions)
        return self
    

    @base_lr_detector_docfiller
    def self_optimize(self, data: list[pd.DataFrame], ic_list, label_list: list[pd.DataFrame], sampling_rate_hz: float = 100):
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
         The instance of the class.
        """

        model = self.model
        
        if not isinstance(data, list):
            raise TypeError("data must be a list")
            
        if not isinstance(ic_list, list):
            raise TypeError("ic_list must be a list")
        
        # preprocess data
        features = []
        for gs in range(len(data)):
            # TODO: Only fit the scaler if it's not already fitted
            fit_scaler = not hasattr(self.scaler, 'scale_')

            # deprecated
            # features.append(self.extract_features(data[gs], ic_list[gs], sampling_rate_hz, fit_scaler = True))
            features.append(self.extract_features(data[gs], ic_list[gs], sampling_rate_hz, fit_scaler = fit_scaler))

        # If there is more than one GS, concatenate the features
        if len(features) > 1:
            all_features = pd.concat(features, axis=0, ignore_index=True)
        # If there is only one GS, no need to concatenate
        else:
            all_features = features[0]

        # Do the same for labels
        if len(label_list) > 1:
            all_labels = pd.concat(label_list, axis=0, ignore_index=True)
        else:
            all_labels = label_list[0]

        # convert the features to numpy, to be consistent with the pretrained models, as there were fit without feature names.
        self.model = model.fit(all_features.to_numpy(), np.ravel(all_labels))

        return self
    
    
    @base_lr_detector_docfiller
    def extract_features(self, data: pd.DataFrame, ics: pd.DataFrame, sampling_rate_hz: float = 100, fit_scaler: bool = False):
        """
        Extracts features from the provided gait data and initial contact list.

        Here, the feature set ic composed of the first and second derivatives of the filtered signals at the time points of the  ICs. Consequently, for a dataset containing a total of  ICs, this results in a feature matrix. To ensure uniformity, the feature set is min-max normalized.

        Parameters:
        ---------------
        data (pd.DataFrame): The gait data.
        ics (pd.DataFrame): The initial contact list.
        sampling_rate_hz (float): The sampling rate in Hz.
        fit_scaler (bool): Whether to fit the scaler on the data. Should be True for training data and False for validation/test data.


        Returns:
        ---------------
        feature_df (pd.DataFrame): The DataFrame containing the extracted features.
        """
        # copy ics, otherwise it will be modified by ics -= 1 and then shifted.
        ics = ics.copy()
        
        # Apply Butterworth filtering and extract the first and second derivatives.
        butter_filter = ButterworthFilter(order=4, cutoff_freq_hz=(self.lower_band, self.upper_band), filter_type="bandpass")
        
        gyr = data[["gyr_x", "gyr_z"]].rename({"gyr_x": "v", "gyr_z": "ap"})
        gyr_filtered = butter_filter.filter(gyr, sampling_rate_hz = sampling_rate_hz).filtered_data_
        gyr_diff = gyr_filtered.diff()
        gyr_diff_2 = gyr_diff.diff()
        signal_paras = pd.concat({"filtered": gyr_filtered, "gradient": gyr_diff, "diff_2": gyr_diff_2}, axis=1)
        # Squash the multi index
        signal_paras.columns = ["_".join(c) for c in signal_paras.columns]
        
        ics -= 1
        # shift the last IC by 3 samples to make the second derivative work
        ics[ics < 2] = 2

        feature_df = signal_paras.loc[ics['ic'].values.tolist()]

        # Initialize the scaler if it doesn't exist (i.e. this is a custom model, which has no pretrained params and the model and the scaler were not previously fit to the data)
        # 

        if self.scaler is None:
            self.scaler = MinMaxScaler()

        # Fit the scaler if fit_scaler is True, otherwise just transform the data
        if fit_scaler:
            feature_df = pd.DataFrame(self.scaler.fit_transform(feature_df.values), columns=feature_df.columns, index=feature_df.index)
        else:
            feature_df = pd.DataFrame(self.scaler.transform(feature_df.values), columns=feature_df.columns, index=feature_df.index)
  
        return feature_df



# old version from here on...
# class UllrichLRDetection(BaseLRDetector):
#     """
#     Left/Right foot detector based on ML approaches [insert reference here].
#     """

#     PRETRAINED_MODEL = PretrainedModel

#     def __init__(self, model: Union[PretrainedModel, ClassifierMixin] = PretrainedModel.svm_linear):
#         self.model = model

#     @staticmethod
#     def _check_and_init_model(model):
#         """
#         Add docs here.
#         """
#         # print("Model_checking...")
#         if isinstance(model, PretrainedModel):
#             model = PretrainedModel.load_pretrained_model(model.value)
#         if isinstance(model, ClassifierMixin):
#             return model
#         raise TypeError(f"Unknown model type {type(model)}. The model must be of type {PretrainedModel} or {ClassifierMixin}")
    

#     @base_lr_detector_docfiller
#     def detect(self, data: list[pd.DataFrame], ic_list: list[pd.DataFrame], sampling_rate_hz: float = 100):
#         """
#         Add docs here.
#         """

#         if not isinstance(data, list):
#             raise TypeError("data must be a list")
            
#         if not isinstance(ic_list, list):
#             raise TypeError("ic_list must be a list")

#         self.data = data
#         self.ics = ic_list
#         self.sampling_rate_hz = sampling_rate_hz

#         self.model = self._check_and_init_model(self.model)

#         # TODO: is this copy necessary? probably not...
#         model = self.model 

#         try:
#             check_is_fitted(model)
#         except NotFittedError:
#             raise RuntimeError("Model is not fitted. Call self_optimize before calling detect.")

#         self.processed_data = []
#         self.ic_lr = []

#         for gs in range(len(data)):
#             self.processed_data.append(self.extract_features(self.data[gs], self.ics[gs], self.sampling_rate_hz))

#             # store the predictions in a df, to be consistent with the label_list
#             prediction_name = ["predicted_lr_label"]
#             prediction_per_gs = pd.DataFrame(model.predict(self.processed_data[gs].to_numpy()), columns = prediction_name)
            
#             # Note that petrained models output 0s and 1s. For consistency, these are mapped to 'Left' and 'Right'
#             mapping = {0: "Left", 1: "Right"}
#             prediction_per_gs = pd.DataFrame(prediction_per_gs).replace(mapping)

#             self.ic_lr.append(prediction_per_gs)
        
#         # ALEX: It might be more elegant to only return the value of the prediction and nothing else: i.e. do not store anything internally (data, ic_list, processed_data, predictions)
#         return self
    
#     @base_lr_detector_docfiller
#     def self_optimize(self, data: list[pd.DataFrame], ic_list, label_list: list[pd.DataFrame], sampling_rate_hz: float = 100):
#         """
#         Add docs here.
#         """

#         model = self.model
        
#         if not isinstance(data, list):
#             raise TypeError("data must be a list")
            
#         if not isinstance(ic_list, list):
#             raise TypeError("ic_list must be a list")
        
#         # preprocess data
#         features = []
#         for gs in range(len(data)):
#             features.append(self.extract_features(data[gs], ic_list[gs], sampling_rate_hz))

#         # If there is more than one GS, concatenate the features
#         if len(features) > 1:
#             all_features = pd.concat(features, axis=0, ignore_index=True)
#         # If there is only one GS, no need to concatenate
#         else:
#             all_features = features[0]

#         # Do the same for labels
#         if len(label_list) > 1:
#             all_labels = pd.concat(label_list, axis=0, ignore_index=True)
#         else:
#             all_labels = label_list[0]

#         # convert the features to numpy, to be consistent with the pretrained models, as there were fit without feature names.
#         self.model = model.fit(all_features.to_numpy(), np.ravel(all_labels))

#         return self
    
    
#     @base_lr_detector_docfiller
#     def extract_features(self, data: pd.DataFrame, ics: pd.DataFrame, sampling_rate_hz: float = 100):
#         """
#         Add docs here.
#         """
#         # copy ics, otherwise it will be modified by ics -= 1 and then shifted.
#         ics = ics.copy()

#         # TODO: We should probably expose these parameters, but than we also need to "store" them together with the pretrained models, as the models are specific to the preprocssing paras.
#         lower_band = 0.5
#         upper_band = 2
        
#         # Apply Butterworth filtering and extract the first and second derivatives.
#         butter_filter = ButterworthFilter(order=4, cutoff_freq_hz=(lower_band, upper_band), filter_type="bandpass")
        
#         gyr = data[["gyr_x", "gyr_z"]].rename({"gyr_x": "v", "gyr_z": "ap"})
#         gyr_filtered = butter_filter.filter(gyr, sampling_rate_hz = sampling_rate_hz).filtered_data_
#         gyr_diff = gyr_filtered.diff()
#         gyr_diff_2 = gyr_diff.diff()
#         signal_paras = pd.concat({"filtered": gyr_filtered, "gradient": gyr_diff, "diff_2": gyr_diff_2}, axis=1)
#         # Squash the multi index
#         signal_paras.columns = ["_".join(c) for c in signal_paras.columns]
        
#         ics -= 1
#         # shift the last IC by 3 samples to make the second derivative work
#         ics[ics < 2] = 2

#         feature_df = signal_paras.loc[ics['ic'].values.tolist()]
  
#         return feature_df
        