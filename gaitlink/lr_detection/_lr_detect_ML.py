import numpy as np
import pandas as pd
from typing import Union

from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from tpcp import Algorithm


from gaitlink.data_transform import ButterworthFilter
from gaitlink.lr_detection.base import BaseLRDetector, base_lr_detector_docfiller, PretrainedModel


class UllrichLRDetection(BaseLRDetector):
    """
    Left/Right foot detector based on ML approaches [insert reference here].
    """

    PRETRAINED_MODEL = PretrainedModel

    def __init__(self, model: Union[PretrainedModel, ClassifierMixin] = PretrainedModel.svm_linear):
        self.model = model

    @staticmethod
    def _check_and_init_model(model):
        """
        Add docs here.
        """
        # print("Model_checking...")
        if isinstance(model, PretrainedModel):
            model = PretrainedModel.load_pretrained_model(model.value)
        if isinstance(model, ClassifierMixin):
            return model
        raise TypeError(f"Unknown model type {type(model)}. The model must be of type {PretrainedModel} or {ClassifierMixin}")
    

    @base_lr_detector_docfiller
    def detect(self, data: list[pd.DataFrame], ic_list: list[pd.DataFrame], sampling_rate_hz: float = 100):
        """
        Add docs here.
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
        Add docs here.
        """

        model = self.model
        
        if not isinstance(data, list):
            raise TypeError("data must be a list")
            
        if not isinstance(ic_list, list):
            raise TypeError("ic_list must be a list")
        
        # preprocess data
        features = []
        for gs in range(len(data)):
            features.append(self.extract_features(data[gs], ic_list[gs], sampling_rate_hz))

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
    def extract_features(self, data: pd.DataFrame, ics: pd.DataFrame, sampling_rate_hz: float = 100):
        """
        Add docs here.
        """
        # copy ics, otherwise it will be modified by ics -= 1 and then shifted.
        ics = ics.copy()

        # TODO: We should probably expose these parameters, but than we also need to "store" them together with the pretrained models, as the models are specific to the preprocssing paras.
        lower_band = 0.5
        upper_band = 2
        
        # Apply Butterworth filtering and extract the first and second derivatives.
        butter_filter = ButterworthFilter(order=4, cutoff_freq_hz=(lower_band, upper_band), filter_type="bandpass")
        
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
  
        
        return feature_df
        