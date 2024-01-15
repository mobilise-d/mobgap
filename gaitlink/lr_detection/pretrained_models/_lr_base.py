from tpcp import Algorithm
import numpy as np
import pandas as pd
import warnings
import joblib
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from _utils import _butter_bandpass_filter
from _ml_train import LR_Train



class LR_Detector(Algorithm):
    """
    This this method detects the left and right leg corresponding to ICs based on a single IMU placed on the lower back.
    
    This is split into two approached.
    1) The McCamley algorithm
    
    2) Machine Learning (ML) based approaches.
    
    Parameters
    ----------
    
    
    Attributes
    ----------
    
    Other Parameters
    ----------------    
    """
    # TODO: presumably, the samplig rate should be inherited?
    def __init__(self,
                 imu_data: pd.DataFrame,
                 sampling_rate: int,
                 event_list: list,
                 approach: str = 'McCamley',
                 reference_data: bool = False,
                 **kwargs):
        """
        imu_data: - pandas DataFrame containing the raw IMU data.
        event_list: - a list of dictionaries containing the gait sequence reference data. From here, we need the gait sequence start and end times, turning points, as well as the the time points associated to ICs.
        """
        self.sampling_rate = sampling_rate
        
        data_list, sequence_label, ic_list, reference_lr_list = self.extract_sequence_data(imu_data, event_list, reference_data)
        
        print(len(data_list))
        
        if approach == 'McCamley':
            print('do something')
            for window_no in range(len(data_list)):
                # apply McCamley per each individual_segments
                
                # TODO: should we exclude curvilinear walking (i.e. turns?)
                
                if sequence_label[window_no] == 'Linear':
                    
                    print('do something')
        
        # TODO: you should also have a utility function, in order to measure the accuracy of the algorithms.
                    
         # you need to pass pretrained, if you want to load up pretrained models
         # you need to pass the patient group ['ALL', 'HC', 'MS', 'PD']  
         
        pretrained = kwargs.get('pretrained')
        model = kwargs.get('model')   
        patient_group = kwargs.get('patient_group')

        
        if approach.upper() == 'ML':
            # TODO: this should default to pre-trained models, allowing the user to select patient groups, but also allow re-training based on proprietary data.
            
            # merge all the features in a pandas DataFrame
            df_all = pd.DataFrame()
            for window_no in range(len(data_list)):
                signal_dict = self.preprocessing_ml(data_list[window_no])
                print(f'pass {window_no}')
                df_features = self.extract_features(signal_dict, ic_list[window_no])
                
                if reference_lr_list == []:
                    # patching with nans
                    df_features['foot'] = np.nan
                    warnings.warn('No reference system data provided. The ground truth left/right labels will not be returned and retraining of classification models will not return accurate results.')
                else:
                    df_features['foot'] = reference_lr_list[window_no]
                    
                df_all = pd.concat([df_all, df_features], ignore_index = True)
            
            df_labels = pd.DataFrame(df_all['foot'] == 'Right').astype(int)
            df_features = df_all.drop(columns = ['foot'])   
            
            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(df_features)
            y_train = df_labels.to_numpy()
            
            # TODO: should the default model be the one trained on all patients?
            current_dir_path = os.getcwd()
            if len(kwargs) == 0:
                # this defaults to a pretrained model, using data from all subjects
                model_name = 'uniss_unige_all_model.gz'
                model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                self.trained_model = joblib.load(model_path)
                
            if len(kwargs) > 0 and pretrained is True:
                if patient_group is None:
                    # defaulting to the model pretrained on all subjects
                    model_name = 'uniss_unige_all_model.gz'
                    model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                    self.trained_model = joblib.load(model_path)
                elif patient_group == 'HC':
                    model_name = 'uniss_unige_hc_model.gz'
                    model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                    self.trained_model = joblib.load(model_path)
                elif patient_group == 'MS':
                    model_name = 'uniss_unige_ms_model.gz'
                    model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                    self.trained_model = joblib.load(model_path)
                elif patient_group == 'PD':
                    model_name = 'uniss_unige_ms_model.gz'
                    model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                    self.trained_model = joblib.load(model_path)
                else:
                    raise NotImplementedError('The patient group you have selected is not supported.')
            else:
                # Starting model training
                if model is None:
                    # do grid search on all the available models
                    warnings.warn('No model selected. Grid search will be performed and the model achieving highest accuracy will be returned.')
                    
                    classifier_list = [
                        LR_Train(x_train, y_train, algo = "svm_lin"),
                        LR_Train(x_train, y_train, algo = "svm_rbf"),
                        LR_Train(x_train, y_train, algo = "knn"),
                        LR_Train(x_train, y_train, algo = "rfc")
                        ]

                    results_dict = []
                    acc_list = []
                    best_estimator_list = []
                    best_params_list = []

                    for classifier in classifier_list:
                        
                        # how can we turn off printing in this for loop?
                        
                        model = classifier.algo
                        grid = classifier.param_grid
                        
                        grid_cv = GridSearchCV(model, grid, cv=3, scoring = 'accuracy')
                        grid_cv.fit(x_train, y_train.ravel())
                        y_pred = grid_cv.predict(x_train)
                        
                        results_dict.append(grid_cv.cv_results_)
                        acc_list.append(metrics.accuracy_score(y_train, y_pred.ravel()))
                        best_estimator_list.append(grid_cv.best_estimator_)
                        best_params_list.append(grid_cv.best_params_)
                    
                    self.trained_model = best_estimator_list[acc_list.index(max(acc_list))]
                    
                elif model == 'svm_lin':
                    
                    classifier = LR_Train(x_train, y_train, algo = "svm_lin")
                    grid_cv = GridSearchCV(classifier.algo, classifier.param_grid, cv=3, scoring = 'accuracy')
                    grid_cv.fit(x_train, y_train.ravel())
                    self.trained_model = grid_cv.best_estimator_ 
                    
                elif model == 'svm_rbf':
                    classifier = LR_Train(x_train, y_train, algo = "svm_rbf")
                    grid_cv = GridSearchCV(classifier.algo, classifier.param_grid, cv=3, scoring = 'accuracy')
                    grid_cv.fit(x_train, y_train.ravel())
                    self.trained_model = grid_cv.best_estimator_ 
                    
                elif model == 'knn':
                    classifier = LR_Train(x_train, y_train, algo = "knn")
                    grid_cv = GridSearchCV(classifier.algo, classifier.param_grid, cv=3, scoring = 'accuracy')
                    grid_cv.fit(x_train, y_train.ravel())
                    self.trained_model = grid_cv.best_estimator_ 
                elif model == 'rfc':
                    classifier = LR_Train(x_train, y_train, algo = "rfc")
                    grid_cv = GridSearchCV(classifier.algo, classifier.param_grid, cv=3, scoring = 'accuracy')
                    grid_cv.fit(x_train, y_train.ravel())
                    self.trained_model = grid_cv.best_estimator_    

            # predictions can now be made using the predict method.
            # TODO: should we integrate a plotting utility here? probably not...
    
            
    def extract_sequence_data(self, imu_data, event_list, reference_data = False, verbose = 0):
        """
        Arguments:
            imu_data: - pandas DataFrame containing the raw IMU data.
            event_list: - a list of dictionaries containing segmentation (both 'linear' and 'curvilinear' walking) and IC detection time points.
            reference_data: - bool, indicating if reference system data is provided within event_list.
            # TODO: You might want to avoid this...
            verbose: - int, 0 for silent, 1 for printing.
        returns:
            data_list: - a list of pandas DataFrames containing the IMU data for each gait sequence.
            sequence_label: - a list of strings containing the label for each sequence, 'linear' or 'curvilinear'.
            ic_list: - a list of numpy arrays containing the IC indices for each sequence, 0 indexed, relative to the start of the corresponding sequence of 'linear' or 'curvilinear' gait sequence. 
            reference_lr_list: - a list of lists containing the left/right labels for each gait sequence.
        """
        
        if ~reference_data:
            warnings.warn("No reference system data provided. The ground truth left/right labels will not be returned.")
        
        signal = imu_data.loc[:, ['gyr_x', 'gyr_y', 'gyr_z']].to_numpy()
        
        sequence_label = []
        data_list = []
        ic_list = []
        reference_lr_list = []
        
        # how many gait sequences (GS) are detected?
        for gs in range(len(event_list)):
            if verbose:
                print(f"Gs number {gs} is being processed...")
            gs_start = np.array([event_list[gs]['Start']])
            gs_end = np.array([event_list[gs]['End']])
            turn_start = np.array(event_list[gs]['Turn_Start'])
            turn_end = np.array(event_list[gs]['Turn_End'])

            # remove duplicates from the turn array, if there are any consecutive turns, i.e. patch the consecutive turns.
            duplicate_start = np.isin(turn_start, turn_end)
            duplicate_end = np.isin(turn_end, turn_start)
            clean_turn_start = turn_start[~duplicate_start]
            clean_turn_end = turn_end[~duplicate_end]
            turn_array = np.sort(np.hstack((clean_turn_start, clean_turn_end)))
        
            # Check whether a GS starts or ends with a turn. If so, remove duplicates.
            if turn_start.shape[0] != 0:
                if gs_start[0] == turn_start[0]:
                    # print('GS starting with curvilinear walking')
                    case = 0
                else:
                    # print('GS stating with linear walking')
                    case = 1
            else:
                case = 1   
                
            alternating_array = np.append(np.concatenate((gs_start, turn_array)), gs_end)
            alternating_array = np.round(alternating_array * self.sampling_rate).astype(int)
            alternating_array = np.unique(alternating_array)
            ic_all = np.round(event_list[gs]['InitialContact_Event'] * self.sampling_rate).astype(int)
            if reference_data:
                lr_all = event_list[gs]['InitialContact_LeftRight']
                
            # what is this?
            n_label = 0
            
            for segments in range(len(alternating_array) - 1):
                if case == 0:
                    # GS starting with curvilinear walking
                    if segments % 2 == 1:
                        sequence_label.append('Linear')
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])        
                        # get the IC indices and labels from within this sequence window
                        ic_sequence =  ic_all[(ic_all >= sequence_start) & (ic_all <= sequence_end)] - sequence_start
                        ic_list.append(ic_sequence)
                        # get the ground truth labels, if reference system data is provided
                        if reference_data:
                            no_ics = len(ic_sequence)
                            lr = lr_all[n_label: n_label + no_ics]
                            n_label += no_ics
                            reference_lr_list.append(lr)       
                    else:
                        sequence_label.append('Curvilinear')
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])
                        # get the IC indices and labels from within this sequence window
                        ic_sequence =  ic_all[(ic_all >= sequence_start) & (ic_all <= sequence_end)] - sequence_start
                        ic_list.append(ic_sequence)
                        # get the ground truth labels, if reference system data is provided
                        if reference_data:
                            no_ics = len(ic_sequence)
                            lr = lr_all[n_label: n_label + no_ics]
                            n_label += no_ics
                            reference_lr_list.append(lr)
                elif case == 1:
                    # GS starting with linear walking
                    if segments % 2 == 0:
                        sequence_label.append('Linear')
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])        
                        # get the IC indices and labels from within this sequence window
                        ic_sequence =  ic_all[(ic_all >= sequence_start) & (ic_all <= sequence_end)] - sequence_start
                        ic_list.append(ic_sequence)
                        # get the ground truth labels, if reference system data is provided
                        if reference_data:
                            no_ics = len(ic_sequence)
                            lr = lr_all[n_label: n_label + no_ics]
                            n_label += no_ics
                            reference_lr_list.append(lr)       
                    else:
                        sequence_label.append('Curvilinear')
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])
                        # get the IC indices and labels from within this sequence window
                        ic_sequence =  ic_all[(ic_all >= sequence_start) & (ic_all <= sequence_end)] - sequence_start
                        ic_list.append(ic_sequence)
                        # get the ground truth labels, if reference system data is provided
                        if reference_data:
                            no_ics = len(ic_sequence)
                            lr = lr_all[n_label: n_label + no_ics]
                            n_label += no_ics
                            reference_lr_list.append(lr)
                            
        return data_list, sequence_label, ic_list, reference_lr_list  
    
    def McCamley(self, wb, ic_list):
        """
        Arguments:
            imu_data
            ic_list: - 
            sampling_rate: - 
        returns:
        """
        return 0
    
    def ML_approaches(self, wb, ic_list):
        return 0
    
    def preprocessing_ml(self, gyr_3d):
        gyr_v = gyr_3d[:, 0]
        gyr_ap = gyr_3d[:, 2]
        
        lower_band = 0.5
        upper_band = 2
        gyr_v_filtered = _butter_bandpass_filter(gyr_v, 
                                                 lower_bound = lower_band,
                                                 upper_bound = upper_band,
                                                 sampling_rate_hz = self.sampling_rate)
        
        gyr_ap_filtered = _butter_bandpass_filter(gyr_ap, 
                                                   lower_bound = lower_band,
                                                   upper_bound = upper_band,
                                                   sampling_rate_hz = self.sampling_rate)
        
        # compute the first derivative
        gyr_v_diff = np.diff(gyr_v_filtered)
        gyr_ap_diff = np.diff(gyr_ap_filtered)
        
        # compute the second derivative
        gyr_v_diff_2 = np.diff(gyr_v_diff)
        gyr_ap_diff_2 = np.diff(gyr_ap_diff)
        
        # output everything into a dictionary
        signal_dict = {}
        signal_dict['v_filtered'] = gyr_v_filtered
        signal_dict['v_gradient'] = gyr_v_diff
        signal_dict['v_diff_2'] = gyr_v_diff_2
        
        signal_dict['ap_filtered'] = gyr_ap_filtered
        signal_dict['ap_gradient'] = gyr_ap_diff
        signal_dict['ap_diff_2'] = gyr_ap_diff_2
        
        return signal_dict
        
    def extract_features(self, signal_dict, ic_list):
        print(len(signal_dict))
        print(len(ic_list))
        """
        Feature extraction: extract the respective values at the IC samples from the six signals.
        """ 
        if ic_list[-1] >= len(signal_dict['v_filtered']):
          # shift the last IC by 3 samples to make the second derivative work
          ic_list[-1] -= 3
        
        print(len(signal_dict['v_filtered']))
        print(ic_list[-1])
        
        # get the feature values @ IC time stamps:
        feature_dict = {}
        feature_dict['v_filtered'] = signal_dict['v_filtered'][ic_list]
        feature_dict['v_gradient'] = signal_dict['v_gradient'][ic_list]
        feature_dict["v_diff_2"] = signal_dict["v_diff_2"][ic_list]

        feature_dict["ap_filtered"] = signal_dict["ap_filtered"][ic_list]
        feature_dict["ap_gradient"] = signal_dict["ap_gradient"][ic_list]
        feature_dict["ap_diff_2"] = signal_dict["ap_diff_2"][ic_list]

        feature_df = pd.DataFrame(feature_dict)    
        
        return feature_df  
    