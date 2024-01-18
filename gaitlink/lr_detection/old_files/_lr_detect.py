import numpy as np
import pandas as pd
import warnings
import joblib
import os

from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from _utils import _butter_bandpass_filter, _butter_lowpass_filter
from _ml_param_grid import ParamGrid

from gaitlink.lr_detection.base import BaseLRDetector, base_lr_detector_docfiller

class LR_detector(BaseLRDetector):
    """
    Left/Right foot detector.
    
    Arguments:
    -------------------------------------
        samplig_rate: - the sampling rate of the IMU sensor.
        approach: - McCamley or ML.
    
    -------------------------------------
    Note: an optional 'McCamley-original' implementation is also available using a lowpass filter. The updated and default version used in the TVS is using bandpass filtering instead.
    """

    def __init__(self,
                sampling_rate: int,
                approach: str = 'McCamley'):  
        self.sampling_rate = sampling_rate
        self.approach = approach
    
    # TODO: think of better names for these variables...
    @base_lr_detector_docfiller
    def predict(self,
               imu_data: pd.DataFrame,
               event_list: list,
               reference_data: bool = False, 
               **kwargs):
        
        # TODO: you should also be able to fit directly a sklearn classifier, without having to do any training at all. This is to be added soon.
        
        """
        Predicts the left/right labels for the corresponding ICs.
        
        Arguments:
        -------------------------------------
            imu_data: - pandas DataFrame containing the raw IMU data.
            
            event_list: - a list of dictionaries containing the gait sequence reference data. From here, we need the gait sequence start and end times, turning points, as well as the the time points associated to ICs.
            
            reference_data: bool, whether reference system data is provided within event_list.
            
        McCamley-based approach optional arguments:
        -------------------------------------
        axis_config: - str, the axis configuration to use for the McCamley-based approach. This can be 'Vertical'/'V'/'Yaw'/'Y', 'Anterior-Posterior'/'AP'/'Roll'/'R' or 'Combined'/'C'. If this argument is not provided, the default is to use the 'Combined' configuration.
        
            
        ML-based approach optional arguments:
        -------------------------------------
            pretrained: - bool, whether to use pretrained models. Default is True. If pretrained is False, then the detector will automatically enter into training mode - this is done according to the classifier option.
            
            patient_group: - str, the patient group to use for pretrained models. This can be 'HC', 'MS', 'PD', 'CH' or 'ST'. If this argument is not provided, the default is to use a pretrained model trained on all patients.
            
            classifier_name: - str, the name of the classifier to use for training. Options are 'SVM_LIN' (linear support vector machine), 'SVM_RBF' (radial-basis function support vector machine), 'KNN' (k-nearest neighbors), 'RFC' (random forest classifier). If this argument is not provided, the default is to perform grid search on all available classifiers and return the one with the highest accuracy.
            
            grid: - dict, a user-specified grid of parameters to used for training. If this argument is not provided, default options for each specified classifier are used.
            
            classifier: sklearn classifier, a user-specified classifier. This will overwrite all the options above.
            
        Returns:
        -------------------------------------
            lr_list: - a list of lists of labeled ICs, corresponding to the input gait sequences.
            """
        # Firstly, extract the the gait data sequences from the IMU data. This is further broken into smaller 'Linear' and 'Curvilinear' walking segments. Information about the type of walking is kept in 'walking_type_label'. 'sequence_no' tracks the gait sequence number, which is then used to feedback the predicted labels from the smaller 'linear_walking' and 'curvilinear_walking' segments back to the corresponding gait sequences.
        data_list, walking_type_label, sequence_no,  ic_list, reference_lr_list = self.extract_sequence_data(imu_data, event_list, reference_data)
        
        verbose = kwargs.get('verbose')
        
        
        
        
        
        # ---------------------------------------------------------------------
        #                       McCamley-based approach
        # ---------------------------------------------------------------------
        
        
        
        
        
        if self.approach.upper() == 'MCCAMLEY' or self.approach.upper() == 'MCCAMLEY-ORIGINAL':
            
            axis_config = kwargs.get('axis_config')
            if axis_config is None:
                axis_config = "Combined"
            elif axis_config.upper() not in ['VERTICAL', 'V', 'YAW', 'Y', 'ANTERIOR-POSTERIOR', 'AP', 'ROLL', 'R', 'COMBINED', 'C']:
                raise NotImplementedError("The axis configuration you have selected is not supported. Please select between 'Vertical', 'Anterior-Posterior' or 'Combined'.")
            
            segment_lr_list = []
            
            for segment in range(len(data_list)):
                if axis_config.upper() in ['VERTICAL', 'V', 'YAW', 'Y']:
                    signal = data_list[segment][:, 0]
                elif axis_config.upper() in ['ANTERIOR-POSTERIOR', 'AP', 'ROLL', 'R']:
                    signal = data_list[segment][:, 2] * - 1 # invert the sign here
                elif axis_config.upper() in ['COMBINED', 'C']:
                    signal = (data_list[segment][:, 2] * - 1) + data_list[segment][:, 0]
                
                if ic_list[segment][-1] >= len(signal):
                    # print('edge case 1')
                    ic_list[segment][-1] = len(signal) - 1

                if len(ic_list[segment]) >= 2 and ic_list[segment][-2] >= len(signal):
                    # print('edge case 2')
                    ic_list[segment][-2] = len(signal) - 2  
                
                # deprecated original version: subtract mean and use lowpass filter: this could be applied in case an "-original" is put into the algorithm string, but was not used for TVS
                if len(self.approach.split("-")) >= 2 and "ORIGINAL" in self.approach.split("-")[1].upper():
                    signal = signal - np.mean(signal)
                    cutoff_frq = 2
                    signal_filtered = _butter_lowpass_filter(signal, cutoff_frq, self.sampling_rate)
                else:
                    # Now using a bandpass filter instead of subtracting the mean to make it more robust for turnings!
                    lower_band = 0.5
                    upper_band = 2
                    signal_filtered = _butter_bandpass_filter(signal, lower_band, upper_band, self.sampling_rate)
                # left and right is determined according to the sign of the signal at the IC instance in the filtered signal
                ic_left_right = np.where(signal_filtered[ic_list[segment]] <= 0, "Left", "Right")
                segment_lr_list.append(ic_left_right)
            
            self.y_pred_labels = np.hstack(segment_lr_list)
            if verbose == 1 or verbose == True:
                print('Predicting L/R Labels...')
            lr_list = self.feedback_labels(sequence_no, ic_list, self.y_pred_labels)
            if verbose == 1 or verbose == True:
                print('Done!')
            
            return lr_list
        
        # ML-based approach args
        pretrained = kwargs.get('pretrained') 
        patient_group = kwargs.get('patient_group')
        classifier_name = kwargs.get('classifier_name')
        grid = kwargs.get('grid')
        classifier = kwargs.get('classifier')
        
        if pretrained is None and patient_group is not None:
            # default to a pretrained model if pretrained arg in not provided
            pretrained = True
        
        
        
        
        
        # ---------------------------------------------------------------------
        #                       ML-based approach
        # ---------------------------------------------------------------------
        
        
        
        
        
        if self.approach.upper() == 'ML':
            # merge all the features in a pandas DataFrame
            df_all = pd.DataFrame()
            for segment in range(len(data_list)):
                signal_dict = self.preprocessing_ml(data_list[segment])
                df_features = self.extract_features(signal_dict, ic_list[segment])
                
                if reference_lr_list == []:
                    # patching with nans
                    df_features['foot'] = np.nan
                    warnings.warn('No reference system data provided. The ground truth left/right labels will not be returned and retraining of classification models will not return accurate results.')
                else:
                    df_features['foot'] = reference_lr_list[segment]
                    
                df_all = pd.concat([df_all, df_features], ignore_index = True)
            
            df_labels = pd.DataFrame(df_all['foot'] == 'Right').astype(int)
            df_features = df_all.drop(columns = ['foot'])   
            
            scaler = MinMaxScaler()
            self.x_train = scaler.fit_transform(df_features)
            self.y_train = df_labels.to_numpy() # left = 0, right = 1
            

            if classifier is not None:
                # using a user-specified classifier
                
                assert isinstance(classifier, BaseEstimator), "The classifier you have provided is not a valid sklearn classifier."
                
                classifier.fit(self.x_train, self.y_train.ravel())
                self.trained_model = classifier
                
            else:
                current_dir_path = os.getcwd()
                if len(kwargs) == 0:
                    # this defaults to a pretrained model, using data from all subjects
                    # TODO: should the default model be the one trained on all patients?
                    if verbose == 1 or verbose == True:
                        print('No extra arguments provided...using default pretrained models')
                    model_name = 'uniss_unige_all_model.gz'
                    model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                    self.trained_model = joblib.load(model_path)
                    
                # TODO: This is not very clever... Think of a better logic here.
                
                elif len(kwargs) > 0 and pretrained is True:
                    if verbose == 1 or verbose == True:
                        print('Using pretrained models...')
                    
                    # TODO: you should also import the scalers here!!!
                    
                    if patient_group.upper() is None or patient_group.upper() == 'ALL':
                        print('Defaulting to a general model...')
                        model_name = 'uniss_unige_all_model.gz'
                        model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                        self.trained_model = joblib.load(model_path)
                    elif patient_group.upper() == 'HC':
                        model_name = 'uniss_unige_hc_model.gz'
                        model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                        self.trained_model = joblib.load(model_path)
                    elif patient_group.upper() == 'MS':
                        model_name = 'msproject_ms_model.gz'
                        model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                        self.trained_model = joblib.load(model_path)
                    elif patient_group.upper() == 'PD':
                        model_name = 'uniss_unige_pd_model.gz'
                        model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                        self.trained_model = joblib.load(model_path)
                    elif patient_group.upper() == 'CH':
                        model_name = 'uniss_unige_ch_model.gz'
                        model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                        self.trained_model = joblib.load(model_path)
                    elif patient_group.upper() == 'ST':
                        model_name = 'uniss_unige_st_model.gz'
                        model_path = os.path.join(current_dir_path, 'pretrained_models', model_name)
                        self.trained_model = joblib.load(model_path)
                    else:
                        raise NotImplementedError('The patient group you have selected is not supported.')
                elif len(kwargs) > 0 and pretrained is False:
                    # Starting model training
                    if verbose == 1 or verbose == True:
                        print('Starting model training...')
                    if classifier_name is None:
                        # do grid search on all the available models
                        if verbose == 1 or verbose == True:
                            print('No model selected. Grid search will be performed and the model achieving highest accuracy will be returned.')
                        
                        classifier_list = [
                            ParamGrid(self.x_train, self.y_train, algo = "svm_lin"),
                            ParamGrid(self.x_train, self.y_train, algo = "svm_rbf"),
                            ParamGrid(self.x_train, self.y_train, algo = "knn"),
                            ParamGrid(self.x_train, self.y_train, algo = "rfc")
                            ]
                        
                        # TODO: Ideally, these should be adjusted externally...
                        classifier_list[2].param_grid['n_neighbors'] = [2, 5, 7, 10]
                        classifier_list[3].param_grid['n_estimators'] = [100]
                        classifier_list[3].param_grid['max_depth'] = [None, 10, 20]

                        results_dict = []
                        acc_list = []
                        best_estimator_list = []
                        best_params_list = []

                        for clf in classifier_list:
                            
                            # TODO: can we turn off printing in this for loop?
                            
                            model = clf.algo
                            grid = clf.param_grid
                            
                            grid_cv = GridSearchCV(model, grid, cv=3, scoring = 'accuracy')
                            grid_cv.fit(self.x_train, self.y_train.ravel())
                            y_pred_temp = grid_cv.predict(self.x_train)
                            
                            results_dict.append(grid_cv.cv_results_)
                            acc_list.append(metrics.accuracy_score(self.y_train, y_pred_temp.ravel()))
                            best_estimator_list.append(grid_cv.best_estimator_)
                            best_params_list.append(grid_cv.best_params_)
                        
                        self.trained_model = best_estimator_list[acc_list.index(max(acc_list))]
                        
                    elif classifier_name.upper() == 'SVM_LIN':
                        if verbose == 1 or verbose == True:
                            print('Training SVM model with linear kernel...')
                        if grid is None:
                            # default grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "svm_lin")
                            self.param_grid = clf.param_grid
                        else:
                            # user-specified grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "svm_lin", custom_grid = grid)
                            self.param_grid = grid
                            
                        grid_cv = GridSearchCV(clf.algo, self.param_grid, cv=3, scoring = 'accuracy')
                        grid_cv.fit(self.x_train, self.y_train.ravel())
                        self.trained_model = grid_cv.best_estimator_
                        if verbose == 1 or verbose == True:
                            print('Training complete!')

                    elif classifier_name.upper() == 'SVM_RBF':
                        if verbose == 1 or verbose == True:
                            print('Training SVM model with rbf kernel...')
                        if grid is None:
                            # default grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "svm_rbf")
                            self.param_grid = clf.param_grid
                        else:
                            # user-specified grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "svm_rbf", custom_grid = grid)
                            self.param_grid = grid
                        
                        grid_cv = GridSearchCV(clf.algo, self.param_grid, cv = 3, scoring = 'accuracy')
                        grid_cv.fit(self.x_train, self.y_train.ravel())
                        
                        self.trained_model = grid_cv.best_estimator_
                        if verbose == 1 or verbose == True:
                            print('Training complete!')
                        
                    elif classifier_name.upper() == 'KNN':
                        if verbose == 1 or verbose == True:
                            print('Training KNN model...')
                        if grid is None:
                            # default grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "knn")
                            self.param_grid = clf.param_grid
                        else:
                            # user-specified grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "knn", custom_grid = grid)
                            self.param_grid = grid

                        grid_cv = GridSearchCV(clf.algo, self.param_grid, cv=3, scoring = 'accuracy')
                        grid_cv.fit(self.x_train, self.y_train.ravel())
                        self.trained_model = grid_cv.best_estimator_
                        if verbose == 1 or verbose == True:
                            print('Training complete!')
                        
                    elif classifier_name.upper() == 'RFC':
                        if verbose == 1 or verbose == True:
                            print('Training Random Forest Classifier...')
                        if grid is None:
                            # default grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "rfc")
                            self.param_grid = clf.param_grid
                        else:
                            # user-specified grid
                            clf = ParamGrid(self.x_train, self.y_train, algo = "rfc", custom_grid = grid)
                            self.param_grid = grid

                        grid_cv = GridSearchCV(clf.algo, self.param_grid, cv=3, scoring = 'accuracy')
                        grid_cv.fit(self.x_train, self.y_train.ravel())
                        self.trained_model = grid_cv.best_estimator_
                        if verbose == 1 or verbose == True:
                            print('Training complete!')

            # predictions can now be made using the predict method.
            self.y_pred = self.trained_model.predict(self.x_train)
            self.y_pred_labels = np.array(['Left' if item == 0 else 'Right' for item in self.y_pred])
            
            # TODO: add a plotting utility function here?
            # TODO: also, some performance metrics would be nice...
            if verbose == 1 or verbose == True:
                print('Predicting L/R Labels...')
            lr_list = self.feedback_labels(sequence_no, ic_list, self.y_pred_labels)
            if verbose == 1 or verbose == True:
                print('Done!')
            
            return lr_list
        
        # TODO: feed this back to the standard Mobilise Data Structure.


    @base_lr_detector_docfiller          
    def extract_sequence_data(self, imu_data, event_list, reference_data = False, verbose = 0):
        """
        Data extraction utility function: extract the respective gait sequences from the IMU data.
        
        # TODO: maybe you should not use sequences for 'linear' and 'curvilinear' walking types... A GS can have both.
        
        Arguments:
            imu_data: - pd.DataFrame, the raw IMU data.
            
            event_list: - a list of dictionaries containing segmentation (both      'linear' and 'curvilinear' walking) and IC detection time points.
            reference_data: - bool, indicating if reference system data is provided within event_list.
            
            verbose: - int, 0 for silent, 1 for printing.
            
        Returns:
            data_list: - a list of pandas DataFrames containing the IMU data for each gait sequence.
            
            walking_type_label: - a list of strings containing the label for each sequence, 'linear' or 'curvilinear' along with the corresponding gait sequence number.
            
            sequence_number: the corresponding gait sequence number for each entry  in data_list.
            
            ic_list: - a list of numpy arrays containing the IC indices for each data_list entry, 0 indexed, relative to the start of the corresponding sequence of 'linear' or 'curvilinear' gait sequence. 
            
            reference_lr_list: - a list of lists containing the left/right labels to accompany ic_list.
        """

        if reference_data is False:
            warnings.warn("No reference system data provided. The ground truth left/right labels will not be returned.")
        
        signal = imu_data.loc[:, ['gyr_x', 'gyr_y', 'gyr_z']].to_numpy()
        
        walking_type_label = []
        sequence_no = []
        data_list = []
        ic_list = []
        reference_lr_list = []
        
        # how many gait sequences (GS) are detected?
        for gs in range(len(event_list)):
            if verbose == 1 or verbose == True:
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
                        walking_type_label.append(f'Linear gs #{gs}')
                        sequence_no.append(gs)
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])        
                        # get the IC indices and labels from within this segment
                        ic_sequence =  ic_all[(ic_all >= sequence_start) & (ic_all <= sequence_end)] - sequence_start
                        ic_list.append(ic_sequence)
                        # get the ground truth labels, if reference system data is provided
                        if reference_data:
                            no_ics = len(ic_sequence)
                            lr = lr_all[n_label: n_label + no_ics]
                            n_label += no_ics
                            reference_lr_list.append(lr)       
                    else:
                        walking_type_label.append(f'Curvilinear gs #{gs}')
                        sequence_no.append(gs)
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])
                        # get the IC indices and labels from within this segment
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
                        walking_type_label.append(f'Linear gs #{gs}')
                        sequence_no.append(gs)
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])        
                        # get the IC indices and labels from within this segment
                        ic_sequence =  ic_all[(ic_all >= sequence_start) & (ic_all <= sequence_end)] - sequence_start
                        ic_list.append(ic_sequence)
                        # get the ground truth labels, if reference system data is provided
                        if reference_data:
                            no_ics = len(ic_sequence)
                            lr = lr_all[n_label: n_label + no_ics]
                            n_label += no_ics
                            reference_lr_list.append(lr)       
                    else:
                        walking_type_label.append(f'Curvilinear gs #{gs}')
                        sequence_no.append(gs)
                        sequence_start = alternating_array[segments]
                        sequence_end = alternating_array[segments + 1]
                        data_list.append(signal[sequence_start : sequence_end, :])
                        # get the IC indices and labels from within this segment
                        ic_sequence =  ic_all[(ic_all >= sequence_start) & (ic_all <= sequence_end)] - sequence_start
                        ic_list.append(ic_sequence)
                        # get the ground truth labels, if reference system data is provided
                        if reference_data:
                            no_ics = len(ic_sequence)
                            lr = lr_all[n_label: n_label + no_ics]
                            n_label += no_ics
                            reference_lr_list.append(lr)
                            
        return data_list, walking_type_label, sequence_no, ic_list, reference_lr_list  
    
    
    @base_lr_detector_docfiller  
    def preprocessing_ml(self, gyr_3d):
        """"
        Preprocessing utility function for the ML prediction approach.
        """
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
     
    @base_lr_detector_docfiller     
    def extract_features(self, signal_dict, ic_list):
        """
        Feature extraction utility function: extract the respective values at the IC samples from the six signals.
        """ 
        if ic_list[-1] >= len(signal_dict['v_filtered']):
        # shift the last IC by 3 samples to make the second derivative work
            ic_list[-1] -= 3
        
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
    
    @base_lr_detector_docfiller  
    def feedback_labels(self, sequence_no, ic_list, y_pred_labels):
        # TODO: finish the docs for this function.
        """
        Utility function for feeding back the predicted L/R labels to the corresponding gait sequences.
    
        Arguments:
            sequence_no: -
            ic_list: -
            y_pred_labels: -
            
        Returns:
            lr_list: - a list of lists of labeled ICs, corresponding to the input gait sequences.
        """
        # TODO: Think of better names for these variables...
        
        array_to_cut = [len(ic_list[i]) for i in range(len(ic_list))]

        for i in range(1, len(array_to_cut)):
            array_to_cut[i] += array_to_cut[i - 1]
            
        xx = np.arange(0, array_to_cut[-1])
        
        slices = {}
        current_start = 0

        for i in range(len(array_to_cut)):
            start = current_start
            end = array_to_cut[i]
            slice_number = sequence_no[i]

            if slice_number not in slices:
                slices[slice_number] = []

            slice_elements = xx[start:end]
            slices[slice_number].extend(slice_elements)

            current_start = array_to_cut[i]
        result_slices = [slices[key] for key in sorted(slices.keys())]
        lr_list = [[y_pred_labels[i] for i in indices] for indices in result_slices]

        return lr_list
