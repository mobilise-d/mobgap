from tpcp import OptimizablePipeline, OptimizableParameter
from tpcp.optimize import GridSearchCV
from gaitlink.lrml._utils import extract_ref_data
from gaitlink.lrml import UllrichLRDetection



class UllrichLROptimizer():
    """
    This class is used to optimize the parameters of a UllrichLRDetection pipeline using GridSearchCV.
    """
    def __init__(self, pipeline, parameter_grid, cross_val_splits = 2):
        """
        Initializes the optimizer with a pipeline, parameter grid, and number of cross-validation splits.

        Args:
            pipeline (LROptiPipeline): The pipeline to be optimized.
            parameter_grid (dict): The grid of parameters to search over.
            cross_val_splits (int, optional): The number of cross-validation splits. Defaults to 2.
        """
        pipeline: LROptiPipeline

        self.pipeline = pipeline
        self.parameter_grid = parameter_grid
        self.cross_val_splits = cross_val_splits

    def optimize(self, dataset, scoring_function):
        """
        Optimizes the pipeline's parameters using GridSearchCV.

        Args:
            dataset (gaitlink.data._example_data.LabExampleDataset): The dataset to use for optimization.
            scoring_function (callable): The scoring function to use for optimization.

        Returns:
            dict: The results of the optimization.
        """

        self.optimization_results = GridSearchCV(self.pipeline,
                                    self.parameter_grid,
                                    scoring = scoring_function,
                                    cv = self.cross_val_splits,
                                    return_optimized = "accuracy").optimize(dataset)

        return self.optimization_results
        # return self.pipeline, self.optimization_results

    

class LROptiPipeline(OptimizablePipeline):
    # This is a trick we use internally to check that the optimization is not doing something strange.
    # Only the model is allowed to change. If other things change, we get an error by default.
    """
    This class represents a pipeline for UllrichLRDetection that can be optimized.
    """
    algo__model: OptimizableParameter
    algo_with_results_: list[UllrichLRDetection]

    def __init__(self, algo):
        """
        Initializes the pipeline with an algorithm.

        Args:
            algo (UllrichLRDetection): The algorithm to use in the pipeline.
        """
        self.algo = algo
        

    @property
    def ic_lr_(self):
        """
        Returns the IC_LR results of the algorithm.

        Returns:
            List of pd.DataFrame: The IC_LR results. <- THIS WILL NEED TO BE CHANGED.
        """

        # unpack the predictions from self.algo_with_results_
        return [prediction_per_gs.ic_lr for prediction_per_gs in self.algo_with_results_]
    
    def run(self, datapoint):
        """
        Runs the pipeline on a datapoint.

        Args:
            datapoint (gaitlink.data._example_data.LabExampleDataset): The datapoint to run the pipeline on.

        Returns:
            UllrichLRDetection: The algorithm with results.
        """
        sampling_rate_hz = datapoint.sampling_rate_hz

        # Firstly, we need to extract the data_list and ic_list from the datapoint. Note, that datapoint can contain multiple GSs.
        # We can use the extract_ref_data utility function for this.
        data_list, ic_list, _ = extract_ref_data(datapoint)

        self.algo_with_results_ = []
        # TODO: add a loop to handle multiple GS.
        for gs in range(len(data_list)):
            self.algo_with_results_.append(self.algo.clone().detect(data_list[gs], ic_list[gs], sampling_rate_hz))
        
        return self
    
    def self_optimize(self, dataset):
        """
        Fits the algorithm to the entire dataset.

        This method extracts the data_list, ic_list, and label_list from each datapoint in the dataset, and then calls the self_optimize method of the algorithm with these lists.

        Args:
            dataset (gaitlink.data._example_data.LabExampleDataset): The dataset to fit the algorithm to. Each element of the list is a datapoint.

        Returns:
            LROptiPipeline: The pipeline itself. This allows for method chaining.
        """
        all_gs_data = []
        all_ics = []
        all_labels = []

        for datapoint in dataset:
            
            # TODO: this is a temporary fix, as HA, with participant_id = 002 (TimeMeasure1, Test5, Trial1) is problematic
            try:
                data_list, ic_list, label_list = extract_ref_data(datapoint)

                all_gs_data.extend(data_list)
                all_ics.extend(ic_list)
                all_labels.extend(label_list)
            except:
                pass
        
        # No cloning here -> we actually want to modify the object
        # this is also going to fit the model and the scaler to the training data.
        self.algo.self_optimize(all_gs_data, all_ics, all_labels)

        # TODO: We should think of a better method to catch any scalers that were not fit to the training data before the optimizer is triggered.

        return self