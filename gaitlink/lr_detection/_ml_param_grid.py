import numpy as np

# import the classifiers:
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier

class ParamGrid():
    def __init__(self,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 algo: str = "svm_lin",
                 **kwargs):
        
        self.x_train = x_train
        self.y_train = y_train
        
        custom_grid = kwargs.get('custom_grid')
        
        # when custom_grid is not provided, the default param grid is used.
        if custom_grid is None:
            if algo.upper() == "SVM_LIN":
                self.algo = svm.SVC()
                self.param_grid = self.get_param_grid_svm_lin()
            elif algo.upper() == "SVM_RBF":
                self.algo = svm.SVC()
                self.param_grid = self.get_param_grid_svm_rbf()
            elif algo.upper() == "KNN":
                self.algo = neighbors.KNeighborsClassifier()
                self.param_grid = self.get_param_grid_knn()
            elif algo.upper() == "RFC":
                self.algo = RandomForestClassifier()
                self.param_grid = self.get_param_grid_rfc()
            else:
                raise NotImplementedError("The algorithm you specified is not supported.")
        
        # switching to the provided grid.
        else:
            if algo.upper() == "SVM_LIN":
                self.algo = svm.SVC()
                self.param_grid = custom_grid
            elif algo.upper() == "SVM_RBF":
                self.algo = svm.SVC()
                self.param_grid = custom_grid
            elif algo.upper() == "KNN":
                self.algo = neighbors.KNeighborsClassifier()
                self.param_grid = custom_grid
            elif algo.upper() == "RFC":
                self.algo = RandomForestClassifier()
                self.param_grid = custom_grid
            
    def get_param_grid_svm_lin(self):
        # SVM linear
        # here, the parameter C is analogous to (the inverse of) a regularization coefficient because it controls the trade-off between training errors and model complexity.
        param_grid_svm_lin = {'C': [0.01, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 5, 7.5, 10, 25, 50, 100, 500, 1000, 5000, 10000],
            'kernel': ['linear']}
        return param_grid_svm_lin

    def get_param_grid_svm_rbf(self):
        # SVM rbf
        param_grid_svm_rbf = {'C': [1, 2.5, 5, 7.5, 10, 25, 50, 100, 500, 1000, 5000, 10000],
            'gamma': [2, 1.5, 1, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']}
        
        return param_grid_svm_rbf

    def get_param_grid_knn(self):
        # KNN
        param_grid_knn = {'n_neighbors': [2,5,7,10,15,20,30,40,60,100]}     
        
        return param_grid_knn

    def get_param_grid_rfc(self): # rfc = random forest classifier
        # Random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]

        # Create the random grid
        param_grid_rfc = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        # 'max_features': max_features,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap
                        }

        return param_grid_rfc