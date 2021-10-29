import warnings
warnings.simplefilter("ignore", UserWarning)

from base_model import BaseModel
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np

class SupportVectorMachines(BaseModel):
    def __init__(self, X_train: np.array, y_train: np.array, param_grid: dict = None) -> None:
        if param_grid is None:
            param_grid = param_grid = {'C': [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'kernel': ('linear', 'rbf')}
        self.classifier = svm.SVC(random_state = 0)
        BaseModel.__init__(self, X_train, y_train, param_grid)
    
    def model_tune(self) -> dict:
        model_grid = GridSearchCV(self.classifier, self.param_grid, verbose=5, n_jobs = -1, return_train_score = True, refit = True)
        classifier_best = model_grid.fit(self.X_train, self.y_train)
        return classifier_best.best_params_
    
    def best_model(self):
        self.best_params
        classifier = svm.SVC(**self.best_params)
        return classifier.fit(self.X_train, self.y_train)