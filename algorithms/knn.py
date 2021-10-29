import warnings
warnings.simplefilter("ignore", UserWarning)
from base_model import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class KNearestNeighbors(BaseModel):
    def __init__(self, X_train: np.array, y_train: np.array, param_grid: dict = None) -> None:
        if param_grid is None:  
            param_grid = param_grid ={'n_neighbors': range(5,30), 'p': [1, 2, 3, 4, np.inf]}
        self.classifier = KNeighborsClassifier()
        BaseModel.__init__(self, X_train, y_train, param_grid, random_state = False)
    
    def model_tune(self) -> dict:
        model_grid = GridSearchCV(self.classifier, self.param_grid, refit = True, verbose = 2, n_jobs = -1)
        classifier_best = model_grid.fit(self.X_train, self.y_train)
        return classifier_best.best_params_
    
    def best_model(self):
        self.best_params
        classifier = KNeighborsClassifier(**self.best_params)
        return classifier.fit(self.X_train, self.y_train)