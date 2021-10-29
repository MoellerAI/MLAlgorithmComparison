import warnings
warnings.simplefilter("ignore", UserWarning)

from base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

class RandomForest(BaseModel):
    def __init__(self, X_train: np.array, y_train: np.array, param_grid: dict = None) -> None:
        if param_grid is None:
            #self.param_grid = {'max_depth':range(2,15), 'criterion': ['gini', 'entropy'], 'min_samples_split': range(2,20), 'min_samples_leaf': range(1,10), 'n_estimators': range(250, 1500, 250)}
            param_grid = {'criterion': ['gini', 'entropy'], 'n_estimators': list(range(50, 1500, 50))}
        self.classifier = RandomForestClassifier(random_state = 0, class_weight = 'balanced')
        BaseModel.__init__(self, X_train, y_train, param_grid)
    
    def model_tune(self) -> dict:
        model_grid = GridSearchCV(self.classifier, self.param_grid, n_jobs = -1, return_train_score = True, refit = True, verbose = 2)
        classifier_best = model_grid.fit(self.X_train, self.y_train)
        return classifier_best.best_params_
    
    def best_model(self):
        self.best_params
        classifier = RandomForestClassifier(**self.best_params)
        return classifier.fit(self.X_train, self.y_train)