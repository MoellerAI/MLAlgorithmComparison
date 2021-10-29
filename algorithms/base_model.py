import warnings
warnings.simplefilter("ignore", UserWarning)
from sklearn.metrics import accuracy_score
import numpy as np

class BaseModel(object):
    def __init__(self, X_train: np.array, y_train: np.array, param_grid: dict = None, random_state: bool = True) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.param_grid = param_grid
        self.best_params = self.model_tune()
        if random_state:
            self.best_params['random_state'] = 0
        self.model = self.best_model()

    def model_tune(self) -> dict:
        pass
    
    def best_model(self):
        pass
    
    def predict(self, X_test: np.array) -> list:
        return self.model.predict(X_test)

    def get_accuracy(self, X_test: np.array, y_test: np.array) -> float:
        y_pred = self.predict(X_test)
        return accuracy_score(y_pred, y_test)