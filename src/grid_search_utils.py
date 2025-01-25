"""
This module contains grid search utility functions for the stock model project.
"""
from sklearn.model_selection import GridSearchCV

def perform_grid_search(model, param_grid, x_train, y_train, cv=5):
    """
    Perform GridSearchCV and return the best model and parameters.
    """
    grid_search = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=cv, verbose=1)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
