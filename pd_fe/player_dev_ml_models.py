from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def ml_stack_model_w_grid_search(X_train, Y_train):

    stack_model = StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(random_state = 42)),
            ('gb', HistGradientBoostingRegressor(random_state = 42))
        ],
        final_estimator=GradientBoostingRegressor(random_state = 42),
        passthrough=True,
        cv=5
    )

    param_grid = {
        'rf__n_estimators': [50, 100],
        'rf__max_depth': [5, 10],
        'gb__max_iter': [100, 200],
        'final_estimator__n_estimators': [50, 100]
    }

    grid_search = GridSearchCV(
        estimator=stack_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, Y_train)

    print("Best parameters:", grid_search.best_params_)
    best_stack_model = grid_search.best_estimator_

    return best_stack_model