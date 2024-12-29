import argparse
from time import time
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt

import numpy as np
import cupy as cp

import xgboost as xgb

from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, PredictionErrorDisplay
from sklearn.model_selection import cross_val_score

"""https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_rmsle.py"""


def native_rmse(dtrain: xgb.DMatrix,
                dtest: xgb.DMatrix) -> Dict[str, Dict[str, List[float]]]:
    '''Train using native implementation of Root Mean Squared Loss.'''
    print('Squared Error')
    squared_error = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'seed': kSeed
    }
    start = time()
    results: Dict[str, Dict[str, List[float]]] = {}
    xgb.train(squared_error,
              dtrain=dtrain,
              num_boost_round=kBoostRound,
              evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
              evals_result=results)
    print('Finished Squared Error in:', time() - start, '\n')
    return results


def native_rmsle(dtrain: xgb.DMatrix,
                 dtest: xgb.DMatrix) -> Dict[str, Dict[str, List[float]]]:
    '''Train using native implementation of Squared Log Error.'''
    print('Squared Log Error')
    results: Dict[str, Dict[str, List[float]]] = {}
    squared_log_error = {
        'objective': 'reg:squaredlogerror',
        'eval_metric': 'rmsle',
        'tree_method': 'hist',
        'seed': kSeed
    }
    start = time()
    xgb.train(squared_log_error,
              dtrain=dtrain,
              num_boost_round=kBoostRound,
              evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
              evals_result=results)
    print('Finished Squared Log Error in:', time() - start)
    return results


def py_rmsle(dtrain: xgb.DMatrix, dtest: xgb.DMatrix) -> Dict:
    '''Train using Python implementation of Squared Log Error.'''

    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        y = dtrain.get_label()
        return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''
        y = dtrain.get_label()
        return ((-np.log1p(predt) + np.log1p(y) + 1) /
                np.power(predt + 1, 2))

    def squared_log(predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.

        :math:`\frac{1}{2}[log(pred + 1) - log(label + 1)]^2`

        '''
        predt[predt < -1] = -1 + 1e-6
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess

    def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        """ Root mean squared log error metric.

        :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
        """
        y = dtrain.get_label()
        predt[predt < -1] = -1 + 1e-6
        elements = np.power(np.log1p(y) - np.log1p(predt), 2)
        return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))

    results: Dict[str, Dict[str, List[float]]] = {}
    xgb.train({'tree_method': 'hist', 'seed': kSeed,
               'disable_default_eval_metric': 1},
              dtrain=dtrain,
              num_boost_round=kBoostRound,
              obj=squared_log,
              custom_metric=rmsle,
              evals=[(dtrain, 'dtrain'), (dtest, 'dtest')],
              evals_result=results)

    return results


def plot_rmsle_history(rmse_evals, rmsle_evals, py_rmsle_evals):
    fig, axs = plt.subplots(3, 1)
    ax0: matplotlib.axes.Axes = axs[0]
    ax1: matplotlib.axes.Axes = axs[1]
    ax2: matplotlib.axes.Axes = axs[2]

    x = np.arange(0, kBoostRound, 1)

    ax0.plot(x, rmse_evals['dtrain']['rmse'], label='train-RMSE')
    ax0.plot(x, rmse_evals['dtest']['rmse'], label='test-RMSE')
    ax0.legend()

    ax1.plot(x, rmsle_evals['dtrain']['rmsle'], label='train-native-RMSLE')
    ax1.plot(x, rmsle_evals['dtest']['rmsle'], label='test-native-RMSLE')
    ax1.legend()

    ax2.plot(x, py_rmsle_evals['dtrain']['PyRMSLE'], label='train-PyRMSLE')
    ax2.plot(x, py_rmsle_evals['dtest']['PyRMSLE'], label='test-PyRMSLE')
    ax2.legend()

    plt.show()


def calculate_metrics(model, train_ds_pd, valid_ds_pd, label='SalePrice', predict_on_full_set=True,
                      print_predictions=True):
    """
    Model prediction and ground truth comparison.
    Calculate R-squared, RMSE, RMSLE for training and validation sets.
    """
    train_predictions = model.predict(train_ds_pd) if predict_on_full_set else (
        model.predict(train_ds_pd.drop(label, axis=1)))

    train_r2 = r2_score(train_ds_pd[label], train_predictions)
    print(f'Train R-squared: {train_r2 * 100:.2f}%')

    train_rmse = rmse(train_ds_pd[label], train_predictions)
    print(f'Train RMSE: {train_rmse:.2f}')
    print()

    valid_predictions = model.predict(valid_ds_pd) if predict_on_full_set else (
        model.predict(valid_ds_pd.drop(label, axis=1)))

    valid_r2 = r2_score(valid_ds_pd[label], valid_predictions)
    print(f'Validation R-squared: {valid_r2 * 100:.2f}%')

    valid_rmse = np.sqrt(mean_squared_error(valid_ds_pd[label], valid_predictions, squared=True))
    print(f'Validation RMSE: {valid_rmse:.2f}')
    print()

    if print_predictions:
        indexes = [np.random.randint(0, len(train_ds_pd), 10)]
        for i in indexes:
            print(f"Train Prediction: {train_predictions[i]}, SalePrice: {train_ds_pd[label].iloc[i]}")
        print()

        valid_indexes = [np.random.randint(0, len(valid_ds_pd), 10)]
        for i in valid_indexes:
            print(f"Validation Prediction: {valid_predictions[i]}, SalePrice: {valid_ds_pd[label].iloc[i]}")
        print()

    train_rmsle = np_rmsle(train_ds_pd[label], train_predictions)
    valid_rmsle = np_rmsle(valid_ds_pd[label], valid_predictions)

    print("Training RMSLE: ", train_rmsle)
    print("Validation RMSLE: ", valid_rmsle)

    print()

    return train_r2, valid_r2, train_rmse, valid_rmse, train_rmsle, valid_rmsle


def cv_rmse(model, X, targets, kf, params):
    rmse = np.sqrt(-cross_val_score(model, X, targets,
                                    scoring="neg_mean_squared_error",
                                    cv=kf,
                                    params=params,
                                    verbose=0,
                                    error_score='raise',
                                    n_jobs=-1))
    return rmse


def cv_rmsle(model, X, targets, kf):
    rmsle = -cross_val_score(model, X, targets,
                             scoring="neg_root_mean_squared_log_error",
                             cv=kf,
                             verbose=0,
                             error_score='raise',
                             n_jobs=-1)
    return rmsle


def manual_cross_val_rmse(model, X, targets, kf):
    rmse = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = targets.iloc[train_index], targets.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    return rmse


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def np_rmsle(y_true: np.array, y_pred: np.array) -> np.float64:
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))


def cupy_rmse(y_true, y_pred):
    return cp.sqrt(mean_squared_error(y_true, y_pred))


# Prediction exploration ###########################################################

def plot_predictions(y_true, y_pred, title='Predictions'):
    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.title(title)
    plt.show()


def plot_residuals(y_true, y_pred, title='Residuals'):
    error = y_pred - y_true
    plt.hist(error, bins=25)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title(title)
    plt.show()


def plot_residuals_vs_predictions(y_true, y_pred, title='Residuals vs Predictions'):
    error = y_pred - y_true
    plt.scatter(y_pred, error)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.show()


def prediction_exploration(y_true, y_pred):
    plot_predictions(y_true, y_pred)
    plot_residuals(y_true, y_pred)
    plot_residuals_vs_predictions(y_true, y_pred)


def pred_error_display(y_test, y_pred, scores):
    """https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html"""

    _, ax = plt.subplots(figsize=(5, 5))
    display = PredictionErrorDisplay.from_predictions(
        y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
    )
    ax.set_title("Ridge model, with regularization")
    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")
    plt.tight_layout()
