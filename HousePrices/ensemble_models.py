import numpy as np
import pandas as pd
import ydf
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import tensorflow as tf

import seaborn as sns


def calculate_metrics(model, train_ds_pd, valid_ds_pd, label='SalePrice'):
    train_ds_pd = train_ds_pd.drop([label], axis=1)
    valid_ds_pd = valid_ds_pd.drop([label], axis=1)

    train_predictions = model.predict(train_ds_pd)
    for i in range(10):
        print(f"Train Prediction: {train_predictions[i]}, SalePrice: {train_ds_pd[label].iloc[i]}")
    print()

    train_r2 = r2_score(train_ds_pd[label], train_predictions)
    print(f'Train R-squared: {train_r2 * 100:.2f}%')

    RMSE = np.sqrt(mean_squared_error(train_ds_pd[label], train_predictions, squared=False))
    print(f'Train RMSE: {RMSE:.2f}')
    print()

    valid_predictions = model.predict(valid_ds_pd)
    for i in range(10):
        print(f"Validation Prediction: {valid_predictions[i]}, SalePrice: {valid_ds_pd[label].iloc[i]}")
    print()

    valid_r2 = r2_score(valid_ds_pd[label], valid_predictions)
    print(f'Validation R-squared: {valid_r2 * 100:.2f}%')

    RMSE = np.sqrt(mean_squared_error(valid_ds_pd[label], valid_predictions, squared=False))
    print(f'Validation RMSE: {RMSE:.2f}')
    print()

    return train_r2, valid_r2, RMSE


def ensemble_model(train_ds_pd, valid_ds_pd, test, ids, exp_name, SEED=476, submit=False):
    """
        Ensemble model using StackingCVRegressor from mlxtend library.

    """
    train_labels = train_ds_pd['SalePrice']
    train_x = train_ds_pd.drop(['SalePrice'], axis=1)

    valid_labels = valid_ds_pd['SalePrice']
    valid_x = valid_ds_pd.drop(['SalePrice'], axis=1)

    kf = KFold(n_splits=12, random_state=SEED, shuffle=True)

    def cv_rmse(model, X=train_x, labels=train_labels):
        rmse = np.sqrt(-cross_val_score(model, X, labels, scoring="neg_mean_squared_error", cv=kf))
        return rmse

    def manual_cross_val_rmse(model, X=train_x, labels=train_labels):
        rmse = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        return rmse

    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def np_rmse(y_true: np.array, y_pred: np.array) -> np.float64:
        return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

    lightgbm = LGBMRegressor(objective='regression',
                             num_leaves=6,
                             learning_rate=0.01,
                             n_estimators=4000,
                             max_bin=200,
                             bagging_fraction=0.5,
                             bagging_freq=4,
                             bagging_seed=8,
                             feature_fraction=0.2,
                             feature_fraction_seed=8,
                             min_sum_hessian_in_leaf=11,
                             verbose=-1,
                             random_state=SEED)

    xgboost = XGBRegressor(learning_rate=0.01,
                           n_estimators=3000,
                           max_depth=4,
                           min_child_weight=0,
                           gamma=0.6,
                           subsample=0.5,
                           colsample_bytree=0.7,
                           objective='reg:squarederror',
                           nthread=-1,
                           scale_pos_weight=1,
                           seed=SEED,
                           reg_alpha=6e-5,
                           random_state=SEED)

    ridge_alphas = [1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1,
                    2, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
    ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf,
                                                  scoring='neg_mean_squared_error',
                                                  gcv_mode='auto')
                          )

    # Sklearn Gradient Boosting
    skl_gbr = GradientBoostingRegressor(n_estimators=6000,
                                        criterion='squared_error',
                                        learning_rate=0.01,
                                        max_depth=4,
                                        max_features='sqrt',
                                        min_samples_leaf=15,
                                        min_samples_split=10,
                                        loss='huber',
                                        random_state=SEED,
                                        verbose=0)

    skl_rf = RandomForestRegressor(n_estimators=5578,
                                   max_depth=12,
                                   criterion='squared_error',
                                   min_samples_split=2,
                                   min_samples_leaf=1,
                                   max_features=None,
                                   bootstrap=True,
                                   max_samples=1.0,
                                   oob_score=False,
                                   n_jobs=-1,
                                   random_state=SEED,
                                   verbose=0)

    # nn_model = tf.keras.models.load_model('HousePrices/saved_models/nn_model_experiment_11.19.2024_22.09.46.tf')

    stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm,
                                                # nn_model,
                                                ridge, skl_gbr, skl_rf),
                                    meta_regressor=xgboost,
                                    cv=5,
                                    use_features_in_secondary=True,
                                    n_jobs=-1,
                                    verbose=1)

    print('Cross validated RMSE scores:\n')

    scores = {}

    score = cv_rmse(lightgbm)
    scores['lightgbm'] = (score.mean(), score.std())
    print("LightGBM: {:.4f}".format(scores['lightgbm'], ))

    score = cv_rmse(xgboost)
    scores['xgboost'] = (score.mean(), score.std())
    print("XGBoost: {:.4f}".format(scores['xgboost']))

    score = cv_rmse(ridge)
    scores['ridge'] = (score.mean(), score.std())
    print("Ridge: {:.4f}".format(scores['ridge']))

    score = cv_rmse(skl_rf)
    scores['skl_rf'] = (score.mean(), score.std())
    print("SKLearn Random Forest: {:.4f}".format(scores['skl_rf']))

    score = cv_rmse(skl_gbr)
    scores['sklearn_gbr'] = (score.mean(), score.std())
    print("sklearn grb: {:.4f}".format(scores['sklearn_gbr']))

    score = cv_rmse(stack_gen)
    scores['stack_gen'] = (score.mean(), score.std())
    print("Stacked model: {:.4f}".format(scores['stack_gen']))

    # score = cv_rmse(nn_model)
    # print("NN model: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    # scores['nn'] = (score.mean(), score.std())

    print("\nModel fitting...\n")

    print('LightGBM')
    lgb_model = lightgbm.fit(train_x, train_labels,
                             eval_set=[(valid_x, valid_labels)])
    # calculate_metrics(lgb_model, train_ds_pd, valid_ds_pd)

    print('XGBoost')
    xgb_model = xgboost.fit(train_x, train_labels)
    # calculate_metrics(xgb_model, train_ds_pd, valid_ds_pd)

    print('Ridge')
    ridge_model = ridge.fit(train_x, train_labels)
    # calculate_metrics(ridge_model, train_ds_pd, valid_ds_pd)

    print('SKLearn Random Forest')
    rf_model = skl_rf.fit(train_x, train_labels)
    # calculate_metrics(rf_model, train_ds_pd, valid_ds_pd)

    print('SKLearn GradientBoosting')
    skl_gbr_model = skl_gbr.fit(train_x, train_labels)
    # calculate_metrics(gbr_model, train_ds_pd, valid_ds_pd)

    print('Stacking')
    stack_gen_model = stack_gen.fit(np.array(train_x), np.array(train_labels))
    # calculate_metrics(stack_gen_model, train_ds_pd, valid_ds_pd)

    # print('NN')
    # nn_model = tf.keras.models.load_model('HousePrices/saved_models/nn_model_experiment_08.23.2024_23.09.50.keras')

    initial_weights = np.array([0.05, 0.1, 0.3, 0.1, 0.05, 0.35,
                                # 0.05
                                ])

    def blended_predictions(X, weights):
        return (weights[0] * ridge_model.predict(X) +
                weights[1] * skl_gbr_model.predict(X) +
                weights[2] * xgb_model.predict(X) +
                weights[3] * lgb_model.predict(X) +
                weights[4] * rf_model.predict(X) +
                weights[5] * stack_gen_model.predict(np.array(X))
                # weights[6] * nn_model.predict(X)
                )

    def objective(weights, X, y_true):
        return np.sqrt(mean_squared_error(y_true, blended_predictions(X, weights)))

    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds for weights (e.g., between 0 and 1)
    bounds = [(0, 1) for _ in initial_weights]

    result = minimize(
        objective,
        x0=initial_weights,
        args=(valid_x, valid_labels),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    print("Optimal weights:", optimal_weights)

    blended_score = rmse(train_labels, blended_predictions(train_x, optimal_weights))
    scores['blended'] = (blended_score, 0)
    print("Blended score: {:.4f} ({:.4f})".format(blended_score.mean(), blended_score.std()))

    train_predictions = blended_predictions(train_x, optimal_weights)
    train_score = np_rmse(train_labels, train_predictions)
    print('Root Mean Squared Logarithmic Error (RMSLE) score on the training dataset:')
    print(train_score)

    validation_predictions = blended_predictions(valid_x, optimal_weights)
    validation_score = np_rmse(valid_ds_pd['SalePrice'], validation_predictions)
    print('Root Mean Squared Logarithmic Error (RMSLE) score on the validation dataset:')
    print(validation_score)

    # Plotting scores
    sns.set_style("white")
    fig = plt.figure(figsize=(24, 12))
    ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()],
                       # markers=['o'],
                       # linestyles=['-']
                       )
    for i, score in enumerate(scores.values()):
        ax.text(i,
                score[0],
                '{:.6f}'.format(score[0]),
                horizontalalignment='left',
                size='large',
                color='black',
                weight='semibold'
                )

    plt.ylabel('Root Mean Squared Error', size=20, labelpad=12.5)
    plt.xlabel('Model name', size=23, labelpad=12.5)
    plt.tick_params(axis='x', labelsize=13.5)
    plt.tick_params(axis='y', labelsize=12.5)

    plt.title('Scores of Models', size=20)
    plt.show()

    # Submission
    if submit:
        submission = pd.read_csv("HousePrices/data/sample_submission.csv")

        submission.iloc[:, 1] = np.floor((blended_predictions(test, optimal_weights)))

        q1 = submission['SalePrice'].quantile(0.0045)
        q2 = submission['SalePrice'].quantile(0.99)
        submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x * 0.77)
        submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x * 1.1)

        submission.to_csv("HousePrices/submissions/submission_" + exp_name + "_1.csv", index=False)

        # submission['SalePrice'] *= 1.001619

        submission.to_csv("HousePrices/submissions/submission_" + exp_name + "_2.csv", index=False)
