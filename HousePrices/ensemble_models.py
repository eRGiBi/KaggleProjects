import numpy as np
import pandas as pd
import ydf
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
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

    RMSE = mean_squared_error(train_ds_pd[label], train_predictions, squared=False)
    print(f'Train RMSE: {RMSE:.2f}')
    print()

    valid_predictions = model.predict(valid_ds_pd)
    for i in range(10):
        print(f"Validation Prediction: {valid_predictions[i]}, SalePrice: {valid_ds_pd[label].iloc[i]}")
    print()

    valid_r2 = r2_score(valid_ds_pd[label], valid_predictions)
    print(f'Validation R-squared: {valid_r2 * 100:.2f}%')

    RMSE = mean_squared_error(valid_ds_pd[label], valid_predictions, squared=False)
    print(f'Validation RMSE: {RMSE:.2f}')
    print()

    return train_r2, valid_r2, RMSE

def ensemble_model(train_ds_pd, valid_ds_pd, test, ids, exp_name):
    """https://www.kaggle.com/code/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition#Train-a-model"""

    train_labels = train_ds_pd['SalePrice']
    X = train_ds_pd.drop(['SalePrice'], axis=1)

    kf = KFold(n_splits=12, random_state=42, shuffle=True)

    def rmsle(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    def cv_rmse(model, X=X):
        rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring="neg_mean_squared_error", cv=kf))
        return (rmse)

    # Light Gradient Boosting Regressor
    lightgbm = LGBMRegressor(objective='regression',
                             num_leaves=6,
                             learning_rate=0.01,
                             n_estimators=7000,
                             max_bin=200,
                             bagging_fraction=0.8,
                             bagging_freq=4,
                             bagging_seed=8,
                             feature_fraction=0.2,
                             feature_fraction_seed=8,
                             min_sum_hessian_in_leaf=11,
                             verbose=-1,
                             random_state=42)

    # XGBoost Regressor
    xgboost = XGBRegressor(learning_rate=0.01,
                           n_estimators=6000,
                           max_depth=4,
                           min_child_weight=0,
                           gamma=0.6,
                           subsample=0.7,
                           colsample_bytree=0.7,
                           objective='reg:linear',
                           nthread=-1,
                           scale_pos_weight=1,
                           seed=27,
                           reg_alpha=0.00006,
                           random_state=42)

    # Ridge Regressor
    ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18,
                    20, 30, 50, 75, 100]
    ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

    # Support Vector Regressor
    # svr = make_pipeline(RobustScaler(), SVR(C=20, epsilon=0.008, gamma=0.0003))

    # Gradient Boosting Regressor
    gbr = GradientBoostingRegressor(n_estimators=6000,
                                    learning_rate=0.01,
                                    max_depth=4,
                                    max_features='sqrt',
                                    min_samples_leaf=15,
                                    min_samples_split=10,
                                    loss='huber',
                                    random_state=42)

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=1200,
                               max_depth=15,
                               min_samples_split=5,
                               min_samples_leaf=5,
                               max_features=None,
                               oob_score=True,
                               random_state=42
                               )


    # nn_model = tf.keras.models.load_model('HousePrices/saved_models/nn_model_experiment_08.20.2024_22.39.52.h5')

    stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm,
                                                # svr,
                                                # nn_model,

                                                ridge, gbr, rf),
                                    meta_regressor=xgboost,
                                    use_features_in_secondary=True
                                    )

    scores = {}

    score = cv_rmse(lightgbm)
    print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['lgb'] = (score.mean(), score.std())

    score = cv_rmse(xgboost)
    print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['xgb'] = (score.mean(), score.std())

    # score = cv_rmse(svr)
    # print("SVR: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    # scores['svr'] = (score.mean(), score.std())

    score = cv_rmse(ridge)
    print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['ridge'] = (score.mean(), score.std())

    score = cv_rmse(rf)
    print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['rf'] = (score.mean(), score.std())

    score = cv_rmse(gbr)
    print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))
    scores['gbr'] = (score.mean(), score.std())


    print('stack_gen')
    stack_gen_model = stack_gen.fit(np.array(X), np.array(train_labels))

    print('lightgbm')
    lgb_model_full_data = lightgbm.fit(X, train_labels)
    # calculate_metrics(lgb_model_full_data, train_ds_pd, valid_ds_pd)

    print('xgboost')
    xgb_model_full_data = xgboost.fit(X, train_labels)
    # calculate_metrics(xgb_model_full_data, train_ds_pd, valid_ds_pd)

    print('Svr')
    # svr_model_full_data = svr.fit(X, train_labels)
    # calculate_metrics(svr_model_full_data, train_ds_pd, valid_ds_pd)

    print('Ridge')
    ridge_model_full_data = ridge.fit(X, train_labels)
    # calculate_metrics(ridge_model_full_data, train_ds_pd, valid_ds_pd)

    print('RandomForest')
    rf_model_full_data = rf.fit(X, train_labels)
    # calculate_metrics(rf_model_full_data, train_ds_pd, valid_ds_pd)

    print('GradientBoosting')
    gbr_model_full_data = gbr.fit(X, train_labels)
    # calculate_metrics(gbr_model_full_data, train_ds_pd, valid_ds_pd)


    def blended_predictions(X):
        return ((0.1 * ridge_model_full_data.predict(X)) + \
                # (0.2 * svr_model_full_data.predict(X)) + \
                # (0.2 * nn_model.predict(X)) + \
                (0.1 * gbr_model_full_data.predict(X)) + \
                (0.3 * xgb_model_full_data.predict(X)) + \
                (0.1 * lgb_model_full_data.predict(X)) + \
                (0.05 * rf_model_full_data.predict(X)) + \
                (0.35 * stack_gen_model.predict(np.array(X)))
                )

    blended_score = rmsle(train_labels, blended_predictions(X))
    scores['blended'] = (blended_score, 0)
    print('RMSLE score on train data:')
    print(blended_score)

    sns.set_style("white")
    fig = plt.figure(figsize=(24, 12))

    ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()],
                       # markers=['o'],
                       # linestyles=['-']
                       )
    for i, score in enumerate(scores.values()):
        ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black',
                weight='semibold')

    plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
    plt.xlabel('Model', size=20, labelpad=12.5)
    plt.tick_params(axis='x', labelsize=13.5)
    plt.tick_params(axis='y', labelsize=12.5)

    plt.title('Scores of Models', size=20)

    plt.show()

    submission = pd.read_csv("HousePrices/data/sample_submission.csv")


    submission.iloc[:, 1] = np.floor((blended_predictions(test)))

    q1 = submission['SalePrice'].quantile(0.0045)
    q2 = submission['SalePrice'].quantile(0.99)
    submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x * 0.77)
    submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x * 1.1)

    submission.to_csv("HousePrices/submissions/submission_" + exp_name + "_1.csv", index=False)

    submission['SalePrice'] *= 1.001619

    submission.to_csv("HousePrices/submissions/submission_" + exp_name + "_2.csv", index=False)



