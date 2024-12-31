import pickle

import numpy as np
import pandas as pd
import ydf
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, QuantileTransformer

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb
import xgboost as xgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

import tensorflow as tf

import seaborn as sns

from metrics import rmse, cv_rmse, np_rmsle


def ensemble_model(train_ds_pd, valid_ds_pd, test, ids, exp_name, from_scratch=True, SEED=476, submit=False):
    """
        Ensemble model using StackingCVRegressor from mlxtend library.
    """
    train_targets = np.array(train_ds_pd['SalePrice'])
    train_x = np.array(train_ds_pd.drop(['SalePrice'], axis=1))

    valid_targets = np.array(valid_ds_pd['SalePrice'])
    valid_x = np.array(valid_ds_pd.drop(['SalePrice'], axis=1))

    kf = KFold(n_splits=12, random_state=SEED, shuffle=True)

    rmsle_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1)),
                               greater_is_better=False)
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))

    # Model declarations

    lightgbm = LGBMRegressor(objective='regression',
                             num_leaves=6,
                             learning_rate=0.1,
                             n_estimators=4000,
                             max_bin=250,
                             bagging_fraction=0.5,
                             subsample=0.5,
                             subsample_freq=1,
                             bagging_freq=4,
                             bagging_seed=SEED + 1,
                             feature_fraction=0.8,
                             feature_fraction_seed=SEED + 2,
                             min_sum_hessian_in_leaf=11,
                             reg_alpha=0.3,
                             reg_lambda=0.015,
                             n_jobs=-1,
                             verbose=-1,
                             random_state=SEED)

    xgb_params = {"tree_method": "hist",
                  "objective": "reg:squarederror",
                  "eval_metric": "rmse",
                  "importance_type": "weight",
                  "scale_pos_weight": 1,
                  'device': "cpu",
                  "random_state": SEED,
                  "verbosity": 0,
                  'colsample_bytree': 0.5,
                  'gamma': 0.4,
                  'learning_rate': 0.024,  # poss lower
                  'max_depth': 3,
                  'max_leaves': 25,  # 5
                  'n_estimators': 2500,
                  'reg_alpha': 0.00657,
                  'reg_lambda': 0.7,
                  'subsample': 0.5,
                  }
    xgboost = xgb.XGBRegressor(**xgb_params)

    # ridge_alphas = np.array([
    #     0.1, 0.5, 1, 2, 5, 10, 15, 18, 20, 22,
    #     500, 1000, 2000, 3000,
    # ], dtype=np.float64)
    #
    # ridge = make_pipeline(
    #     RobustScaler(with_centering=False,
    #                  with_scaling=True,
    #                  quantile_range=(25.0, 75.0)),
    #     MinMaxScaler(feature_range=(0, 1)),
    #
    #     RidgeCV(alphas=ridge_alphas,
    #             cv=kf,
    #             scoring="neg_root_mean_squared_error",
    #             fit_intercept=False,
    #             gcv_mode='auto',
    #             store_cv_results=False,
    #             alpha_per_target=False,
    #             ),
    #     verbose=False)

    ridge_alphas = np.array([

        1e-2, 3e-2, 5e-2, 7e-2, 9e-2,
        0.1, 0.3, 0.5, 0.7,
        1, 3, 5, 7, 10, 15,
        20, 22, 25, 30, 50, 60, 67, 75, 90, 100,
        500, 1000, 2000,
    ], dtype=np.float64)

    ridge_regressor = RidgeCV(alphas=ridge_alphas,
                              cv=None,
                              scoring="neg_root_mean_squared_error",
                              fit_intercept=False,
                              gcv_mode='auto',
                              store_cv_results=True,
                              alpha_per_target=False,
                              )

    transformed_ridge = TransformedTargetRegressor(
        ridge_regressor,
        transformer=QuantileTransformer(output_distribution='normal',
                                        ignore_implicit_zeros=False,
                                        random_state=SEED)
    )

    ridge = make_pipeline(
        # QuantileTransformer(output_distribution='normal',
        #                     ignore_implicit_zeros=True,
        #                     random_state=SEED),
        RobustScaler(with_centering=False,
                     with_scaling=True,
                     quantile_range=(5.0, 95.0)),
        # MinMaxScaler(feature_range=(0, 1)),
        transformed_ridge,
        verbose=True)

    # sklearn Gradient Boosting
    skl_gbr = GradientBoostingRegressor(n_estimators=3500,
                                        criterion='friedman_mse',
                                        learning_rate=0.01,
                                        subsample=0.5,
                                        max_depth=4,
                                        max_features='sqrt',
                                        min_samples_leaf=4,
                                        min_samples_split=7,
                                        loss='huber',
                                        random_state=SEED,
                                        verbose=0)

    # skl_rf = RandomForestRegressor(n_estimators=5500,
    #                                max_depth=12,
    #                                criterion='squared_error',
    #                                min_samples_split=2,
    #                                min_samples_leaf=1,
    #                                max_features=None,
    #                                bootstrap=True,
    #                                max_samples=1.0,
    #                                oob_score=False,
    #                                n_jobs=-1,
    #                                random_state=SEED,
    #                                verbose=0)
    skl_rf = RandomForestRegressor(n_estimators=2100,
                                   max_depth=16,
                                   criterion='squared_error',
                                   min_samples_split=2,
                                   min_samples_leaf=2,
                                   max_leaf_nodes=None,
                                   min_impurity_decrease=0,
                                   max_features=None,
                                   bootstrap=True,
                                   max_samples=1.0,
                                   oob_score=True,
                                   n_jobs=-1,
                                   random_state=SEED,
                                   verbose=0)

    stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm,
                                                ridge, skl_gbr, skl_rf),
                                    meta_regressor=xgboost,
                                    cv=5,
                                    use_features_in_secondary=True,
                                    refit=True,
                                    shuffle=True,
                                    random_state=SEED,
                                    n_jobs=-1,
                                    verbose=0)

    if from_scratch:
        print()
        print('Cross-validated RMSE scores:\n')

        scores = {}

        print('XGBoost')
        # score = cv_rmse(xgboost, params={"eval_set": (valid_x, valid_targets)})
        xgboost.fit(train_x, train_targets, eval_set=[(valid_x, valid_targets)])
        score = rmse(train_targets, xgboost.predict(train_x))
        scores['XGBoost'] = (score.mean(), score.std())

        for regressor in [lightgbm, ridge, skl_rf, skl_gbr, stack_gen]:
            name = regressor.__class__.__name__
            print(name)
            regressor = TransformedTargetRegressor(regressor,
                                                   transformer=QuantileTransformer(output_distribution='normal',
                                                                                   ignore_implicit_zeros=False,
                                                                                   n_quantiles=1000,
                                                                                   random_state=SEED))
            score = cv_rmse(regressor, train_x, train_targets, kf, params=None)
            scores[name] = (score.mean(), score.std())

        print("Model fitting...\n")

        print('LightGBM')
        lgb_model = lightgbm.fit(train_x, train_targets,
                                 eval_set=[(valid_x, valid_targets)])
        # calculate_metrics(lgb_model, train_ds_pd, valid_ds_pd)

        print('XGBoost')
        xgb_model = xgboost.fit(train_x, train_targets,
                                eval_set=[(valid_x, valid_targets)])
        # calculate_metrics(xgb_model, train_ds_pd, valid_ds_pd)

        print('Ridge')
        ridge_model = ridge.fit(X=np.concatenate((train_x, valid_x), axis=0),
                                y=np.concatenate((train_targets, valid_targets), axis=0))
        # calculate_metrics(ridge_model, train_ds_pd, valid_ds_pd)
        # ridgecv_step = ridge.named_steps['ridgecv']
        # print("Best alpha: ", ridgecv_step.alpha_)
        # print("Best score: ", ridgecv_step.best_score_)

        print('sklearn Random Forest')
        rf_model = skl_rf.fit(train_x, train_targets)
        # calculate_metrics(rf_model, train_ds_pd, valid_ds_pd)

        print('sklearn Gradient Boosting')
        skl_gbr_model = skl_gbr.fit(train_x, train_targets)
        # calculate_metrics(gbr_model, train_ds_pd, valid_ds_pd)

        print('mlxtend CVStacking')
        stack_gen_model = stack_gen.fit(np.array(train_x), np.array(train_targets),
                                        sample_weight=None)
        # calculate_metrics(stack_gen_model, train_ds_pd, valid_ds_pd)

    else:
        print('Loading models...')

        print('Ridge')
        ridge_model = pickle.load(open("HousePrices/saved_models/best_ridge_pipeline.joblib", "rb"))

        print('LightGBM')
        lgb_model = pickle.load(open("HousePrices/saved_models/best_lgb_model.joblib", "rb"))

        print('XGBoost')
        xgb_model = xgboost.load_model('HousePrices/saved_models/best_xgb_model.json')

        print("sklearn Random Forest")
        rf_model = pickle.load(open("HousePrices/saved_models/best_skl_rf_model.joblib", "rb"))

        print('sklearn Gradient Boosting')
        skl_gbr_model = pickle.load(open("HousePrices/saved_models/best_skl_gb_model.joblib", "rb"))

        print('mlxtend CVStacking')

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

        # np.sqrt(mean_squared_error(y_true, blended_predictions(X, weights)))

        return np.sqrt(np.mean(np.square(np.log1p(blended_predictions(X, weights)) - np.log1p(y_true))))

    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

    # Bounds for weights (e.g., between 0 and 1)
    bounds = [(0, 1) for _ in initial_weights]

    result = minimize(
        objective,
        x0=initial_weights,
        args=(valid_x, valid_targets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    print("Optimal weights:", optimal_weights)

    # # Refitting StackGen model with optimal weights
    # opt_stack_gen = stack_gen.fit(np.array(train_x), np.array(train_targets),
    #                               sample_weight=None)
    #
    # score = cv_rmse(opt_stack_gen)
    # scores['opt_stack_gen'] = (score.mean(), score.std())
    # print(stack_gen.score(np.array(valid_x), np.array(valid_targets)))

    blended_score = rmse(train_targets, blended_predictions(train_x, initial_weights))
    scores['original_blended'] = (blended_score, 0)
    print("Original Blended score: {:.4f} ({:.4f})".format(blended_score.mean(), blended_score.std()))

    train_predictions = blended_predictions(train_x, optimal_weights)

    blended_score = rmse(train_targets, train_predictions)
    scores['blended'] = (blended_score, 0)
    print("Blended score: {:.4f} ({:.4f})".format(blended_score.mean(), blended_score.std()))

    train_score = np_rmsle(train_targets, train_predictions)
    print('Root Mean Squared Logarithmic Error (RMSLE) score on the training dataset:')
    print(train_score)

    validation_predictions = blended_predictions(valid_x, optimal_weights)
    validation_score = np_rmsle(valid_ds_pd['SalePrice'], validation_predictions)
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

        submission.to_csv("HousePrices/submissions/submission_" + exp_name + ".csv", index=False)
