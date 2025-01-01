import pickle
from copy import deepcopy
from pickle import dump

import matplotlib
import numpy as np
import pandas as pd
import cupy as cp
import sklearn.metrics
from numpy import sort

from scipy.stats import stats, uniform, rv_discrete, randint
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, make_scorer, mean_squared_log_error, \
    root_mean_squared_error
# import category_encoders as ce

from sklearn.linear_model import RidgeCV
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestRegressor, IsolationForest, GradientBoostingRegressor
from lightgbm import LGBMRegressor

import xgboost as xgb

import tensorflow as tf
# import tensorflow_decision_forests as tfdf
import ydf

from sklearn.metrics._dist_metrics import parse_version
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, QuantileTransformer
from tensorflow.python import ops

from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.model_selection import GridSearchCV
# from hyperopt import fmin, tpe, hp
import optuna


import matplotlib.pyplot as plt
import seaborn as sns

from ExperimentLogger import ExperimentLogger
from metrics import calculate_metrics, pred_error_display, rmse, np_rmsle

# TODO: Refactoring
exp_logger = ExperimentLogger('HousePrices/submissions/experiment_aggregate.csv')


def make_submission(model, test_data, ids, exp_name='experiment'):
    sample_submission_df = pd.read_csv('./HousePrices/data/sample_submission.csv')
    sample_submission_df['SalePrice'] = model.predict(test_data)
    sample_submission_df.to_csv('./HousePrices/submissions/submission_' + exp_name + '.csv', index=False)
    print(sample_submission_df.head())


def tf_decision_forests(train_ds_pd, valid_ds_pd, test, ids, exp_name, SEED):
    # Random Forest regression with TensorFlow Decision Forests

    label = 'SalePrice'
    train_ds_pd = train_ds_pd.drop(['SalePrice'], axis=1)
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label,
                                                     task=tfdf.keras.Task.REGRESSION,
                                                     max_num_classes=700)
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label,
                                                     task=tfdf.keras.Task.REGRESSION,
                                                     max_num_classes=700)
    tuner = tfdf.tuner.RandomSearch(num_trials=20, trial_num_threads=3)

    # Hyperparameters to optimize
    tuner.choice("max_depth", [4, 5, 7, 16, 32])
    tuner.choice("num_trees", [50, 100, 200, 500])

    print(tuner.train_config())

    model = tfdf.keras.RandomForestModel(tuner=tuner, task=tfdf.keras.Task.REGRESSION,
                                         bootstrap_training_dataset=True,
                                         bootstrap_size_ratio=1.0,
                                         categorical_algorithm='CART',  # RANDOM
                                         growing_strategy='LOCAL',  # BEST_FIRST_GLOBAL
                                         honest=False,
                                         min_examples=1,
                                         missing_value_policy='GLOBAL_IMPUTATION',
                                         num_candidate_attributes=0,
                                         random_seed=SEED,
                                         winner_take_all=True,
                                         verbose=2)

    model.compile(metrics=["mse"])

    model.fit(x=train_ds)

    model.evaluate(x=train_ds)

    print(model.summary())
    print()

    print("TensorFlow Decision Forests Results: -------------------")
    calculate_metrics(model, train_ds_pd, valid_ds_pd)

    # tfdf.model_plotter.plot_model(model, tree_idx=0, max_depth=None)
    # plt.show()

    logs = model.make_inspector().training_logs()
    plt.plot([log.num_trees for log in logs], [log.evaluation.rmse for log in logs])
    plt.xlabel("Number of trees")
    plt.ylabel("RMSE (out-of-bag)")
    plt.show()

    # Variable importance
    inspector = model.make_inspector()
    inspector.evaluation()

    print(f"Available variable importance:")
    for importance in inspector.variable_importances().keys():
        print("\t", importance)
    print()

    inspector = model.make_inspector()
    print(inspector.evaluation())

    print(inspector.variable_importances()["NUM_AS_ROOT"])
    print()

    plt.figure(figsize=(12, 4))

    variable_importance_metric = "NUM_AS_ROOT"
    variable_importances = inspector.variable_importances()[variable_importance_metric]

    feature_names = [vi[0].name for vi in variable_importances]
    feature_importances = [vi[1] for vi in variable_importances]
    feature_ranks = range(len(feature_names))

    bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
    plt.yticks(feature_ranks, feature_names)
    plt.gca().invert_yaxis()

    # Label each bar with values
    for importance, patch in zip(feature_importances, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

    plt.xlabel(variable_importance_metric)
    plt.title("NUM AS ROOT of the class 1 vs the others")
    plt.tight_layout()
    plt.show()

    train_r2, valid_r2, RMSE = calculate_metrics(model, train_ds_pd, valid_ds_pd, label)

    exp_logger.save({"Id": exp_name, "Model": "TensorFlow Decision Forests",
                     "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                     "Hyperparameters": model.predefined_hyperparameters()})

    model.save("./saved_models/")


def sklearn_random_forest(data, valid_ds_pd, test, ids, exp_name, SEED, tune=False):
    print("SKLearn Random Forest Regressor: -------------------")

    x_train = np.array(data)
    y_train = data['SalePrice']

    x_test = np.array(valid_ds_pd)
    y_test = valid_ds_pd['SalePrice']

    if not tune:

        # rf = RandomForestRegressor(n_estimators=750,
        #                            max_depth=16,
        #                            criterion='friedman_mse',
        #                            min_samples_split=5,
        #                            min_samples_leaf=5,
        #                            max_features=1,
        #                            bootstrap=True,
        #                            max_samples=0.925,
        #                            oob_score=True,
        #                            n_jobs=5,
        #                            random_state=SEED,
        #                            verbose=1
        #                            )

        rf = RandomForestRegressor(n_estimators=2100,
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
                                   verbose=1)

        rf.fit(x_train, y_train)

        # with open("HousePrices/saved_models/best_skl_rf_model.joblib", "wb") as f:
        #     dump(rf, f, protocol=5)

    else:
        random_grid = {'criterion': [  #'squared_error',
            'friedman_mse'],
            'n_estimators': [int(x) for x in np.linspace(start=500, stop=3500, num=100)],
            'max_features': ['auto', 'sqrt', None],
            'max_depth': [int(x) for x in np.linspace(16, 32, num=4)],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_leaf_nodes': [None, 10, 20, 30, 40, 50],
            'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'bootstrap': [True, False],
            'max_samples': [0.9, 0.925, 0.95, 0.975, 1.0],
            'oob_score': [True, False]
        }

        rfr = RandomForestRegressor(n_jobs=-1, random_state=SEED, verbose=2)

        rf = RandomizedSearchCV(estimator=rfr,
                                param_distributions=random_grid,
                                n_iter=1000,
                                cv=5,
                                verbose=1,
                                random_state=SEED,
                                n_jobs=-1)

        search = rf.fit(x_train, y_train)

        print(search.best_params_)

    calculate_metrics(rf, data, valid_ds_pd)

    y_pred_train = rf.predict(x_train)
    y_pred_test = rf.predict(x_test)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f'Train R-squared: {train_r2 * 100:.2f}%')
    print(f'Test R-squared: {test_r2 * 100:.2f}%')
    print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
    print(f'Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print(f'Test Mean Absolute Error (MAE): {test_mae:.2f}')


def yggdrassil_random_forest(train_ds_pd, valid_ds_pd, test, ids, exp_name, SEED, submit=False, tune=False):
    if not tune:

        # Best found hyperparameters

        learner = ydf.RandomForestLearner(label="SalePrice",
                                          task=ydf.Task.REGRESSION,
                                          include_all_columns=True,
                                          features=[
                                              # ydf.Feature("PoolArea", monotonic=+1),
                                          ],
                                          num_trees=750,
                                          max_depth=16,
                                          bootstrap_size_ratio=0.925,
                                          categorical_algorithm="RANDOM",
                                          growing_strategy="LOCAL",
                                          winner_take_all=False,
                                          num_threads=-1,
                                          random_seed=SEED,
                                          )

        model = learner.train(ds=train_ds_pd, verbose=2)

        print(model.describe())

        evaluation = model.evaluate(valid_ds_pd)

        # Query individual evaluation metrics
        print(f"test accuracy: {evaluation.accuracy}")

        print("Full evaluation report:")
        print(evaluation)

        print("Analytics Results: -------------------")
        print()
        analysis = model.analyze(valid_ds_pd, sampling=0.1)
        print(analysis)
        analysis.to_file("HousePrices/results/analysis.html")

        # best_tree = max(tree in tree for tree in model.iter_trees())
        # model.plot_tree()

        print("Benchmark Results: -------------------")
        print(model.benchmark(valid_ds_pd))

        # model.save("./HousePrices/saved_models/best_hyp_model_" + exp_name)

        train_r2, valid_r2, RMSE = calculate_metrics(model, train_ds_pd, valid_ds_pd)

        exp_logger.save({"Id": exp_name, "Model": "Tuned Yggdrasil Random Forest",
                         "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                         "Hyperparameters": "num_trees=750, max_depth=16, bootstrap_size_ratio=0.925, "}
                        )

        print("Yggdrasil Submission: -------------------")
        make_submission(model, test, ids, exp_name)

    else:

        print("Hyperparameter tuning: -------------------")

        tuner = ydf.RandomSearchTuner(num_trials=1000)

        # Hyperparameters to optimize.
        tuner.choice("max_depth", [16, 32])
        # tuner.choice("max_vocab_count", [500, 100, 200, 300, 1000, 2000, 3000])
        # tuner.choice("min_vocab_frequency", [5, 10, 2, 3])
        tuner.choice("num_trees", [500, 750, 1000])
        tuner.choice("bootstrap_size_ratio", [0.9, .925, .91, 1.0])
        tuner.choice("categorical_algorithm", ["RANDOM", "CART"])
        tuner.choice("growing_strategy", ["LOCAL", "BEST_FIRST_GLOBAL"])
        tuner.choice("winner_take_all", [True, False])

        learner = ydf.RandomForestLearner(label="SalePrice",
                                          tuner=tuner,
                                          task=ydf.Task.REGRESSION,
                                          include_all_columns=True,
                                          features=[
                                              # ydf.Feature("PoolArea", monotonic=+1),
                                          ],
                                          random_seed=SEED,
                                          num_threads=11
                                          )

        model = learner.train(ds=train_ds_pd, verbose=1)

        evaluation = learner.cross_validation(valid_ds_pd, folds=10)

        print("Cross validation evaluation:")
        print(evaluation)

        print("Model description:")
        print(model.describe())
        # print(model)

        evaluation = model.evaluate(valid_ds_pd)

        # Query individual evaluation metrics
        print(f"test accuracy: {evaluation.accuracy}")

        # Show the full evaluation report
        print("Full evaluation report:")
        print(evaluation)

        print("Analytics Results: -------------------")
        print()
        analysis = model.analyze(valid_ds_pd, sampling=0.1)
        print(analysis)
        analysis.to_file("HousePrices/results/" + exp_name + "_analysis.html")

        print("Benchmark Results: -------------------")
        # print(model.benchmark(valid_ds_pd))
        # print(model.to_cpp())

        train_r2, valid_r2, RMSE = calculate_metrics(model, train_ds_pd, valid_ds_pd)

        exp_logger.save({"Id": exp_name, "Model": "Tuned Yggdrasil Random Forest",
                         "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                         # "Hyperparameters": model.predefined_hyperparameters()
                         })

        logs = model.hyperparameter_optimizer_logs()
        top_score = max(t.score for t in logs.trials)
        selected_trial = [t for t in logs.trials if t.score == top_score][0]

        best_trial_index = logs.trials.index(selected_trial)
        model.plot_tree(best_trial_index)

        print("Best hyperparameters:")
        print(selected_trial.params)

        model.save("saved_models/model_" + exp_name)

        print("Yggdrasil Submission: -------------------")
        if submit:
            make_submission(model, test, ids, exp_name)


def ridge_regression(train_ds_pd, valid_ds_pd, test, ids, exp_name, SEED, submit=False, tune=False):
    train_y = train_ds_pd['SalePrice']
    train_x = deepcopy(train_ds_pd).drop('SalePrice', axis=1)

    valid_y = valid_ds_pd['SalePrice']
    valid_x = deepcopy(valid_ds_pd).drop('SalePrice', axis=1)

    kf = KFold(n_splits=12, random_state=SEED, shuffle=True)

    # transformer = RobustScaler(with_centering=True,
    #                            with_scaling=True,
    #                            quantile_range=(25.0, 75.0))
    #
    # train_y_tr = transformer.fit_transform(train_x, train_y)

    ridge_alphas = np.array([
        1e-10, 1e-8, 5e-6, 5e-5,
        1e-4, 3e-4, 5e-4, 7e-4, 9e-4,
        1e-3, 3e-3, 5e-3, 7e-3, 9e-3,
        1e-2, 3e-2, 5e-2, 7e-2, 9e-2,
        0.1, 0.3, 0.5, 0.7, 0.9,
        1, 2, 3, 5, 7, 10, 15, 18, 19,
        20, 22, 25, 30, 50, 60, 67, 75, 80, 90, 100,
        500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000,
    ], dtype=np.float64)

    rmsle_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1)),
                               greater_is_better=False)
    rmse_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)))

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
                                        random_state=SEED),
        check_inverse=True,
        # inverse_func=np.expm1,
    )
    ridge_pipeline = make_pipeline(
        # QuantileTransformer(output_distribution='normal',
        #                     ignore_implicit_zeros=True,
        #                     random_state=SEED),
        RobustScaler(with_centering=False,
                     with_scaling=True,
                     quantile_range=(5.0, 95.0)),
        # MinMaxScaler(feature_range=(0, 1)),
        transformed_ridge,
        verbose=True)

    print("Pipeline components:", ridge_pipeline.steps)

    fitted_ridge_model = ridge_pipeline.fit(X=np.concatenate((train_x, valid_x), axis=0),
                                            y=np.concatenate((train_y, valid_y), axis=0))

    # with open("HousePrices/saved_models/best_ridge_pipline.joblib", "wb") as f:
    #     dump(fitted_ridge_model, f, protocol=5)

    print()
    print("Regression Results: -------------------")
    train_r2, valid_r2, train_rmse, valid_rmse, train_rmsle, valid_rmsle = calculate_metrics(fitted_ridge_model,
                                                                                             train_ds_pd, valid_ds_pd,
                                                                                             predict_on_full_set=False,
                                                                                             print_predictions=False)
    print()

    ridgecv_step = transformed_ridge.regressor_

    print("Best alpha: ", ridgecv_step.alpha_)
    print("Best score: ", ridgecv_step.best_score_)

    rmse_scores = {"Train R2": train_r2, "Validation R2": valid_r2}
    pred_error_display(valid_y, fitted_ridge_model.predict(valid_x), rmse_scores)

    # Cross-validation results
    if ridgecv_step.get_params()['store_cv_results']:

        cv_results = ridgecv_step.cv_results_

        mean_test_scores = -np.mean(cv_results, axis=0)

        for i, score in enumerate(mean_test_scores):
            print(f"Alpha: {ridge_alphas[i]}, MSE: {score}")

        plt.figure(figsize=(10, 5))
        plt.plot(ridge_alphas, mean_test_scores, marker="o", color="b", label="Mean Test Score")
        plt.xscale("log")
        plt.xlabel("Alpha on log scale")
        plt.ylabel("Mean Test Score (MSE)")
        plt.title("Cross-Validation Results")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.legend()
        plt.show()

        coef_df = pd.DataFrame({
            'Feature': train_ds_pd.drop('SalePrice', axis=1).columns,
            'Coefficient': ridgecv_step.coef_
        })

        sorted_coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)

        # Plot coefficients
        sorted_coef_df.head(20).plot(x='Feature', y='Coefficient', kind='barh', figsize=(10, 12))
        plt.xlabel("Coefficient Value")
        plt.ylabel("Feature")
        plt.title("Ridge Regression Coefficients")
        plt.tight_layout()
        plt.show()

        quantile_transformer = QuantileTransformer(output_distribution='normal', ignore_implicit_zeros=False,
                                                   random_state=SEED)
        transformed_x = quantile_transformer.fit_transform(np.concatenate((train_x, valid_x), axis=0))
        std_dev_transformed = transformed_x.std(axis=0)

        # for coef, std, tr_std in zip(ridgecv_step.coef_,
        #                              np.concatenate((train_x, valid_x), axis=0).std(axis=0),
        #                              std_dev_transformed):
        #     print(coef, std, tr_std)

        corrected_coef_df = coef_df.copy()
        corrected_coef_df['Coefficient'] *= std_dev_transformed
        corrected_coef_df = corrected_coef_df.reindex(
            corrected_coef_df.Coefficient.abs().sort_values(ascending=False).index)

        corrected_coef_df.head(20).plot(x='Feature', y='Coefficient', kind="barh", figsize=(10, 12))
        plt.xlabel("Coefficient values corrected by the feature's std. dev.")
        plt.title("Corrected Ridge Regression Coefficients")
        plt.ylabel("Feature")
        plt.axvline(x=0, color=".5")
        plt.subplots_adjust(left=0.3)
        plt.show()


def gradient_booster(train_ds_pd, valid_ds_pd, test, ids, exp_name, SEED, submit=False, tune=False):
    visualize = True

    use_xgb = True
    lgbm = False
    use_skl = False

    train_y = train_ds_pd['SalePrice']
    train_x = deepcopy(train_ds_pd).drop('SalePrice', axis=1)
    valid_y = valid_ds_pd['SalePrice']
    valid_x = deepcopy(valid_ds_pd).drop('SalePrice', axis=1)

    regressor = None
    monotonic_cst = {}

    # Sklearn Gradient Boosting

    if use_skl:

        if not tune:

            params = {
                'n_estimators': 3500,
                'criterion': 'friedman_mse',
                'learning_rate': 0.01,
                'subsample': 0.5,
                'max_depth': 4,
                'max_features': 'sqrt',
                'min_samples_leaf': 4,
                'min_samples_split': 7,
                'loss': 'huber',
                'random_state': SEED,
                'verbose': 2
            }

            regressor = GradientBoostingRegressor(**params)

            # regressor = pickle.load(open("HousePrices/saved_models/best_skl_gb_model.joblib", "rb"))

            model = regressor.fit(train_x, train_y)

            print("SKL Gradient Boosting Regressor: -------------------")
            calculate_metrics(model, train_ds_pd, valid_ds_pd, predict_on_full_set=False, print_predictions=False)

            if visualize:
                feature_importance = regressor.feature_importances_
                sorted_idx = np.argsort(feature_importance)[-10:]

                pos = np.arange(sorted_idx.shape[0]) + 0.5
                fig = plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.barh(pos, feature_importance[sorted_idx], align="center")
                plt.yticks(pos, np.array(train_ds_pd.columns)[sorted_idx])
                plt.title("Feature Importance (MDI)")

                result = permutation_importance(
                    regressor, valid_x, valid_y, n_repeats=10, random_state=SEED, n_jobs=-1)

                sorted_idx = result.importances_mean.argsort()
                sorted_idx = sorted_idx[-10:]
                plt.subplot(1, 2, 2)

                tick_labels_parameter_name = (
                    "tick_labels"
                    if parse_version(matplotlib.__version__) >= parse_version("3.9")
                    else "labels"
                )

                tick_labels_dict = {
                    tick_labels_parameter_name: np.array(train_ds_pd.columns)[sorted_idx]
                }
                plt.boxplot(result.importances[sorted_idx].T, vert=False, **tick_labels_dict)
                plt.title("Permutation Importance (test set)")
                fig.tight_layout()
                plt.show()

                with open("HousePrices/saved_models/best_skl_gb_model.joblib", "wb") as f:
                    dump(model, f, protocol=5)

        else:
            parameter_grid = {
                'n_estimators': [1800, 3500, 4500, 5500],
                'max_depth': [4],
                'learning_rate': [0.1, 0.05, 0.01],
                'subsample': [0.5],
                'min_samples_leaf': [7, 4],
                'min_samples_split': [7],
            }

            regressor = GradientBoostingRegressor(random_state=SEED, verbose=2)

            grid_search = GridSearchCV(regressor, parameter_grid, cv=5, scoring='neg_root_mean_squared_log_error',
                                       n_jobs=-1,
                                       return_train_score=True,
                                       verbose=3)

            grid_search.fit(train_x, train_y)

            print("Best set of hyperparameters: ", grid_search.best_params_)
            print("Best score: ", grid_search.best_score_)

            df = pd.DataFrame(grid_search.cv_results_)
            print(df)
            df.to_csv("HousePrices/results/skl_grb_grid_search_results.csv")

            plt.show()

        if visualize:
            # xgb.plot_importance(bst, max_num_features=20)
            # plt.show()

            test_score = np.zeros((params["n_estimators"]), dtype=np.float64)
            for i, y_pred in enumerate(regressor.staged_predict(valid_x)):
                test_score[i] = mean_squared_error(valid_y, y_pred)

            fig = plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            plt.title("Deviance")
            plt.plot(
                np.arange(params["n_estimators"]) + 1,
                regressor.train_score_,
                "b-",
                label="Training Set Deviance",
            )
            plt.plot(
                np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
            )
            plt.legend(loc="upper right")
            plt.xlabel("Boosting Iterations")
            plt.ylabel("Deviance")
            fig.tight_layout()
            plt.show()

    # XGBoost
    if use_xgb:

        accuracy_history = []

        dtrain = xgb.DMatrix(train_x.to_numpy(), label=train_y.to_numpy(), nthread=-1, silent=True)
        dvalid = xgb.DMatrix(valid_x.to_numpy(), label=valid_y.to_numpy(), nthread=-1, silent=True)

        dtrain.set_float_info("label_lower_bound", train_y.to_numpy())
        dtrain.set_float_info("label_upper_bound", train_y.to_numpy())
        dvalid.set_float_info("label_lower_bound", valid_y.to_numpy())
        dvalid.set_float_info("label_upper_bound", valid_y.to_numpy())

        eval_callback = xgb.callback.EvaluationMonitor(rank=0, period=5, show_stdv=True)
        early_stop = xgb.callback.EarlyStopping(
            rounds=100,
            # min_delta=1e-3,
            save_best=True,
            maximize=False,
            data_name="valid",
            metric_name="rmse",
        )
        learning_rates = [0.3, 0.1]
        learning_rate_scheduler = xgb.callback.LearningRateScheduler(learning_rates)

        if tune:

            use_optuna = False
            use_skl_api = True

            if use_optuna:

                # Hyperparameters common to all trials
                base_xgb_params = {'verbosity': 1,
                                   'device': 'cpu',
                                   'nthread': 11,
                                   'seed': SEED,
                                   'booster': 'gblinear',
                                   'objective': 'reg:squarederror',
                                   'eval_metric': 'rmse',
                                   'tree_method': 'exact',
                                   'scale_pos_weight': 1,
                                   'validate_parameters': True,
                                   }
                base_skl_api_params = {'verbosity': 1,
                                       'device': 'cpu',
                                       'random_state': SEED,
                                       'objective': 'reg:squarederror',
                                       'eval_metric': root_mean_squared_error,
                                       'importance_type': 'weight',
                                       'max_bin ': 512,
                                       'tree_method': 'exact',
                                       'scale_pos_weight': 1,
                                       'n_jobs': 1,
                                       }

                def objective(trial):

                    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'reg:squaredlogerror')

                    if use_skl_api:
                        params = {'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
                                  'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.7, log=False),
                                  'max_depth': trial.suggest_int('max_depth', 2, 16),
                                  'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                                  'subsample': trial.suggest_float('subsample', 0.5, .6),
                                  'max_leaves': trial.suggest_int('max_leaves', 2, 32),
                                  'min_child_weight': 0,
                                  # trial.suggest_float('min_child_weight', 1e-2, 3.0, log=True),
                                  'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                                  'reg_lambda': trial.suggest_float('lambda', 1e-2, 1.0, log=True),
                                  'reg_alpha': trial.suggest_float('alpha', 1e-6, 0.01, log=True)}  # Search space

                        params.update(base_skl_api_params)

                        regressor = xgb.XGBRegressor(**params,
                                                     callbacks=[early_stop])

                        regressor.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], verbose=True)

                        return rmse(valid_y, regressor.predict(valid_x))

                    else:
                        params = {'num_boost_round': trial.suggest_int('num_boost_round', 1000, 4000),
                                  'eta': trial.suggest_float('learning_rate', 0.001, 0.7, log=False),
                                  'max_depth': trial.suggest_int('max_depth', 2, 16),
                                  'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
                                  'subsample': trial.suggest_float('subsample', 0.5, .6),
                                  'max_leaves': trial.suggest_int('max_leaves', 2, 32),
                                  'min_child_weight': 0,
                                  # trial.suggest_float('min_child_weight', 1e-2, 3.0, log=True),
                                  'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                                  'lambda': trial.suggest_float('lambda', 1e-2, 1.0, log=True),
                                  'alpha': trial.suggest_float('alpha', 1e-6, 0.01, log=True)}  # Search space

                        params.update(base_xgb_params)

                        bst = xgb.train(params, dtrain,
                                        # num_boost_round=10000,
                                        evals=[(dvalid, 'valid')],
                                        early_stopping_rounds=100,
                                        verbose_eval=10,
                                        callbacks=[
                                            eval_callback,
                                            # early_stop,
                                            # pruning_callback,
                                        ])

                        print("Best score: ", bst.best_score, "at iteration: ", bst.best_iteration)

                        return bst.best_score

                # Run hyperparameter search
                study = optuna.create_study(direction='minimize', study_name=exp_name)
                study.optimize(objective,
                               n_trials=200,
                               callbacks=[],
                               n_jobs=11,
                               show_progress_bar=False)

                print('Completed hyperparameter tuning with best = {}.'.format(study.best_trial.value))
                params = {}
                params.update(base_skl_api_params) if use_skl_api else params.update(base_xgb_params)
                params.update(study.best_trial.params)

                # Re-run training with the best hyperparameter combination
                res: xgb.callback.TrainingCallback.EvalsLog = {}

                print('Re-running the best trial... params = {}'.format(params))
                if use_skl_api:
                    bst = xgb.XGBRegressor(**params)
                    bst.fit(train_x, train_y,
                            eval_set=[(valid_x, valid_y)],
                            verbose=True, )

                else:

                    bst = xgb.train(params, dtrain,
                                    # num_boost_round=10000,
                                    evals=[(dtrain, 'train'), (dvalid, 'valid')],
                                    evals_result=res,
                                    early_stopping_rounds=100,
                                    verbose_eval=True,
                                    # custom_metric=np_rmsle,
                                    callbacks=[eval_callback])

                bst.save_model('HousePrices/saved_models/best_xgb_model.json')

                optuna.visualization.plot_param_importances(study)

                optuna.visualization.plot_parallel_coordinate(study, params=['lambda', 'alpha'])

                # optuna.visualization.plot_intermediate_values(study)

                optuna.visualization.plot_contour(study, params=['lambda', 'alpha'])
                optuna.visualization.plot_contour(study, params=['learning_rate', 'alpha'])

                if params['eval_metric'] == 'rmsle':
                    epochs = len(res['train']['rmsle'])
                    x_axis = range(0, epochs)
                    plt.figure(figsize=(10, 5))
                    plt.plot(x_axis, res['train']['rmsle'], label='Train')
                    plt.plot(x_axis, res['valid']['rmsle'], label='Validation')
                    plt.legend()
                    plt.xlabel('Boosting Rounds')
                    plt.ylabel('RMSLE')
                    plt.title('XGBoost RMSLE over Boosting Rounds')
                    plt.show()
                elif params['eval_metric'] == 'rmse':
                    epochs = len(res['train']['rmse'])
                    x_axis = range(0, epochs)
                    plt.figure(figsize=(10, 5))
                    plt.plot(x_axis, res['train']['rmse'], label='Train')
                    plt.plot(x_axis, res['valid']['rmse'], label='Validation')
                    plt.legend()
                    plt.xlabel('Boosting Rounds')
                    plt.ylabel('RMSE')
                    plt.title('XGBoost RMSE over Boosting Rounds')
                    plt.show()

            # XGB search
            else:
                params = {
                    'n_estimators': randint(2000, 8000),
                    'max_depth': randint(2, 5),
                    'max_leaves': randint(2, 30),
                    'learning_rate': uniform(loc=0.001, scale=0.3),
                    'subsample': uniform(loc=0.5, scale=0.1),
                    # 'colsample_bytree': uniform(loc=0.5, scale=0.1),
                    # 'colsample_bynode ': uniform(loc=0.5, scale=0.5),
                    'grow_policy': ['lossguide', 'depthwise'],
                    # 'min_child_weight ': uniform(loc=0.0, scale=1.0), ??????????????????????
                    'gamma': uniform(loc=0.4, scale=.2),
                    'reg_lambda': uniform(loc=1e-2, scale=1.0),
                }
                regressor = xgb.XGBRegressor(tree_method="hist",
                                             booster="gbtree",
                                             objective="reg:squarederror",
                                             eval_metric="rmse",
                                             importance_type="weight",
                                             max_bin=256,

                                             colsample_bytree=0.5,
                                             scale_pos_weight=1,
                                             reg_alpha=0.007,

                                             device="cuda",
                                             random_state=SEED,

                                             n_jobs=1,
                                             verbosity=2)

                search = RandomizedSearchCV(regressor, params,
                                            n_iter=300,
                                            cv=5,
                                            scoring='neg_root_mean_squared_error',
                                            n_jobs=-1,
                                            return_train_score=True,
                                            verbose=5)

                # xy = xgb.QuantileDMatrix(train_x_gpu, y_train_y_gpu)

                train_x = cp.asarray(np.concatenate((train_x.to_numpy(), valid_x.to_numpy()), axis=0))
                train_y = cp.asarray(np.concatenate((train_y.to_numpy(), valid_y.to_numpy()), axis=0))

                bst = search.fit(X=train_x.get(),
                                 y=train_y.get())

                # {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 2000, 'subsample': 0.5}
                print("Best set of hyperparameters: ", search.best_params_)
                print("Best score: ", search.best_score_)

                cv_results = search.cv_results_
                sorted_res = np.argsort(cv_results['mean_test_score'])

                for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
                    print(-mean_score, params)
                print(sorted_res)

        # Train with best hyperparameters
        else:

            {'colsample_bytree': 0.5, 'gamma': 0.4, 'learning_rate': 0.024, # poss lower
             'max_depth': 3, 'max_leaves': 25, # 5
             'n_estimators': 2500, 'reg_alpha': 0.00657,
             'reg_lambda': 0.7, 'subsample': 0.5}

            {'gamma': 0.44518296766885146, 'grow_policy': 'lossguide', 'learning_rate': 0.0036219025650929817,
             'max_depth': 4, 'max_leaves': 12, 'n_estimators': 7562, 'reg_alpha': 0.007,
             'reg_lambda': 0.12335978311448657, 'subsample': 0.6}

            params = {'verbosity': 3, 'device': 'cpu', 'seed': 476, 'objective': 'reg:linear',
                      'eval_metric': 'rmse', 'tree_method': 'exact', 'scale_pos_weight': 1,
                      'n_estimators': 5068, 'learning_rate': 0.01,
                      'max_depth': 12, 'grow_policy': 'lossguide',
                      'subsample': 1,
                      'max_leaves': 31, 'gamma': 0.24, 'lambda': 0.0135,
                      'alpha': 0.0077}

            regressor = xgb.XGBRegressor(**params,
                                         # learning_rate=0.01,
                                         # n_estimators=6000,
                                         # max_depth=4,
                                         # min_child_weight=0,
                                         # gamma=0.6,
                                         # subsample=0.7,
                                         # colsample_bytree=0.7,
                                         # objective='reg:linear',
                                         # nthread=-1,
                                         # scale_pos_weight=1,
                                         # seed=SEED,
                                         # reg_alpha=0.00006,
                                         )

            bst = regressor.fit(train_x, train_y,
                                eval_set=[(train_x, train_y), (valid_x, valid_y)],
                                verbose=True,
                                )

            print("XGBoost Regressor Results: -------------------")
            calculate_metrics(bst, train_ds_pd, valid_ds_pd, predict_on_full_set=False, print_predictions=True)

        # Training history
        if visualize:
            epochs = len(bst.evals_result()['validation_0']['rmse'])
            x_axis = range(0, epochs)
            plt.figure(figsize=(10, 5))
            plt.plot(x_axis, bst.evals_result()['validation_0']['rmse'], label='Train')
            plt.plot(x_axis, bst.evals_result()['validation_1']['rmse'], label='Validation')
            plt.legend()
            plt.xlabel('Boosting Rounds')
            plt.ylabel('RMSE')
            plt.title('XGBoost RMSE over Boosting Rounds')
            plt.show()


def tf_neural_network(train_ds_pd, valid_ds_pd, test, ids, exp_name, submit=False):
    # Neural Network with TensorFlow

    device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    activation_func = 'gelu'

    # activation_func = 'relu'
    # activation_func = 'mish'

    def scheduler(epoch, lr):
        if epoch < 550:
            return lr
        else:
            return lr * np.exp(-0.1)

    num_epochs = 550
    learning_rate = 1.5e-4
    criterion = tf.keras.losses.MeanSquaredError()
    # criterion = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)

    x_train = train_ds_pd.drop('SalePrice', axis=1).values
    y_train = train_ds_pd['SalePrice'].values
    x_test = valid_ds_pd.drop('SalePrice', axis=1).values
    y_test = valid_ds_pd['SalePrice'].values

    train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_train, dtype=tf.float32),
                                                        tf.convert_to_tensor(y_train, dtype=tf.float32)))
    val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_test, dtype=tf.float32),
                                                      tf.convert_to_tensor(y_test, dtype=tf.float32)))

    train_loader = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size, drop_remainder=True)
    val_loader = val_dataset.shuffle(buffer_size=len(val_dataset)).batch(batch_size, drop_remainder=True)

    class Net(tf.keras.Model):
        def __init__(self):
            super(Net, self).__init__()

            self.input_layer = tf.keras.layers.Dense(234, activation=activation_func)

            # self.feature_extractor = []
            # for i in range(5):
            #     self.hidden_layers.append(tf.keras.layers.Dense(2048, activation=activation_func))
            #     self.hidden_layers.append(tf.keras.layers.Dropout(0.2))

            self.hidden_layers = []
            for i in range(7):
                self.hidden_layers.append(tf.keras.layers.Dense(1024, activation=activation_func))
                self.hidden_layers.append(tf.keras.layers.Dropout(0.2))
                self.hidden_layers.append(tf.keras.layers.BatchNormalization())

            self.additional_layers = []
            for i in range(2):
                self.additional_layers.append(tf.keras.layers.Dense(512, activation=activation_func))
                self.additional_layers.append(tf.keras.layers.Dropout(0.2))

            self.output_layer = tf.keras.layers.Dense(1)

        def call(self, x):
            x = self.input_layer(x)
            # for layer in self.feature_extractor:
            #     x = layer(x)
            for layer in self.hidden_layers:
                x = layer(x)
            for layer in self.additional_layers:
                x = layer(x)

            return self.output_layer(x)

    model = Net()

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            with tf.GradientTape() as tape:
                outputs = model(batch_x, training=True)
                loss = criterion(batch_y, tf.squeeze(outputs))

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            train_loss += loss.numpy()

        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        val_loss = 0
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x, training=False)
            loss = criterion(batch_y, tf.squeeze(outputs))
            val_loss += loss.numpy()

        val_losses.append(val_loss / len(val_loader))

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Validation Loss: {val_loss / len(val_loader):.4f}"
        )

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # Making predictions
    x_train_tensor = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)

    y_train_pred = model(x_train_tensor, training=False).numpy()
    y_test_pred = model(x_test_tensor, training=False).numpy()

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f'Train R-squared: {train_r2 * 100:.2f}%')
    print(f'Test R-squared: {test_r2 * 100:.2f}%')
    print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
    print(f'Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    print(f'Test Mean Absolute Error (MAE): {test_mae:.2f}')

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual', marker='o', linestyle='None')
    plt.plot(y_test_pred, label='Predicted', marker='x', linestyle='None')
    plt.legend()
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted Values')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.show()

    print(model.summary())
    print(model.get_config())
    print(optimizer.get_config())

    model.save("HousePrices/saved_models/nn_model_" + exp_name + ".tf",
               save_format='tf')

    exp_logger.save({"Id": exp_name, "Model": "TF Neural Network",
                     "Train_R2": train_r2, "Validation_R2": test_r2, "RMSE": test_rmse,
                     "Hyperparameters": f"Epochs: {num_epochs}, "
                                        f"Batch Size: {batch_size}, "
                                        f"Activation Function: {activation_func}, "
                                        f"Optimizer: Adam, Learning Rate: {learning_rate}, Loss: MSE, "
                                        f"Dropout: 0.2"})

    make_submission(model, test, ids, exp_name)
