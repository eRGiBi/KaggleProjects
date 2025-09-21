import xgboost as xgb
import cupy as cp
from sklearn.metrics import root_mean_squared_error


# XGBoost
def train_xgboost(train_x, train_y, valid_x, valid_y, seed):
    """"""
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
            base_xgb_params = {
                'verbosity': 1,
                'device': 'cpu',
                'nthread': 11,
                'seed': seed,
                'booster': 'gblinear',
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'exact',
                'scale_pos_weight': 1,
                'validate_parameters': True,
            }
            base_skl_api_params = {
                'verbosity': 1,
                'device': 'cpu',
                'random_state': seed,
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
                                         random_state=seed,

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
        {'colsample_bytree': 0.5, 'gamma': 0.4, 'learning_rate': 0.024,  # poss lower
         'max_depth': 3, 'max_leaves': 25,  # 5
         'n_estimators': 2500, 'reg_alpha': 0.00657,
         'reg_lambda': 0.7, 'subsample': 0.5}

        params = {'gamma': 0.44518296766885146, 'grow_policy': 'lossguide', 'learning_rate': 0.0036219025650929817,
                  'max_depth': 4, 'max_leaves': 12, 'n_estimators': 7562, 'reg_alpha': 0.007,
                  'reg_lambda': 0.12335978311448657, 'subsample': 0.6}

        regressor = xgb.XGBRegressor(**params,
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