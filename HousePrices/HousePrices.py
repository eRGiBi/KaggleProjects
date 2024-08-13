
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
import tensorflow_decision_forests as tfdf

import ydf

import seaborn as sns
import matplotlib.pyplot as plt

from ExperimentLogger import ExperimentLogger

import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


SEED = 47612

def calculate_metrics(model, train_ds_pd, valid_ds_pd, label='SalePrice'):

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

def make_submission(model, test_data, ids, exp_name='experiment'):

    # X_test = df_test.drop(, axis=1)

    # test_data = test_data.fillna(test_data.mean())

    # test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
    #     test_data,
    #     task=tfdf.keras.Task.REGRESSION)

    # preds = model.predict(test_data)
    #
    # output = pd.DataFrame({'Id': ids,
    #                        'SalePrice': preds.squeeze()}
    #                       )
    #
    # print(output.head())

    sample_submission_df = pd.read_csv('./HousePrices/data/sample_submission.csv')
    sample_submission_df['SalePrice'] = model.predict(test_data)
    sample_submission_df.to_csv('./HousePrices/submissions/submission_' + exp_name + '.csv', index=False)
    print(sample_submission_df.head())


def encode_data(data):
    # Label encoding
    # LotShape, LandContour, Utilities, LotConfig, LandSlope, ExterQual, ExterCond, BsmtQual,
    # BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, HeatingQC, KitchenQual, Functional,
    # FireplaceQu, GarageFinish, GarageQual, GarageCond, PavedDrive, PoolQC, Fence

    label_encoder = LabelEncoder()
    lab_enc_cols = ['LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional',
                    'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                    'PoolQC', 'Fence']


    for col in lab_enc_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # One hot encoding
    # MSSubClass, MSZoning, Street, Alley, Neighborhood, Condition1, Condition2, BldgType, HouseStyle,
    # RoofStyle, RoofMatl, Exterior1st, Exterior2nd, MasVnrType, Foundation, Heating, CentralAir,
    # Electrical, GarageType, MiscFeature, SaleType, SaleCondition

    #
    one_hot_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'Neighborhood', 'Condition1',
                    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                    'CentralAir', 'Electrical', 'GarageType', 'MiscFeature', 'SaleType',
                    'SaleCondition'
                    ]
    # print(data['MSZoning'].head())

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_enc = encoder.fit_transform(data[one_hot_cols])

    one_hot_df = pd.DataFrame(one_hot_enc, columns=encoder.get_feature_names_out(one_hot_cols))

    df_encoded = pd.concat([data, one_hot_df], axis=1)

    df_encoded = df_encoded.drop(one_hot_cols, axis=1)

    # print(data['MSZoning'].head())
    #
    # df_encoded = df_encoded.drop('MSZoning', axis=1)

    # Binary encoding
    # large number of unique categories

    # encoder = ce.BinaryEncoder(cols=['Country'])
    # data = encoder.fit_transform(data)

    return df_encoded


def preprocess_data(train, test):


    # Outliers
    train.drop(train[(train['OverallQual'] < 5) & (train['SalePrice'] > 200000)].index, inplace=True)
    train.drop(train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 300000)].index, inplace=True)
    train.reset_index(drop=True, inplace=True)

    train_labels = train['SalePrice'].reset_index(drop=True)
    train_features = train.drop(['SalePrice'], axis=1)
    test_features = test

    all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
    print(all_features.shape)

    all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
    all_features['YrSold'] = all_features['YrSold'].astype(str)
    all_features['MoSold'] = all_features['MoSold'].astype(str)

    def handle_missing(features):
        """https://www.kaggle.com/code/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition#Feature-Engineering"""

        # the data description states that NA refers to typical ('Typ') values
        features['Functional'] = features['Functional'].fillna('Typ')
        # Replace the missing values in each of the columns below with their mode
        features['Electrical'] = features['Electrical'].fillna("SBrkr")
        features['KitchenQual'] = features['KitchenQual'].fillna("TA")
        features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
        features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
        features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
        # features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
        # features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

        # the data description stats that NA refers to "No Pool"
        features["PoolQC"] = features["PoolQC"].fillna("None")
        # Replacing the missing values with 0, since no garage = no cars in garage
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
            features[col] = features[col].fillna(0)
        # Replacing the missing values with None
        for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
            features[col] = features[col].fillna('None')
        # NaN values for these categorical basement features, means there's no basement
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
            features[col] = features[col].fillna('None')

        # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
        features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))

        # We have no particular intuition around how to fill in the rest of the categorical features
        # So we replace their missing values with None
        objects = []
        for i in features.columns:
            if features[i].dtype == object:
                objects.append(i)
        features.update(features[objects].fillna('None'))

        # And we do the same thing for numerical features, but this time with 0s
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numeric = []
        for i in features.columns:
            if features[i].dtype in numeric_dtypes:
                numeric.append(i)
        features.update(features[numeric].fillna(0))

        return features

    # Deal with missing values

    # missing_val_count_by_column = (data.isnull().sum())
    # print(missing_val_count_by_column[missing_val_count_by_column > 0])

    all_features = handle_missing(all_features)

    print("Encoded data: -------------------")
    all_features = encode_data(all_features)

    # Impute missing values
    # print("Imputed data: -------------------")

    # my_imputer = SimpleImputer()
    # # data = my_imputer.fit_transform(data)
    # # test = my_imputer.fit_transform(test)
    # all_features = my_imputer.fit_transform(all_features)

    # data = data.drop((missing_data[missing_data['Total'] > 1]).index, 1)
    # df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
    # df_train.isnull().sum().max()

    print("Missing data: -------------------")
    total = all_features.isnull().sum().sort_values(ascending=False)
    percent = (all_features.isnull().sum() / all_features.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    print(missing_data.head(20))
    print()


    print("Recombined data: -------------------")
    train = all_features.iloc[:len(train_labels), :]
    train.insert(loc=0, column='SalePrice', value=train_labels)
    # train = pd.concat([train, train_labels], axis=1)
    test = all_features.iloc[len(train_labels):, :]

    print(train.shape, test.shape)

    # for col in data.columns:
    #     if data[col].dtype != 'int64' and data[col].dtype != 'int32' and data[col].dtype != 'float64':
    # print(col, data[col].dtype)
    # print(data[col].unique())

    # for col in df_encoded.columns:
    #     if df_encoded[col].dtype == 'int64' or df_encoded[col].dtype == 'int32':
    #         df_encoded[col] = df_encoded[col].astype('float64')
    # for col in df_encoded.columns:
    #     for i in range(len(df_encoded[col])):
    #         # print(f"Checking {col} at index {i}")
    #         # print(df_encoded[col].iloc[i])
    #         if df_encoded[col].iloc[i] == 'NA':
    #             df_encoded[col].iloc[i] = 0
    #             print(f"Replaced NA in {col} with 0")
    #         try:
    #             tensor = tf.convert_to_tensor(i)
    #             print("Conversion successful!")
    #         except Exception as e:
    #             print(f"Error during conversion: {e}")

    return train, test


def explore_data(data, plot=False, test=False):
    if plot:
        # plt.figure(figsize=(9, 8))
        # sns.displot(data['SalePrice'], color='g', bins=100);
        # plt.show()

        # Most important features (according to previous analysis)
        num_cols = ['GrLivArea',  'TotalBsmtSF',  'YearBuilt']
        num_sec_cols = ['GarageArea', 'BsmtFinSF1', 'LotArea', '2ndFlrSF', 'FullBath', '1stFlrSF'
                    'YearRemodAdd']

        cat_cols = ['OverallQual','GarageCars', 'ExterQual',]

        sns.set()
        sns.pairplot(data[num_cols], size=2.5)
        plt.show()

        for col in num_cols:
            col_data =  pd.concat([data['SalePrice'], data[col]], axis=1)
            col_data.plot.scatter(x=col, y='SalePrice', ylim=(0,800000));
            plt.show()

        for col in cat_cols:
            col_data = pd.concat([data['SalePrice'], data[col]], axis=1)
            f, ax = plt.subplots(figsize=(8, 6))
            fig = sns.boxplot(x=col, y="SalePrice", data=col_data)
            fig.axis(ymin=0, ymax=800000);
            plt.show()

        df_num = data.select_dtypes(include=['float64', 'int64'])
        # print(df_num.head())
        # df_num.hist(figsize=(8, 10), bins=50, xlabelsize=4, ylabelsize=4);
        # plt.show()

        # Correlation matrix
        corrmat = df_num.corr()
        plt.figure(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=.8, square=True)
        plt.show()

        # SalePrice correlation matrix
        k = 10
        cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
        cm = np.corrcoef(data[cols].values.T)
        sns.set(font_scale=1)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values, xticklabels=cols.values)
        plt.show()

    print("Data Shape, Sample, info: -------------------")
    print(data.shape)
    print(data.head())
    print()
    print("Data info:")
    print(data.info())
    print()
    print("Data description:")
    print(data.describe())
    print()

    object_cols = data.select_dtypes(include=['object'])
    print("Still Object typed columns:", object_cols)

    if not test:
        # Skewness and kurtosis
        print("Skewness: %f" % data['SalePrice'].skew())
        print("Kurtosis: %f" % data['SalePrice'].kurt())


class HousePricesRegression:

    def __init__(self, algorithm, tune=False):

        self.algorithm = algorithm
        self.tune = tune

        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        print("TensorFlow v" + tf.__version__)
        print("TensorFlow Decision Forests v" + tfdf.__version__)

        # Check if TensorFlow can access the GPU
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


    def run(self):

        # Ensure that TensorFlow is not initialized more than once
        if not tf.executing_eagerly():
            tf.compat.v1.reset_default_graph()

        now = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        exp_name = "experiment_" + str(now)
        exp_logger = ExperimentLogger('HousePrices/submissions/experiment_aggregate.csv')

        # Load the data #############################################

        data = pd.read_csv('HousePrices/data/train.csv')
        data = data.drop('Id', axis=1)

        test = pd.read_csv('HousePrices/data/test.csv')
        ids = test.pop('Id')
        # test.insert(loc=0, column='SalePrice', value=data['SalePrice'], allow_duplicates=True)

        # Data exploration ###########################################
        explore_data(data, plot=True)
        # explore_data(test, plot=False, test=True)

        sns.heatmap(data.isnull(), cmap='viridis')
        plt.show()


        # Data preprocessing ###########################################
        print("Preprocessed data: -------------------")

        data, test = preprocess_data(data, test)
        data = pd.DataFrame(data, columns=data.columns)

        explore_data(data, plot=True)

        sns.heatmap(data.isnull(), cmap='viridis')
        plt.show()

        # Univariate analysis

        # saleprice_scaled = StandardScaler().fit_transform(data['SalePrice'][:,np.newaxis]);
        # low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
        # high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
        # print('outer range (low) of the distribution:')
        # print(low_range)
        # print('\nouter range (high) of the distribution:')
        # print(high_range)

        # Bivariate analysis

        # data.sort_values(by='GrLivArea', ascending=False)[:2]
        # data = data.drop(data[data['Id'] == 1299].index)
        # data = data.drop(data[data['Id'] == 524].index)


        # Split the data to train and validation #############################################

        def split_dataset(dataset, test_ratio=0.15):
            test_indices = np.random.rand(len(dataset)) < test_ratio
            return dataset[~test_indices], dataset[test_indices]

        train_ds_pd, valid_ds_pd = split_dataset(data)
        print("{} examples in training, {} examples in testing.".format(
            len(train_ds_pd), len(valid_ds_pd)))



        if self.algorithm == 'tfdf':

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

            # Hyperparameters to optimize.
            tuner.choice("max_depth", [4, 5, 7, 16, 32])
            tuner.choice("num_trees", [50, 100, 200, 500])

            print(tuner.train_config())

            model = tfdf.keras.RandomForestModel(tuner=tuner, task=tfdf.keras.Task.REGRESSION,
                                                 bootstrap_training_dataset=True,
                                                 bootstrap_size_ratio=1.0,
                                                 categorical_algorithm='CART', #RANDOM
                                                 growing_strategy='LOCAL', #BEST_FIRST_GLOBAL
                                                 honest=False,
                                                 min_examples=1,
                                                 missing_value_policy='GLOBAL_IMPUTATION',
                                                 num_candidate_attributes=0,
                                                 random_seed=SEED,
                                                 winner_take_all=True,
                                                 )

            model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, verbose=2)
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


            # Variable importances
            inspector = model.make_inspector()
            inspector.evaluation()

            print(f"Available variable importances:")
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

            # exp_logger.save({"Id": now, "Model": "TensorFlow Decision Forests",
            #                  "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
            #                  "Hyperparameters": model.predefined_hyperparameters()})

            # model.save("./saved_models/")

        if self.algorithm == 'sklearndf':

            # Random Forest with scikit-learn

            print("SKLearn Random Forest Regressor Results: -------------------")

            x_train = np.array(data)
            y_train = data['SalePrice']

            x_test = np.array(valid_ds_pd)
            y_test = valid_ds_pd['SalePrice']

            rf_reg = RandomForestRegressor(n_estimators=750, random_state=SEED)
            rf_reg.fit(x_train, y_train)

            y_pred_train = rf_reg.predict(x_train)
            y_pred_test = rf_reg.predict(x_test)

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

            print()

        elif self.algorithm == 'yggdf':

            if not self.tune:

                # Best found hyperparameters

                learner = ydf.RandomForestLearner(label="SalePrice",
                                                  task=ydf.Task.REGRESSION,
                                                  include_all_columns=True,
                                                  features=[
                                                      # ydf.Feature("PoolArea", monotonic=+1),
                                                      ],
                                                  random_seed=SEED,
                                                  num_trees=750,
                                                  max_depth=16,
                                                  bootstrap_size_ratio=0.925,
                                                  categorical_algorithm="RANDOM",
                                                  growing_strategy="LOCAL",
                                                  winner_take_all=False,
                                                  )

                model = learner.train(ds=train_ds_pd, verbose=2)

                print(model.describe())
                # print(model)

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
                # print(model.benchmark(valid_ds_pd))
                # print(model.to_cpp())

                # model.save("./HousePrices/saved_models/best_hyp_model_" + exp_name)

                train_r2, valid_r2, RMSE = calculate_metrics(model, train_ds_pd, valid_ds_pd)

                exp_logger.save({"Id": now, "Model": "Tuned Yggdrasil Random Forest",
                                 "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                                 "Hyperparameters": "num_trees=750, max_depth=16, bootstrap_size_ratio=0.925, "}
                                )

                print("Yggdrasil Submission: -------------------")
                print(data.shape, test.shape)
                make_submission(model, test, ids, exp_name)

            else:

                print("Hyperparameter tuning: -------------------")

                tuner = ydf.RandomSearchTuner(num_trials=100)

                # Hyperparameters to optimize.
                tuner.choice("max_depth", [16, 32])
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
                analysis.to_file("results/" + exp_name + "analysis.html")



                print("Benchmark Results: -------------------")
                # print(model.benchmark(valid_ds_pd))
                # print(model.to_cpp())

                train_r2, valid_r2, RMSE = calculate_metrics(model, train_ds_pd, valid_ds_pd, label)

                exp_logger.save({"Id": now, "Model": "Tuned Yggdrasil Random Forest",
                                 "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                                 "Hyperparameters": model.predefined_hyperparameters()}
                                )

                model.save("saved_models/model_" + exp_name)

                print("Yggdrasil Submission: -------------------")
                print(data.shape, test.shape)
                make_submission(model, test, ids, exp_name)

                logs = model.hyperparameter_optimizer_logs()
                top_score = max(t.score for t in logs.trials)
                selected_trial = [t for t in logs.trials if t.score == top_score][0]

                best_trial_index = logs.trials.index(selected_trial)
                model.plot_tree(best_trial_index)

                print("Best hyperparameters:")
                print(selected_trial.params)


        elif self.algorithm == 'NN':

            # Neural Network with TensorFlow

            device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
            print(f"Using device: {device}")

            # Hyperparameters
            batch_size = 128
            activation_func = 'gelu'
            # activation_func = 'relu'
            # activation_func = 'mish'

            num_epochs = 450
            learning_rate = 3.5e-4
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

            # Set model to inference mode (not strictly necessary in TensorFlow as it handles this automatically)
            # Making predictions on training and testing data
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
            print(model.loss)

            exp_logger.save({"Id": now, "Model": "TF Neural Network",
                             "Train_R2": train_r2, "Validation_R2": test_r2, "RMSE": test_rmse,
                             "Hyperparameters": f"Epochs: {num_epochs}, "
                                                f"Batch Size: {batch_size}, "
                                                f"Activation Function: {activation_func}, "
                                                f"Optimizer: Adam, Learning Rate: {learning_rate}, Loss: MSE, "
    # f"Layers: {str(len(model.hidden_layers))} * {len(model.hidden_layers[0])} + {str(len(model.additional_layers))} * {len(model.additional_layers[0])}, "
                                                f"Dropout: 0.2"})

            make_submission(model, test, ids, exp_name)

            exit()
