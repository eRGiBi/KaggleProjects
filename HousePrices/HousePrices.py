
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
import tensorflow_decision_forests as tfdf

import ydf

import seaborn as sns
import matplotlib.pyplot as plt

from ExperimentLogger import ExperimentLogger

SEED = 123456
np.random.seed(SEED)
tf.random.set_seed(SEED)

def make_submission(model, test_data):
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data,
        task=tfdf.keras.Task.REGRESSION)

    preds = model.predict(test_ds)
    output = pd.DataFrame({'Id': ids,
                           'SalePrice': preds.squeeze()})

    print(output.head())

    sample_submission_df = pd.read_csv('./data/sample_submission.csv')
    sample_submission_df['SalePrice'] = model.predict(test_ds)
    sample_submission_df.to_csv('./submissions/submission_' + exp_name + '.csv', index=False)
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

    one_hot_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'Neighborhood', 'Condition1',
                    'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                    'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                    'CentralAir', 'Electrical', 'GarageType', 'MiscFeature', 'SaleType',
                    'SaleCondition']

    encoder = OneHotEncoder(sparse_output=False)

    one_hot_enc = encoder.fit_transform(data[one_hot_cols])

    one_hot_df = pd.DataFrame(one_hot_enc, columns=encoder.get_feature_names_out(one_hot_cols))

    df_encoded = pd.concat([data, one_hot_df], axis=1)

    df_encoded = df_encoded.drop(one_hot_cols, axis=1)

    # Binary encoding
    # large number of unique categories

    # encoder = ce.BinaryEncoder(cols=['Country'])
    # data = encoder.fit_transform(data)

    return df_encoded


def preprocess_data(data):
    missing_val_count_by_column = (data.isnull().sum())
    print(missing_val_count_by_column[missing_val_count_by_column > 0])

    data['LotShape'] = str(data['LotShape'])

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

    return data


def explore_data(data, plot=False):
    if plot:
        # plt.figure(figsize=(9, 8))
        # sns.displot(data['SalePrice'], color='g', bins=100);
        # plt.show()

        df_num = data.select_dtypes(include=['float64', 'int64'])
        print(df_num.head())
        df_num.hist(figsize=(8, 10), bins=50, xlabelsize=4, ylabelsize=4);
        plt.show()

    print("Data Shape, Sample, info: -------------------")

    print(data.shape)
    print(data.head())
    print("Data info:")
    print(data.info())
    print("Data description:")
    print(data.describe())
    print()


if __name__ == '__main__':

    print("TensorFlow v" + tf.__version__)
    print("TensorFlow Decision Forests v" + tfdf.__version__)

    use_tfdf = False
    use_sklrean_df = False
    use_yggdrasil = True

    # Check if TensorFlow can access the GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Ensure that TensorFlow is not initialized more than once
    if not tf.executing_eagerly():
        tf.compat.v1.reset_default_graph()

    now = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    exp_name = "experiment_" + str(now)
    exp_logger = ExperimentLogger('submissions/' + exp_name + '.csv')

    # Load the data #############################################

    data = pd.read_csv('data/train.csv')
    data = data.drop('Id', axis=1)

    test = pd.read_csv('data/test.csv')
    ids = test.pop('Id')
    # test.insert(loc=0, column='SalePrice', value=data['SalePrice'], allow_duplicates=True)

    # Data exploration ###########################################

    # explore_data(data)
    explore_data(test, plot=False)

    # Data preprocessing ###########################################
    print("Encoded data: -------------------")

    data.dropna(axis=0, subset=['SalePrice'], inplace=True)

    data = preprocess_data(data)
    test = preprocess_data(test)

    # Encode data
    train_encoded = encode_data(data)
    test_encoded = encode_data(test)

    explore_data(train_encoded, plot=False)

    # Impute missing values
    print("Imputed data: -------------------")

    my_imputer = SimpleImputer()
    data = my_imputer.fit_transform(train_encoded)
    test = my_imputer.fit_transform(test_encoded)

    data = pd.DataFrame(data, columns=train_encoded.columns)
    test = pd.DataFrame(test, columns=test_encoded.columns)

    explore_data(data, plot=False)

    # Check if the columns match
    # for train_col, test_col in zip(data.columns, test.columns):
    #     if train_col != test_col:
    #         print(f"Train column: {train_col}, Test column: {test_col}")

    # for i in range(len(data.columns) - 1):
    #     if data.columns[i] != test.columns[i + 1]:
    #         print(f"Train column: {data.columns[i]}, Test column: {test.columns[i]}")


    # Split the data to train and validation #############################################

    def split_dataset(dataset, test_ratio=0.10):
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]

    train_ds_pd, valid_ds_pd = split_dataset(data)
    print("{} examples in training, {} examples in testing.".format(
        len(train_ds_pd), len(valid_ds_pd)))


    # Visualize the data #########################################
    # df_numericals = data.select_dtypes(include=['float64', 'int64', 'int32', 'float32'])
    #
    # df_numericals.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    # plt.show()


    # Random Forest regression with TensorFlow Decision Forests
    # with and without hyperparameter tuning

    if use_tfdf:

        label = 'SalePrice'
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
                                             bootstrap_training_dataset=True, bootstrap_size_ratio=1.0,
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
        train_predictions = model.predict(train_ds)
        for i in range(10):
            print(f"Train Prediction: {train_predictions[i]}, SalePrice: {train_ds_pd[label].iloc[i]}")
        print()
        train_r2 = r2_score(train_ds_pd[label], train_predictions)
        print(f'Train R-squared: {train_r2 * 100:.2f}%')
        RMSE = mean_squared_error(train_ds_pd[label], train_predictions, squared=False)
        print(f'Train RMSE: {RMSE:.2f}')
        print()

        valid_predictions = model.predict(valid_ds)
        for i in range(10):
            print(f"Validation Prediction: {valid_predictions[i]}, SalePrice: {valid_ds_pd[label].iloc[i]}")
        print()
        valid_r2 = r2_score(valid_ds_pd[label], valid_predictions)
        print(f'Validation R-squared: {valid_r2 * 100:.2f}%')
        RMSE = mean_squared_error(valid_ds_pd[label], valid_predictions, squared=False)
        print(f'Validation RMSE: {RMSE:.2f}')
        print()


        # tfdf.model_plotter.plot_model(model, tree_idx=0, max_depth=None)
        plt.show()

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

        # Mean decrease in AUC of class 1 vs. the others.
        variable_importance_metric = "NUM_AS_ROOT"
        variable_importances = inspector.variable_importances()[variable_importance_metric]

        # Extract the feature name and importance values.
        feature_names = [vi[0].name for vi in variable_importances]
        feature_importances = [vi[1] for vi in variable_importances]
        # The feature are ordered in decreasing importance value.
        feature_ranks = range(len(feature_names))

        bar = plt.barh(feature_ranks, feature_importances, label=[str(x) for x in feature_ranks])
        plt.yticks(feature_ranks, feature_names)
        plt.gca().invert_yaxis()

        # TODO: Replace with "plt.bar_label()" when available.
        # Label each bar with values
        for importance, patch in zip(feature_importances, bar.patches):
            plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{importance:.4f}", va="top")

        plt.xlabel(variable_importance_metric)
        plt.title("NUM AS ROOT of the class 1 vs the others")
        plt.tight_layout()
        plt.show()


        exp_logger.save({"Id": now, "Model": "TensorFlow Decision Forests",
                         "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                         "Hyperparameters": model.predefined_hyperparameters()})


        # Error exploration

        # Check the model input keys and semantics #################
        sample_inputs = {
            'LotFrontage': tf.constant([80.0], dtype=tf.float32),
            'LotArea': tf.constant([9600], dtype=tf.int64),
        }


        # Predict on the test data ###################################



        # model.save("./saved_models/")




    if use_sklrean_df:

        # Random Forest with scikit-learn

        print("SKLearn Random Forest Regressor Results: -------------------")

        x_train = np.array(data)
        y_train = data['SalePrice']

        x_test = np.array(valid_ds_pd)
        y_test = valid_ds_pd['SalePrice']

        rf_reg = RandomForestRegressor(n_estimators=100, random_state=SEED)
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

    if use_yggdrasil:

        learner = ydf.RandomForestLearner(label="SalePrice",
                                          task=ydf.Task.REGRESSION,
                                          include_all_columns=True,
                                          features=[
                                              # ydf.Feature("PoolArea", monotonic=+1),
                                          ],
                                          )

        model = learner.train(ds=train_ds_pd, verbose=2)

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
        analysis.to_file("results/analysis.html")

        print("Benchmark Results: -------------------")
        # print(model.benchmark(valid_ds_pd))
        # print(model.to_cpp())

        # model.save("/my_model")

        print("Yggdrasil Submission: -------------------")
        print(data.shape, test.shape)
        make_submission(model, test)

        # Hyperparameter tuning

        tuner = ydf.RandomForestTuner(task=ydf.Task.REGRESSION, num_trials=10)

        logs = model.hyperparameter_optimizer_logs()
        top_score = max(t.score for t in logs.trials)
        selected_trial = [t for t in logs.trials if t.score == top_score][0]
        print(selected_trial.params)  # This is a dictionary

