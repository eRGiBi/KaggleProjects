
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

SEED = 476

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

def make_submission(model, test_data, tune=False):
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

        self.main()

    def main(self):

        # Ensure that TensorFlow is not initialized more than once
        if not tf.executing_eagerly():
            tf.compat.v1.reset_default_graph()

        now = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        exp_name = "experiment_" + str(now)
        # exp_logger = ExperimentLogger('submissions/experiment_aggregate.csv')

        # Load the data #############################################

        data = pd.read_csv('HousePrices/data/train.csv')
        data = data.drop('Id', axis=1)

        test = pd.read_csv('HousePrices/data/test.csv')
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

        if self.algorithm == 'tfdf':

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

                # Show the full evaluation report
                print("Full evaluation report:")
                print(evaluation)

                print("Analytics Results: -------------------")
                print()
                analysis = model.analyze(valid_ds_pd, sampling=0.1)
                print(analysis)
                analysis.to_file("results/analysis.html")

                # best_tree = max(tree in tree for tree in model.iter_trees())
                # model.plot_tree()

                print("Benchmark Results: -------------------")
                # print(model.benchmark(valid_ds_pd))
                # print(model.to_cpp())

                model.save("./saved_models/best_hyp_model_" + exp_name)

                train_r2, valid_r2, RMSE = calculate_metrics(model, train_ds_pd, valid_ds_pd)

                # exp_logger.save({"Id": now, "Model": "Tuned Yggdrasil Random Forest",
                #                  "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                #                  "Hyperparameters": model.data_spec()})

                print("Yggdrasil Submission: -------------------")
                # print(data.shape, test.shape)
                # make_submission(model, test)

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

                # exp_logger.save({"Id": now, "Model": "Tuned Yggdrasil Random Forest",
                #                  "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
                #                  "Hyperparameters": model.predefined_hyperparameters()})

                model.save("saved_models/model_" + exp_name)

                print("Yggdrasil Submission: -------------------")
                # print(data.shape, test.shape)
                # make_submission(model, test)

                logs = model.hyperparameter_optimizer_logs()
                top_score = max(t.score for t in logs.trials)
                selected_trial = [t for t in logs.trials if t.score == top_score][0]

                best_trial_index = logs.trials.index(selected_trial)
                model.plot_tree(best_trial_index)

                print("Best hyperparameters:")
                print(selected_trial.params)


        elif self.algorithm == 'NN':

            batch_size = 64

            x_train = train_ds_pd.drop('SalePrice', axis=1).values
            y_train = train_ds_pd['SalePrice'].values
            x_test = valid_ds_pd.drop('SalePrice', axis=1).values
            y_test = valid_ds_pd['SalePrice'].values

            train_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_train, dtype=tf.float32),
                                                                tf.convert_to_tensor(y_train, dtype=tf.float32)))
            val_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(x_test, dtype=tf.float32),
                                                              tf.convert_to_tensor(y_test, dtype=tf.float32)))

            train_loader = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)
            val_loader = val_dataset.batch(batch_size)

            class Net(tf.keras.Model):
                def __init__(self):
                    super(Net, self).__init__()
                    self.fc1 = tf.keras.layers.Dense(512, activation='relu', input_shape=(234,))
                    self.fc3 = tf.keras.layers.Dense(512, activation='relu')
                    self.fc4 = tf.keras.layers.Dense(256, activation='relu')
                    self.fc5 = tf.keras.layers.Dense(1)

                def call(self, x):
                    x = self.fc1(x)
                    x = self.fc3(x)
                    x = self.fc4(x)
                    x = self.fc5(x)
                    return x

            model = Net()

            criterion = tf.keras.losses.MeanSquaredError()
            # criterion = tf.keras.losses.MeanAbsoluteError()
            optimizer = tf.keras.optimizers.Adam(learning_rate=2.5e-4, epsilon=1e-5)
            device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
            print(f"Using device: {device}")

            num_epochs = 50
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

            # Make predictions
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

