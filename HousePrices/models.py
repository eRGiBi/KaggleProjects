import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import stats
from sklearn.cluster import DBSCAN
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest

import tensorflow as tf
import tensorflow_decision_forests as tfdf

import ydf

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

from ExperimentLogger import ExperimentLogger

warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

SEED = 47612
exp_logger = ExperimentLogger('HousePrices/submissions/experiment_aggregate.csv')


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


def tf_decision_forests(train_ds_pd, valid_ds_pd):
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
                                         verbose=2
                                         )

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

    # exp_logger.save({"Id": exp_name, "Model": "TensorFlow Decision Forests",
    #                  "Train_R2": train_r2, "Validation_R2": valid_r2, "RMSE": RMSE,
    #                  "Hyperparameters": model.predefined_hyperparameters()})

    # model.save("./saved_models/")


def sklearndf_random_forest(data, valid_ds_pd, test, ids, exp_name):
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


def yggdrassil_random_forest(train_ds_pd, valid_ds_pd, test, ids, exp_name, tune=False):
    if not tune:

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
                         }
                        )

        logs = model.hyperparameter_optimizer_logs()
        top_score = max(t.score for t in logs.trials)
        selected_trial = [t for t in logs.trials if t.score == top_score][0]

        best_trial_index = logs.trials.index(selected_trial)
        model.plot_tree(best_trial_index)

        print("Best hyperparameters:")
        print(selected_trial.params)

        model.save("saved_models/model_" + exp_name)

        print("Yggdrasil Submission: -------------------")
        make_submission(model, test, ids, exp_name)


def tf_neural_network(train_ds_pd, valid_ds_pd, test, ids, exp_name, submit=False):
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

    model.save("HousePrices/saved_models/nn_model_" + exp_name + ".keras")

    exp_logger.save({"Id": exp_name, "Model": "TF Neural Network",
                     "Train_R2": train_r2, "Validation_R2": test_r2, "RMSE": test_rmse,
                     "Hyperparameters": f"Epochs: {num_epochs}, "
                                        f"Batch Size: {batch_size}, "
                                        f"Activation Function: {activation_func}, "
                                        f"Optimizer: Adam, Learning Rate: {learning_rate}, Loss: MSE, "
                                        f"Dropout: 0.2"})

    make_submission(model, test, ids, exp_name)
