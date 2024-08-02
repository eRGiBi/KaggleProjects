import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
import tensorflow_decision_forests as tfdf

import seaborn as sns
import matplotlib.pyplot as plt

SEED = 123456

if __name__ == '__main__':

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

        data['LotShape'] = str(data['LotShape'])

        for col in data.columns:
            if data[col].dtype != 'int64' and data[col].dtype != 'int32' and data[col].dtype != 'float64':
                print(col, data[col].dtype)
                print(data[col].unique())

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

    print("TensorFlow v" + tf.__version__)
    print("TensorFlow Decision Forests v" + tfdf.__version__)

    # Check if TensorFlow can access the GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Ensure that TensorFlow is not initialized more than once
    if not tf.executing_eagerly():
        tf.compat.v1.reset_default_graph()


    data = pd.read_csv('data/train.csv')
    data = data.drop('Id', axis=1)

    test = pd.read_csv('data/test.csv')
    ids = test.pop('Id')

    # Data exploration ###########################################

    print(data['SalePrice'].describe())
    plt.figure(figsize=(9, 8))
    sns.displot(data['SalePrice'], color='g', bins=100);
    plt.show()

    print("Original data: -------------------")
    df_num = data.select_dtypes(include=['float64', 'int64'])
    df_num.head()

    print(data.head())

    df_num = data.select_dtypes(include=['float64', 'int64'])
    df_num.head()

    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);


    # Data preprocessing ###########################################

    data = preprocess_data(data)
    test = preprocess_data(test)

    # Encode data

    print("Encoded data: -------------------")

    train_encoded = encode_data(data)
    test_encoded = encode_data(test)

    data = train_encoded
    test = test_encoded

    print(train_encoded.head())
    print(train_encoded.info())
    print()

    def split_dataset(dataset, test_ratio=0.30):
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]

    train_ds_pd, valid_ds_pd = split_dataset(data)
    print("{} examples in training, {} examples in testing.".format(
        len(train_ds_pd), len(valid_ds_pd)))

    df_num = data.select_dtypes(include=['float64', 'int64', 'int32', 'float32'])

    df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    plt.show()


    # Random Forest with TensorFlow Decision Forests

    label = 'SalePrice'
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label,
                                                     task=tfdf.keras.Task.REGRESSION,
                                                       max_num_classes=700)
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label,
                                                     task=tfdf.keras.Task.REGRESSION,
                                                       max_num_classes=700)
    # tuner = tfdf.tuner.RandomSearch(num_trials=20, trial_num_threads=3)

    # Hyperparameters to optimize.
    # tuner.choice("max_depth", [4, 5, 7, 16, 32])
    # tuner.choice("num_trees", [50, 100, 200, 500])

    # print(tuner.train_config())

    # model = tfdf.keras.RandomForestModel(tuner=tuner, task=tfdf.keras.Task.REGRESSION,
    #                                      bootstrap_training_dataset=True, bootstrap_size_ratio=1.0,
    #                                      categorical_algorithm='CART', #RANDOM
    #                                      growing_strategy='LOCAL', #BEST_FIRST_GLOBAL
    #                                      honest=False,
    #                                      min_examples=1,
    #                                      missing_value_policy='GLOBAL_IMPUTATION',
    #                                      num_candidate_attributes=0,
    #                                      random_seed=SEED,
    #                                      winner_take_all=True,
    #                                      )

    model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
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



    # Check the model input keys and semantics #################
    sample_inputs = {
        'LotFrontage': tf.constant([80.0], dtype=tf.float32),
        'LotArea': tf.constant([9600], dtype=tf.int64),
    }

    # Check the expected inputs of the model
    model_input_keys = model._normalized_column_keys

    # Check the semantics defined in the model
    model_semantics = model._semantics

    print("Model Input Keys: ", model_input_keys)
    print("Model Semantics: ", model_semantics)

    # Ensure all semantics are in the inputs
    for key in model_semantics.keys():
        if key not in sample_inputs:
            print(f"Missing input for key: {key}")

    # Ensure all inputs are in the semantics
    for key in sample_inputs.keys():
        if key not in model_semantics:
            print(f"Extra input provided for key: {key}")

    # Predict on the test data ###################################

    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_encoded,
        task=tfdf.keras.Task.REGRESSION)


    preds = model.predict(test_ds)
    output = pd.DataFrame({'Id': ids,
                           'SalePrice': preds.squeeze()})

    print(output.head())

    # model.save("./saved_models/")

    sample_submission_df = pd.read_csv('./data/sample_submission.csv')
    sample_submission_df['SalePrice'] = model.predict(test_ds)
    sample_submission_df.to_csv('./results/sample_submission' + str(time.time() / 60) + '.csv', index=False)
    print(sample_submission_df.head())

    
    # Random Forest with scikit-learn

    print("SKLearn Random Forest Regressor Results: -------------------")

    # x_train = np.array(data)
    # y_train = data['SalePrice']
    #
    # x_test = np.array(valid_ds_pd)
    # y_test = valid_ds_pd['SalePrice']
    #
    # rf_reg = RandomForestRegressor(n_estimators=100, random_state=SEED)
    # rf_reg.fit(x_train, y_train)
    #
    # y_pred_train = rf_reg.predict(x_train)
    # y_pred_test = rf_reg.predict(x_test)
    #
    # train_r2 = r2_score(y_train, y_pred_train)
    # test_r2 = r2_score(y_test, y_pred_test)
    # test_mse = mean_squared_error(y_test, y_pred_test)
    # test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    # test_mae = mean_absolute_error(y_test, y_pred_test)
    #
    # print(f'Train R-squared: {train_r2 * 100:.2f}%')
    # print(f'Test R-squared: {test_r2 * 100:.2f}%')
    # print(f'Test Mean Squared Error (MSE): {test_mse:.2f}')
    # print(f'Test Root Mean Squared Error (RMSE): {test_rmse:.2f}')
    # print(f'Test Mean Absolute Error (MAE): {test_mae:.2f}')

    print()




