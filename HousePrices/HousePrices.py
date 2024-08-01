import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
import tensorflow_decision_forests as tfdf

if __name__ == '__main__':

    SEED = 123456

    data = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    print(data.head())

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

    print(data.head())
    print(data['LotShape'].unique())

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

    print(df_encoded.head())

    # Binary encoding
    # large number of unique categories

    # encoder = ce.BinaryEncoder(cols=['Country'])

    # data = encoder.fit_transform(data)

    for col in df_encoded.columns:
        if df_encoded[col].dtype != 'int64' and df_encoded[col].dtype != 'int32' and df_encoded[col].dtype != 'float64':
            print(col, df_encoded[col].dtype)
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'int64' or df_encoded[col].dtype == 'int32':
            df_encoded[col] = df_encoded[col].astype('float64')

    # Random Forest

    print("TensorFlow v" + tf.__version__)
    print("TensorFlow Decision Forests v" + tfdf.__version__)

    tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(df_encoded, label="SalePrice",
                                                       task=tfdf.keras.Task.REGRESSION,
                                                       max_num_classes=700)

    tuner = tfdf.tuner.RandomSearch(num_trials=20, trial_num_threads=3)

    # Hyper-parameters to optimize.
    tuner.choice("max_depth", [4, 5, 7, 16, 32])
    # tuner.choice("num_trees", [50, 100, 200, 500])

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
    model.fit(tf_dataset)

    model.evaluate(test)

    print(model.summary())

    model.save("/saved_models")
    
    # Random Forest with scikit-learn

    print("Random Forest Regressor Results: -------------------")

    x_train = df_encoded.drop('SalePrice', axis=1)
    y_train = df_encoded['SalePrice']

    x_test = test.drop('SalePrice', axis=1)
    y_test = test['SalePrice']

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




