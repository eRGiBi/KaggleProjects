import pandas as pd
# import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# import tensorflow as tf
# import tensorflow_decision_forests as tfdf

if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
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

    # Binarry encoding
    # large number of unique categories

    # encoder = ce.BinaryEncoder(cols=['Country'])

    # data = encoder.fit_transform(data)

    for col in df_encoded.columns:
        if df_encoded[col].dtype != 'int64' and df_encoded[col].dtype != 'float64':
            print(col, df_encoded[col].dtype)
