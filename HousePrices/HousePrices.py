
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

from ExperimentLogger import ExperimentLogger

import warnings

from HousePrices.ensemble_models import ensemble_model
from HousePrices.models import tf_decision_forests, sklearndf_random_forest, yggdrassil_random_forest, tf_neural_network

warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000


SEED = 476


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
    all_features['YrSold'] = all_features['YrSold'].astype(float)
    all_features['MoSold'] = all_features['MoSold'].astype(float)

    def handle_missing(features):
        """https://www.kaggle.com/code/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition#Feature-Engineering"""

        # The data description states that NA refers to typical ('Typ') values
        features['Functional'] = features['Functional'].fillna('Typ')
        # Replace the missing values in each of the columns below with their mode
        features['Electrical'] = features['Electrical'].fillna("SBrkr")
        features['KitchenQual'] = features['KitchenQual'].fillna("TA")
        features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
        features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
        features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
        # features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
        # features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

        # The data description stats that NA refers to "No Pool"
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

        objects = []
        for i in features.columns:
            if features[i].dtype == object:
                objects.append(i)
        features.update(features[objects].fillna('None'))

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

def outlier_test(data):
    """https://www.kaggle.com/code/nareshbhat/outlier-the-silent-killer"""
    def grubbs_test(x):

        n = len(x)
        mean_x = np.mean(x)
        sd_x = np.std(x)
        numerator = max(abs(x - mean_x))
        g_calculated = numerator / sd_x
        print("Grubbs Calculated Value:", g_calculated)
        t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
        g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
        print("Grubbs Critical Value:", g_critical)

        if g_critical > g_calculated:
            print(
                "From grubbs_test we observe that calculated value is lesser than critical value, Accept null hypothesis and conclude that there is no outliers\n")
        else:
            print(
                "From grubbs_test we observe that calculated value is greater than critical value, Reject null hypothesis and conclude that there is an outliers\n")


    def z_score_outlier(df):
        out = []

        m = np.mean(df)
        sd = np.std(df)
        for i in df:
            z = (i - m) / sd
            if np.abs(z) > 3:
                out.append(i)
        print("Outliers:", out)

    def ZRscore_outlier(df):

        out = []

        med = np.median(df)
        ma = stats.median_absolute_deviation(df)
        for i in df:
            z = (0.6745 * (i - med)) / (np.median(ma))
            if np.abs(z) > 3:
                out.append(i)
        print("Outliers:", out)

    def iqr_outliers(df):

        out = []

        q1 = df.quantile(0.25)
        q3 = df.quantile(0.75)
        iqr = q3 - q1
        Lower_tail = q1 - 1.5 * iqr
        Upper_tail = q3 + 1.5 * iqr
        for i in df:
            if i > Upper_tail or i < Lower_tail:
                out.append(i)
        print("Outliers:", out)

    def winsorization_outliers(df):
        out = []
        q1 = np.percentile(df, 1)
        q3 = np.percentile(df, 99)
        for i in df:
            if i > q3 or i < q1:
                out.append(i)
        print("Outliers:", out)

    def DB_outliers(df):

        outlier_detection = DBSCAN(eps=2, metric='euclidean', min_samples=5)
        clusters = outlier_detection.fit_predict(df.values.reshape(-1, 1))
        data = pd.DataFrame()
        data['cluster'] = clusters
        print(data['cluster'].value_counts().sort_values(ascending=False))

    def iso_outliers(df):
        iso = IsolationForest(behaviour='new', random_state=1, contamination='auto')
        preds = iso.fit_predict(df.values.reshape(-1, 1))
        data = pd.DataFrame()
        data['cluster'] = preds
        print(data['cluster'].value_counts().sort_values(ascending=False))


    print("Outlier detection: -------------------")





def explore_data(data, plot=False, test=False):

    outlier_test(data)


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
        # explore_data(data, plot=True)
        # explore_data(test, plot=False, test=True)

        # sns.heatmap(data.isnull(), cmap='viridis')
        # plt.show()


        # Data preprocessing ###########################################
        print("Preprocessed data: -------------------")

        data, test = preprocess_data(data, test)
        data = pd.DataFrame(data, columns=data.columns)

        # explore_data(data, plot=True)

        # sns.heatmap(data.isnull(), cmap='viridis')
        # plt.show()

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

        vds = ydf.create_vertical_dataset(train_ds_pd, include_all_columns=True)
        print(vds.memory_usage())


        # Train the model #############################################
        match self.algorithm:

            case 'tfdf':
                tf_decision_forests(train_ds_pd, valid_ds_pd, test, ids, exp_name)
            case 'sklearndf':
                sklearndf_random_forest(data, valid_ds_pd, test, ids, exp_name)
            case 'yggdf':
                yggdrassil_random_forest(train_ds_pd, valid_ds_pd, test, ids, exp_name, tune=self.tune)
            case 'ensemble':
                ensemble_model(train_ds_pd, valid_ds_pd, test, ids, exp_name)
            case 'NN':
                tf_neural_network(train_ds_pd, valid_ds_pd, test, ids, exp_name)
            case _:
                print("Invalid algorithm.")


