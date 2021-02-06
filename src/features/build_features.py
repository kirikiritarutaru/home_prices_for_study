import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import boxcox1p
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder


def clean_for_reg(
    train: pd.DataFrame,
    test: pd.DataFrame,
    check_missing_data_rate: bool = False,
    check_corr_mat: bool = False,
    check_num_feats: bool = False
) -> (pd.DataFrame, np.ndarray, pd.DataFrame):

    # 予測に使わないのでID削除
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)

    # 外れ値削除
    train = train.drop(
        train[
            (train['GrLivArea'] > 4000) &
            (train['SalePrice'] < 300000)
        ].index
    )

    ntrain = train.shape[0]
    y_train = np.log1p(train['SalePrice'].values)

    # trainとtestに同じ変換をするためにひとまとめにしたall_data作成
    all_data = pd.concat([train, test]).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)

    print(f'all_data size before cleaning is :{all_data.shape}')

    # 欠損値を確認
    if check_missing_data_rate:
        all_data_na = (all_data.isnull().sum()/len(all_data))*100
        all_data_na = all_data_na.drop(
            all_data_na[all_data_na == 0].index
        ).sort_values(ascending=False)[:30]

        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        print(missing_data.head(20))

        if not all_data_na.empty:
            fig, ax = plt.subplots(figsize=(16, 14))
            plt.xticks(rotation=90)
            sns.barplot(x=all_data_na.index, y=all_data_na)
            plt.xlabel('Features')
            plt.ylabel('Percent of missing values')
            plt.title('Percent missing data by feature')
            plt.show()
        else:
            print('all_data_na is empty!')

    if check_corr_mat:
        corrmat = train.corr()
        plt.subplots(figsize=(12, 9))
        sns.heatmap(corrmat, vmax=0.9, square=True)
        plt.show()

    # 欠損値処理
    for col in (
        'PoolQC',
        'MiscFeature',
        'Alley',
        'Fence',
        'FireplaceQu',
        'GarageType',
        'GarageFinish',
        'GarageQual',
        'GarageCond',
        'BsmtQual',
        'BsmtCond',
        'BsmtExposure',
        'BsmtFinType1',
        'BsmtFinType2',
        'MasVnrType',
        'MSSubClass'
    ):
        all_data[col] = all_data[col].fillna('None')

    for col in (
        'GarageYrBlt',
        'GarageArea',
        'GarageCars',
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        'BsmtFullBath',
        'BsmtHalfBath',
        'MasVnrArea'
    ):
        all_data[col] = all_data[col].fillna(0)

    all_data['LotFrontage'] = (
        all_data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )
    )

    for col in (
        'MSZoning',
        'Electrical',
        'KitchenQual',
        'Exterior1st',
        'Exterior2nd',
        'SaleType'
    ):
        all_data[col] = (
            all_data[col].fillna(all_data[col].mode()[0])
        )

    all_data = all_data.drop(['Utilities'], axis=1)

    all_data["Functional"] = all_data["Functional"].fillna("Typ")

    # 欠損値を確認
    if check_missing_data_rate:
        all_data_na = (all_data.isnull().sum()/len(all_data))*100
        all_data_na = all_data_na.drop(
            all_data_na[all_data_na == 0].index
        ).sort_values(ascending=False)[:30]

        missing_data = pd.DataFrame({'Missing Ratio': all_data_na})
        print(missing_data.head(20))

        if not all_data_na.empty:
            fig, ax = plt.subplots(figsize=(16, 14))
            plt.xticks(rotation=90)
            sns.barplot(x=all_data_na.index, y=all_data_na)
            plt.xlabel('Features')
            plt.ylabel('Percent of missing values')
            plt.title('Percent missing data by feature')
            plt.show()
        else:
            print('all_data_na is empty!')

    # 数値変数→カテゴリカル変数
    for col in ('MSSubClass', 'OverallCond', 'YrSold', 'MoSold'):
        all_data[col] = all_data[col].astype(str)

    # LabelEncode
    for col in (
        'FireplaceQu',
        'BsmtQual',
        'BsmtCond',
        'GarageQual',
        'GarageCond',
        'ExterQual',
        'ExterCond',
        'HeatingQC',
        'PoolQC',
        'KitchenQual',
        'BsmtFinType1',
        'BsmtFinType2',
        'Functional',
        'Fence',
        'BsmtExposure',
        'GarageFinish',
        'LandSlope',
        'LotShape',
        'PavedDrive',
        'Street',
        'Alley',
        'CentralAir',
        'MSSubClass',
        'OverallCond',
        'YrSold',
        'MoSold'
    ):
        all_data[col] = LabelEncoder().fit_transform(
            list(all_data[col].values)
        )

    # 総面積の特徴追加
    all_data['TotalSF'] = (
        all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
    )

    numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
    skewed_feats = all_data[numeric_feats].apply(
        lambda x: skew(x.dropna())
    ).sort_values(ascending=False)

    skewness = pd.DataFrame({'Skew': skewed_feats})

    if check_num_feats:
        print('Skew in numerical features: ')
        print(skewness.head(10))

    # 歪んだ特徴のBox-Cox変換
    skewness = skewness[abs(skewness) > 0.75]
    print("There are {} skewed numerical features to Box Cox transform".format(
        skewness.shape[0]))
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)

    # カテゴリカル変数をダミー変数へ
    all_data = pd.get_dummies(all_data)
    print(f'all_data size is : {all_data.shape}')

    train = all_data[:ntrain]
    test = all_data[ntrain:]

    return train, y_train, test
